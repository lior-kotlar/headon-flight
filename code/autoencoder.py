"""
Convolutional Autoencoder for individual wingbeat cycle reconstruction.

Input/Output shape: (B, 6, n) where 6 channels are [S_phi, S_theta, S_psi, A_phi, A_theta, A_psi]
and n is the number of time samples in one wingbeat (~61-75).
"""
import argparse
import copy
import itertools
import json
import os
import shutil
import sys
import time
from datetime import datetime

import matplotlib
matplotlib.use('Agg')  # non-interactive backend — safe for headless servers
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from transform_data import (
    _wingbeat_peaks,
    _segment_to_sa,
    _sa_to_segment,
    _segment_to_single_wing,
    _single_wing_to_segment,
    _cubic_resample,
    SA_PHYSICAL_SCALE,
    SINGLE_WING_PHYSICAL_SCALE,
    single_wing_template_path,
    build_fixed_len_dataset_from_disk,
    fixed_len_dataset_is_valid,
    fixed_len_dataset_path,
    fixed_len_sidecar_path,
)
from data_handling.bucket_eval import (
    WING_ANGLE_LABELS as _WING_ANGLE_LABELS,
    WING_ANGLE_SCALE  as _WING_ANGLE_SCALE,
    sa_to_lr_norm     as _sa_to_lr_norm,
    channel_rmse_to_degrees as _channel_rmse_to_degrees,
    format_rmse_degrees     as _format_rmse_degrees,
    get_representation,
    evaluate_by_maneuver_bucket,
    plot_per_phase_error,
    plot_phase_range_distributions,
    DEFAULT_PHASE_RANGES,
)
from data_handling.maneuver_scoring import expand_score_axis
from plot_fixed_len_vs_performance import plot_fixed_len_vs_performance

# Architecture constants
_BASE_CHANNELS  = 128  # default channels at the deepest conv layer; overridable per model
_BOTTLENECK_LEN = 12    # temporal length retained at the encoder bottleneck
_NORM_GROUPS    = 8    # GroupNorm groups; must divide every conv output channel count


def _validate_base_channels(base_channels: int) -> None:
    """`base_channels` sets the deepest layer; intermediate layers are base/2 and base/4.
    All three must be divisible by _NORM_GROUPS for GroupNorm, so base must be a multiple of 4*_NORM_GROUPS."""
    if base_channels <= 0 or base_channels % (4 * _NORM_GROUPS) != 0:
        raise ValueError(
            f"base_channels={base_channels} must be a positive multiple of {4 * _NORM_GROUPS} "
            f"(smallest layer = base_channels//4 must be divisible by _NORM_GROUPS={_NORM_GROUPS})."
        )


def _make_optimizer(name: str, params, lr: float, weight_decay: float = 0.0) -> torch.optim.Optimizer:
    """Returns an optimizer instance for the given name."""
    key = name.lower()
    if key == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if key == 'adamw':
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if key == 'sgd':
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    if key == 'rmsprop':
        return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer '{name}'. Options: adam, adamw, sgd, rmsprop")


def _make_activation(name: str) -> nn.Module:
    """Returns a fresh activation module instance for the given name."""
    table = {
        'gelu':      nn.GELU,
        'silu':      nn.SiLU,
        'swish':     nn.SiLU,
        'elu':       nn.ELU,
        'relu':      nn.ReLU,
        'leakyrelu': nn.LeakyReLU,
        'mish':      nn.Mish,
        'tanh':      nn.Tanh,
    }
    key = name.lower()
    if key not in table:
        raise ValueError(f"Unknown activation '{name}'. Options: {list(table)}")
    return table[key]()


class WingbeatEncoder(nn.Module):
    """
    Dilated 1D-conv encoder. Channels double each layer (base/4 → base/2 → base).
    Stride-1 dilated convs (dilation 1, 2, 4) preserve full temporal resolution through
    the conv stack — high-frequency content reaches the pool intact. Receptive field in
    input coordinates is ~21 samples, same as a strided version. The final AdaptiveAvgPool
    reduces to `bottleneck_len`.
    """
    def __init__(
        self,
        latent_dim: int,
        in_channels: int = 6,
        activation: str = 'gelu',
        dropout: float = 0.0,
        base_channels: int = _BASE_CHANNELS,
        bottleneck_len: int = _BOTTLENECK_LEN,
    ):
        super().__init__()
        _validate_base_channels(base_channels)
        c1 = base_channels // 4
        c2 = base_channels // 2
        c3 = base_channels
        self.base_channels  = base_channels
        self.bottleneck_len = bottleneck_len

        # For kernel=k, dilation=d, stride=1, 'same' padding is d*(k-1)//2.
        self.convs = nn.Sequential(
            nn.Conv1d(in_channels, c1, kernel_size=5, padding=2, padding_mode='replicate'),
            nn.GroupNorm(_NORM_GROUPS, c1),
            _make_activation(activation),
            nn.Conv1d(c1, c2, kernel_size=5, padding=4, dilation=2, padding_mode='replicate'),
            nn.GroupNorm(_NORM_GROUPS, c2),
            _make_activation(activation),
            nn.Conv1d(c2, c3, kernel_size=3, padding=4, dilation=4, padding_mode='replicate'),
            nn.GroupNorm(_NORM_GROUPS, c3),
            _make_activation(activation),
        )

        self.pool    = nn.AdaptiveAvgPool1d(bottleneck_len)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(c3 * bottleneck_len, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convs(x)             # (B, c3, ~n)  (full temporal resolution preserved)
        x = self.pool(x)              # (B, c3, bottleneck_len)
        x = x.flatten(start_dim=1)    # (B, c3 * bottleneck_len)
        x = self.dropout(x)
        return self.fc(x)             # (B, latent_dim)


class WingbeatDecoder(nn.Module):
    """
    Learnable-upsampler decoder, always producing the same fixed `output_len`. Structure:
      fc → view → two Upsample(2×) + Conv1d blocks (4× learnable upsample) → F.interpolate
      to handle the fractional remainder up to output_len → learnable refiner (Conv1d+GN+act)
      → channel-reduction conv stack.

    The nearest+conv upsampler avoids ConvTranspose1d checkerboard artifacts while keeping
    upsampling learnable. The refiner after F.interpolate gives the model a chance to sharpen
    what linear interpolation smoothed before the channel reduction kicks in.
    """
    def __init__(
        self,
        latent_dim: int,
        out_channels: int = 6,
        activation: str = 'gelu',
        dropout: float = 0.0,
        base_channels: int = _BASE_CHANNELS,
        bottleneck_len: int = _BOTTLENECK_LEN,
        decoder_kernel_size: int = 5,
        output_len: int = 80,
    ):
        super().__init__()
        _validate_base_channels(base_channels)
        if decoder_kernel_size < 1 or decoder_kernel_size % 2 == 0:
            raise ValueError(
                f"decoder_kernel_size={decoder_kernel_size} must be a positive odd integer "
                "so 'same' padding is well-defined."
            )
        # Channel reduction halves each layer: base → base/2 → base/4 → out.
        c3 = base_channels
        c2 = base_channels // 2
        c1 = base_channels // 4
        self.base_channels  = base_channels
        self.bottleneck_len = bottleneck_len
        self.output_len     = int(output_len)

        kw  = decoder_kernel_size
        pad = kw // 2  # 'same' padding for stride=1

        self.fc      = nn.Linear(latent_dim, c3 * bottleneck_len)
        self.dropout = nn.Dropout(dropout)

        # Two learnable upsample stages. Stays at c3 channels so the downstream conv stack
        # is unchanged. Each stage doubles the temporal dim → 4× total.
        # No GroupNorm in the upsampler — empirically regresses val loss ~3% (likely
        # because the model is near its capacity ceiling and the extra activation constraint
        # costs more than it stabilizes).
        self.upsampler = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(c3, c3, kernel_size=3, padding=1, padding_mode='replicate'),
            _make_activation(activation),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(c3, c3, kernel_size=3, padding=1, padding_mode='replicate'),
            _make_activation(activation),
        )

        # Learnable refinement applied immediately after the F.interpolate(linear).
        # Gives the model a chance to sharpen / correct linearly-smoothed transitions
        # before the channel-reduction conv stack.
        self.refiner = nn.Sequential(
            nn.Conv1d(c3, c3, kernel_size=kw, padding=pad, padding_mode='replicate'),
            nn.GroupNorm(_NORM_GROUPS, c3),
            _make_activation(activation),
        )

        self.convs = nn.Sequential(
            nn.Conv1d(c3, c2, kernel_size=kw, padding=pad, padding_mode='replicate'),
            nn.GroupNorm(_NORM_GROUPS, c2),
            _make_activation(activation),
            nn.Conv1d(c2, c1, kernel_size=kw, padding=pad, padding_mode='replicate'),
            nn.GroupNorm(_NORM_GROUPS, c1),
            _make_activation(activation),
            nn.Conv1d(c1, out_channels, kernel_size=kw, padding=pad, padding_mode='replicate'),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)                                                                    # (B, C*L)
        x = self.dropout(x)
        x = x.view(x.size(0), self.base_channels, self.bottleneck_len)                    # (B, C, L)
        x = self.upsampler(x)                                                             # (B, C, 4L)
        if x.size(-1) != self.output_len:
            # Final fractional step (4·bottleneck_len → output_len) when the two don't align.
            x = F.interpolate(x, size=self.output_len, mode='linear', align_corners=False)
        x = self.refiner(x)                                                               # (B, C, output_len)
        return self.convs(x)                                                              # (B, 6, output_len)


class WingbeatAutoencoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 16,
        in_channels: int = 6,
        activation: str = 'gelu',
        dropout: float = 0.0,
        base_channels: int = _BASE_CHANNELS,
        bottleneck_len: int = _BOTTLENECK_LEN,
        decoder_kernel_size: int = 5,
        output_len: int = 80,
    ):
        super().__init__()
        self.encoder = WingbeatEncoder(
            latent_dim, in_channels, activation, dropout, base_channels, bottleneck_len,
        )
        self.decoder = WingbeatDecoder(
            latent_dim, in_channels, activation, dropout, base_channels, bottleneck_len,
            decoder_kernel_size, output_len,
        )
        self.latent_dim          = latent_dim
        self.in_channels         = in_channels
        self.out_channels        = in_channels
        self.activation          = activation
        self.dropout             = dropout
        self.base_channels       = base_channels
        self.bottleneck_len      = bottleneck_len
        self.decoder_kernel_size = decoder_kernel_size
        self.output_len          = int(output_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class FixedLenWingbeatDataset(Dataset):
    """
    Wraps a pre-built fixed-length SA wingbeat array of shape (N, 6, L).
    All wingbeats share the same length L (already CubicSpline-resampled by
    build_fixed_len_dataset) and are pre-normalized by SA_PHYSICAL_SCALE.

    Each sample is a (6, L) tensor: [S_phi, S_theta, S_psi, A_phi, A_theta, A_psi].
    No collate function is needed — default collation stacks to (B, 6, L).
    """

    def __init__(self, sa_wingbeats: np.ndarray):
        if sa_wingbeats.ndim != 3:
            raise ValueError(f"wingbeats must be (N, C, L); got {sa_wingbeats.shape}")
        # One torch tensor lets the DataLoader hand out views without per-sample copies.
        self.samples = torch.from_numpy(np.ascontiguousarray(sa_wingbeats, dtype=np.float32))
        self.L = int(sa_wingbeats.shape[-1])

    def __len__(self) -> int:
        return self.samples.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.samples[idx]


def _mse_fixed(
    recon: torch.Tensor,
    target: torch.Tensor,
    channel_weight: torch.Tensor | None = None,
    endpoint_weight: float = 1.0,
    endpoint_samples: int = 3,
) -> torch.Tensor:
    """
    Plain MSE over all (B, C, L) entries, with optional per-channel and endpoint
    weighting. Same semantics as the old masked version but without the mask —
    every sample has the same length L now.

    `channel_weight` (C,): per-channel multiplier on squared errors. Used to
        up-weight A channels so the model can't trivially minimize loss by
        predicting symmetric (A=0) outputs.
    `endpoint_weight`: multiplier applied to the first and last `endpoint_samples`
        time-steps of each wingbeat (the seam region between consecutive beats).
        Pass 1.0 to disable.

    Denominator stays unweighted element count so the weights are *relative*
    scales rather than re-normalizations.
    """
    sq_err = (recon - target) ** 2                              # (B, C, L)
    if channel_weight is not None:
        sq_err = sq_err * channel_weight.view(1, -1, 1).to(sq_err.device, sq_err.dtype)
    if endpoint_weight != 1.0:
        L = sq_err.size(-1)
        k = min(endpoint_samples, L)
        pos_weight = sq_err.new_ones((1, 1, L))
        pos_weight[:, :, :k]  = endpoint_weight
        pos_weight[:, :, -k:] = endpoint_weight
        sq_err = sq_err * pos_weight
    return sq_err.mean()


def _cycle_consistency_loss(recon: torch.Tensor) -> torch.Tensor:
    """
    Mean squared difference between each wingbeat's first sample and its last
    sample, across all channels. Pulls `recon[:, :, 0]` toward `recon[:, :, -1]`
    so the reconstructed SA residual is cyclic — and since the template is
    cyclic by construction, this makes the physical seam between consecutive
    wingbeats continuous.
    """
    first = recon[:, :, 0]
    last  = recon[:, :, -1]
    return ((first - last) ** 2).mean()


def _derivative_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    MSE between the time-derivatives of recon and target.

    Plain MSE is dominated by large smooth features and is nearly blind to small
    high-frequency wiggles — a reconstruction that smooths them out has tiny extra
    error. The derivative magnifies high-frequency content (each wiggle adds two
    opposite-sign diffs), so the model is forced to render it.
    """
    d_recon  = recon[:, :, 1:] - recon[:, :, :-1]
    d_target = target[:, :, 1:] - target[:, :, :-1]
    return ((d_recon - d_target) ** 2).mean()


def train_autoencoder(
    model: WingbeatAutoencoder,
    train_dataset: FixedLenWingbeatDataset,
    val_dataset: FixedLenWingbeatDataset | None = None,
    n_epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 32,
    device: str = "cpu",
    loss_fig_path: str | None = None,
    optimizer_name: str = 'adam',
    weight_decay: float = 0.0,
    channel_loss_weight: tuple[float, float] | list[float] | None = None,  # (alpha_s, alpha_a)
    cycle_weight: float = 0.0,
    endpoint_weight: float = 1.0,
    endpoint_samples: int = 3,
    derivative_weight: float = 0.0,
    restart_check_epoch: int | None = None,
    restart_threshold: float | None = None,
    representation: str = 'sa',
) -> tuple[list[float], list[float], dict, bool, np.ndarray | None]:
    """
    Trains the autoencoder over fixed-length wingbeats (shape (C, L), C from the
    representation: 6 for 'sa', 3 for 'single_wing').

    If both `restart_check_epoch` and `restart_threshold` are set, training aborts
    early once epoch `restart_check_epoch` completes if the best val loss so far is
    above `restart_threshold`. The caller can then decide to re-init and retry.

    Returns:
        train_losses:      per-epoch average training loss
        val_losses:        per-epoch average validation loss (empty list if no val_dataset)
        best_state:        state_dict of the epoch with the lowest monitored loss
        was_stuck:         True if training was aborted early by the plateau detector
        best_val_rmse_deg: (6,) array of per-channel RMSE in degrees from the
                           best-monitor epoch (None if no val_loader was used)
    """
    # FixedLenWingbeatDataset returns (6, L) tensors directly; default collate stacks
    # them to (B, 6, L) with no padding needed.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = (
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        if val_dataset else None
    )

    optimizer = _make_optimizer(optimizer_name, model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, patience=4, factor=0.5)

    # Representation spec: drives channel count, the conversion into per-angle-normalized
    # space for the val RMSE, and the per-channel scale used for RMSE-degrees.
    repr_spec   = get_representation(representation)
    n_channels  = repr_spec['n_channels']
    to_physical = repr_spec['to_physical_norm']
    chan_scale  = repr_spec['channel_scale']
    chan_labels = repr_spec['channel_labels']

    # channel_loss_weight is (alpha_s, alpha_a): broadcast to the 6 S/A channels
    # [S_phi, S_theta, S_psi, A_phi, A_theta, A_psi] as [αs, αs, αs, αa, αa, αa].
    # Only meaningful for the 'sa' representation (single_wing has no S/A split, so the
    # A-channel up-weighting hack is dropped); ignored otherwise.
    channel_weight_tensor = None
    if channel_loss_weight is not None and representation == 'sa':
        if len(channel_loss_weight) != 2:
            raise ValueError(
                f"channel_loss_weight must be a 2-element sequence (alpha_s, alpha_a); "
                f"got {channel_loss_weight}"
            )
        alpha_s, alpha_a = float(channel_loss_weight[0]), float(channel_loss_weight[1])
        channel_weight_tensor = torch.tensor(
            [alpha_s, alpha_s, alpha_s, alpha_a, alpha_a, alpha_a], dtype=torch.float32,
        )
    elif channel_loss_weight is not None and representation != 'sa':
        print(f"  (channel_loss_weight ignored for representation={representation!r})", flush=True)

    model.to(device)
    train_losses, val_losses = [], []
    val_rmse_deg_history: list[np.ndarray] = []  # per-epoch per-channel RMSE in degrees
    best_val_rmse_deg: np.ndarray | None = None
    best_monitor = float('inf')
    best_state = None

    for epoch in range(n_epochs):
        model.train()
        total_train = 0.0
        for x in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{n_epochs}", leave=False, file=sys.stdout, disable=not sys.stdout.isatty()):
            x = x.to(device)
            recon = model(x)
            mse = _mse_fixed(
                recon, x,
                channel_weight   = channel_weight_tensor,
                endpoint_weight  = endpoint_weight,
                endpoint_samples = endpoint_samples,
            )
            loss = mse
            if cycle_weight > 0.0:
                loss = loss + cycle_weight * _cycle_consistency_loss(recon)
            if derivative_weight > 0.0:
                loss = loss + derivative_weight * _derivative_loss(recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train += loss.item()

        avg_train = total_train / len(train_loader)
        train_losses.append(avg_train)

        avg_val = None
        val_rmse_deg = None
        if val_loader:
            model.eval()
            total_val = 0.0
            sse_per_channel = torch.zeros(n_channels, device=device, dtype=torch.float64)
            n_elements_per_channel = 0
            with torch.no_grad():
                for x in val_loader:
                    x = x.to(device)
                    recon = model(x)
                    # Val loss is unweighted (no channel/endpoint/cycle terms) so it
                    # stays comparable across runs with different loss-weight settings.
                    total_val += _mse_fixed(recon, x).item()
                    # Per-channel SSE in the representation's per-angle-normalized space
                    # ('sa' → L/R residuals; 'single_wing' → identity), so RMSE-degrees
                    # corresponds to per-wing physical angles.
                    recon_p  = to_physical(recon)
                    target_p = to_physical(x)
                    sq_err = (recon_p - target_p).double() ** 2          # (B, C, L)
                    sse_per_channel += sq_err.sum(dim=(0, 2))
                    n_elements_per_channel += x.size(0) * x.size(2)
            avg_val = total_val / len(val_loader)
            val_losses.append(avg_val)
            mse_per_channel = (sse_per_channel / max(n_elements_per_channel, 1)).cpu().numpy()
            val_rmse_deg = _channel_rmse_to_degrees(mse_per_channel, scale=chan_scale)
            val_rmse_deg_history.append(val_rmse_deg)

        monitor = avg_val if avg_val is not None else avg_train
        scheduler.step(monitor)

        if monitor < best_monitor:
            best_monitor = monitor
            best_state = copy.deepcopy(model.state_dict())
            if val_rmse_deg is not None:
                best_val_rmse_deg = val_rmse_deg.copy()

        msg = f"Epoch {epoch + 1:>4}/{n_epochs}  train={avg_train:.6f}"
        if avg_val is not None:
            msg += f"  val={avg_val:.6f}  {_format_rmse_degrees(val_rmse_deg, labels=chan_labels)}"
        print(msg, flush=True)

        # Early-stuck detection: if at the configured check-point the best val loss
        # so far is above the threshold, this run is plateaued at a bad minimum.
        # Abort early so the caller can re-init and retry.
        if (
            restart_check_epoch is not None
            and restart_threshold is not None
            and (epoch + 1) == restart_check_epoch
            and best_monitor > restart_threshold
        ):
            print(
                f"STUCK detected at epoch {epoch + 1}: best val loss {best_monitor:.6f} "
                f"> threshold {restart_threshold:.6f}. Aborting for restart.",
                flush=True,
            )
            if loss_fig_path:
                _save_loss_figure(train_losses, val_losses, loss_fig_path)
            return train_losses, val_losses, best_state, True, best_val_rmse_deg

    if loss_fig_path:
        _save_loss_figure(train_losses, val_losses, loss_fig_path)

    return train_losses, val_losses, best_state, False, best_val_rmse_deg

def _save_loss_figure(train_losses: list[float], val_losses: list[float], path: str) -> None:
    epochs = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train_losses, label='Train', linewidth=2)
    if val_losses:
        ax.plot(epochs, val_losses, label='Validation', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Autoencoder Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.4)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Loss curve saved → {path}", flush=True)

def load_decoder(path: str, device: str = "cpu") -> WingbeatDecoder:
    """
    Loads a saved decoder from a checkpoint file. Output length is fixed at the
    value recorded in the checkpoint; callers resample to native duration themselves.

    Usage:
        decoder = load_decoder("data/models/autoencoder/best_decoder.pt")
        wings_L = decoder(z)   # shape (B, 6, decoder.output_len)
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    decoder = WingbeatDecoder(
        latent_dim=ckpt['latent_dim'],
        out_channels=ckpt.get('out_channels', 6),
        activation=ckpt.get('activation', 'gelu'),
        dropout=ckpt.get('dropout', 0.0),
        base_channels=ckpt.get('base_channels', _BASE_CHANNELS),
        bottleneck_len=ckpt.get('bottleneck_len', _BOTTLENECK_LEN),
        decoder_kernel_size=ckpt.get('decoder_kernel_size', 5),
        output_len=ckpt['output_len'],
    )
    decoder.load_state_dict(ckpt['state_dict'])
    decoder.to(device)
    return decoder


def _plot_reconstructed_trajectory(
    model: WingbeatAutoencoder,
    val_trajs: list,
    template: np.ndarray,
    save_path: str,
    device: str,
    n_beats: int = 5,
    seed: int | None = None,
) -> None:
    """
    Picks n_beats consecutive wingbeats from a random validation trajectory,
    reconstructs each through the autoencoder, and plots original vs reconstruction
    vs template as a single continuous signal.
    """
    rng = np.random.default_rng(seed)

    # Find a trajectory that has enough consecutive wingbeats
    candidates = [(traj, _wingbeat_peaks(traj)) for traj in val_trajs]
    candidates = [(traj, peaks) for traj, peaks in candidates if len(peaks) - 1 >= n_beats]

    if not candidates:
        print(f"No validation trajectory has {n_beats} consecutive wingbeats — skipping reconstruction plot.", flush=True)
        return

    selected_traj_idx = rng.integers(len(candidates))
    traj, peaks = candidates[selected_traj_idx]
    max_start   = len(peaks) - 1 - n_beats
    start_beat  = int(rng.integers(0, max_start + 1))

    model.eval()
    model.to(device)

    orig_parts, recon_parts, tmpl_parts = [], [], []

    L = int(model.output_len)
    for i in range(start_beat, start_beat + n_beats):
        start, end = int(peaks[i]), int(peaks[i + 1])
        segment = traj[start:end]                                    # (n, 6)
        n       = segment.shape[0]

        # Template matched to this segment length (S=A=0 → hat=0 → output = matched template)
        tmpl_matched = _sa_to_segment(np.zeros((n, 6), dtype=np.float32), template)

        # Native-length SA → CubicSpline to L → autoencoder → CubicSpline back to native n.
        # The model operates in SA_PHYSICAL_SCALE-normalized space; physical units
        # are restored when we invert the SA transform for plotting.
        sa_native = _segment_to_sa(segment, template) / SA_PHYSICAL_SCALE     # (n, 6)
        sa_L      = _cubic_resample(sa_native, L)                              # (L, 6)
        x         = torch.from_numpy(sa_L.T).unsqueeze(0).to(device)           # (1, 6, L)
        with torch.no_grad():
            recon_sa_L = model(x).squeeze(0).T.cpu().numpy()                   # (L, 6) normalized
        recon_sa_native = _cubic_resample(recon_sa_L, n)                       # (n, 6) normalized
        reconstruction  = _sa_to_segment(recon_sa_native * SA_PHYSICAL_SCALE, template)

        orig_parts.append(segment)
        recon_parts.append(reconstruction)
        tmpl_parts.append(tmpl_matched)

    original      = np.concatenate(orig_parts,  axis=0)
    reconstruction = np.concatenate(recon_parts, axis=0)
    template_line  = np.concatenate(tmpl_parts,  axis=0)
    x_axis         = np.arange(original.shape[0])

    # Vertical lines at wingbeat boundaries
    boundaries = np.cumsum([0] + [p.shape[0] for p in orig_parts])

    angle_labels = ['Stroke φ [rad]', 'Deviation θ [rad]', 'Rotation ψ [rad]']
    left_cols    = [0, 1, 2]
    right_cols   = [3, 4, 5]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"Best Autoencoder — Reconstruction of {n_beats} Consecutive Wingbeats", fontsize=14)

    for ax, label, lc, rc in zip(axes, angle_labels, left_cols, right_cols):
        ax.plot(x_axis, original[:, lc],        color='blue', lw=2,   ls='-',  label='Left — original')
        ax.plot(x_axis, reconstruction[:, lc],  color='blue', lw=1.5, ls='--', label='Left — reconstruction')
        ax.plot(x_axis, template_line[:, lc],   color='blue', lw=1,   ls=':',  alpha=0.5, label='Left — template')

        ax.plot(x_axis, original[:, rc],        color='red',  lw=2,   ls='-',  label='Right — original')
        ax.plot(x_axis, reconstruction[:, rc],  color='red',  lw=1.5, ls='--', label='Right — reconstruction')
        ax.plot(x_axis, template_line[:, rc],   color='red',  lw=1,   ls=':',  alpha=0.5, label='Right — template')

        for b in boundaries[1:-1]:
            ax.axvline(x=b, color='gray', lw=0.8, ls='--', alpha=0.4)

        ax.set_ylabel(label)
        ax.grid(True, alpha=0.4)

    axes[0].legend(loc='upper right', fontsize=8, ncol=2)
    axes[2].set_xlabel('Sample Index')

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Reconstruction plot saved → {save_path}", flush=True)

    # --- Interactive plotly HTML version ---
    html_path = os.path.splitext(save_path)[0] + ".html"
    _save_reconstruction_plotly(
        x_axis, original, reconstruction, template_line, boundaries,
        angle_labels, left_cols, right_cols, n_beats, html_path,
    )

def _save_reconstruction_plotly(
    x_axis: np.ndarray,
    original: np.ndarray,
    reconstruction: np.ndarray,
    template_line: np.ndarray,
    boundaries: np.ndarray,
    angle_labels: list,
    left_cols: list,
    right_cols: list,
    n_beats: int,
    html_path: str,
) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("plotly not installed — skipping interactive HTML plot.", flush=True)
        return

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=[a.split(' [')[0] for a in angle_labels],
        vertical_spacing=0.06,
    )

    series_specs = [
        ('Left — original',        'blue', 2.0,  'solid', 1.0),
        ('Left — reconstruction',  'blue', 1.5,  'dash',  1.0),
        ('Left — template',        'blue', 1.0,  'dot',   0.5),
        ('Right — original',       'red',  2.0,  'solid', 1.0),
        ('Right — reconstruction', 'red',  1.5,  'dash',  1.0),
        ('Right — template',       'red',  1.0,  'dot',   0.5),
    ]

    for row, (lc, rc, label) in enumerate(zip(left_cols, right_cols, angle_labels), start=1):
        series_data = [
            original[:, lc], reconstruction[:, lc], template_line[:, lc],
            original[:, rc], reconstruction[:, rc], template_line[:, rc],
        ]
        for (name, color, width, dash, opacity), y in zip(series_specs, series_data):
            fig.add_trace(
                go.Scatter(
                    x=x_axis, y=y, mode='lines', name=name,
                    line=dict(color=color, width=width, dash=dash),
                    opacity=opacity,
                    legendgroup=name,
                    showlegend=(row == 1),  # one legend entry per series, not per subplot
                ),
                row=row, col=1,
            )
        fig.update_yaxes(title_text=label, row=row, col=1)

    # Wingbeat boundaries as vertical lines on every subplot
    for b in boundaries[1:-1]:
        for row in (1, 2, 3):
            fig.add_vline(x=float(b), line=dict(color='gray', width=1, dash='dash'),
                          opacity=0.4, row=row, col=1)

    fig.update_xaxes(title_text='Sample Index', row=3, col=1)
    fig.update_layout(
        title=f"Best Autoencoder — Reconstruction of {n_beats} Consecutive Wingbeats",
        height=900, width=1200,
        hovermode='x unified',
        template='plotly_white',
    )

    os.makedirs(os.path.dirname(os.path.abspath(html_path)), exist_ok=True)
    fig.write_html(html_path, include_plotlyjs='cdn')
    print(f"Interactive reconstruction plot saved → {html_path}", flush=True)


def _plot_reconstructed_trajectory_single_wing(
    model: WingbeatAutoencoder,
    val_trajs: list,
    template3: np.ndarray,
    save_path: str,
    device: str,
    n_beats: int = 5,
    seed: int | None = None,
) -> None:
    """
    Single-wing analogue of _plot_reconstructed_trajectory. Picks n_beats consecutive
    wingbeats from a random val trajectory; for each, the LEFT and RIGHT wings are
    each independently residual-transformed against the single-wing template, run
    through the 3-channel autoencoder, and inverted back to wing angles. Left is
    drawn blue, right red — directly comparable to the 6-ch model's reconstruction plot.
    """
    rng = np.random.default_rng(seed)
    candidates = [(traj, _wingbeat_peaks(traj)) for traj in val_trajs]
    candidates = [(traj, peaks) for traj, peaks in candidates if len(peaks) - 1 >= n_beats]
    if not candidates:
        print(f"No validation trajectory has {n_beats} consecutive wingbeats — skipping reconstruction plot.", flush=True)
        return

    traj, peaks = candidates[rng.integers(len(candidates))]
    max_start  = len(peaks) - 1 - n_beats
    start_beat = int(rng.integers(0, max_start + 1))

    model.eval()
    model.to(device)
    L = int(model.output_len)

    def _reconstruct_wing(wing_native: np.ndarray, n: int) -> np.ndarray:
        """wing_native: (n, 3) raw single-wing angles → (n, 3) reconstruction."""
        res_native = _segment_to_single_wing(wing_native, template3) / SINGLE_WING_PHYSICAL_SCALE  # (n,3)
        res_L      = _cubic_resample(res_native, L)                                                 # (L,3)
        x          = torch.from_numpy(res_L.T).unsqueeze(0).to(device)                              # (1,3,L)
        with torch.no_grad():
            recon_res_L = model(x).squeeze(0).T.cpu().numpy()                                       # (L,3)
        recon_res_native = _cubic_resample(recon_res_L, n) * SINGLE_WING_PHYSICAL_SCALE             # (n,3)
        return _single_wing_to_segment(recon_res_native, template3)                                 # (n,3)

    orig_parts, recon_parts, tmpl_parts = [], [], []
    for i in range(start_beat, start_beat + n_beats):
        start, end = int(peaks[i]), int(peaks[i + 1])
        segment = traj[start:end]                                  # (n, 6)
        n       = segment.shape[0]
        left_recon  = _reconstruct_wing(segment[:, 0:3], n)
        right_recon = _reconstruct_wing(segment[:, 3:6], n)
        tmpl_matched = _single_wing_to_segment(np.zeros((n, 3), dtype=np.float32), template3)  # (n,3)

        orig_parts.append(segment)
        recon_parts.append(np.concatenate([left_recon, right_recon], axis=1))   # (n, 6)
        tmpl_parts.append(np.concatenate([tmpl_matched, tmpl_matched], axis=1))  # (n, 6)

    original       = np.concatenate(orig_parts,  axis=0)
    reconstruction = np.concatenate(recon_parts, axis=0)
    template_line  = np.concatenate(tmpl_parts,  axis=0)
    x_axis         = np.arange(original.shape[0])
    boundaries     = np.cumsum([0] + [p.shape[0] for p in orig_parts])

    angle_labels = ['Stroke φ [rad]', 'Deviation θ [rad]', 'Rotation ψ [rad]']
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"Single-Wing Autoencoder — Reconstruction of {n_beats} Consecutive Wingbeats", fontsize=14)
    for ax, label, lc, rc in zip(axes, angle_labels, [0, 1, 2], [3, 4, 5]):
        ax.plot(x_axis, original[:, lc],       color='blue', lw=2,   ls='-',  label='Left — original')
        ax.plot(x_axis, reconstruction[:, lc], color='blue', lw=1.5, ls='--', label='Left — reconstruction')
        ax.plot(x_axis, template_line[:, lc],  color='blue', lw=1,   ls=':',  alpha=0.5, label='Left — template')
        ax.plot(x_axis, original[:, rc],       color='red',  lw=2,   ls='-',  label='Right — original')
        ax.plot(x_axis, reconstruction[:, rc], color='red',  lw=1.5, ls='--', label='Right — reconstruction')
        ax.plot(x_axis, template_line[:, rc],  color='red',  lw=1,   ls=':',  alpha=0.5, label='Right — template')
        for b in boundaries[1:-1]:
            ax.axvline(x=b, color='gray', lw=0.8, ls='--', alpha=0.4)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.4)
    axes[0].legend(loc='upper right', fontsize=8, ncol=2)
    axes[2].set_xlabel('Sample Index')
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Single-wing reconstruction plot saved → {save_path}", flush=True)


def _save_per_dim_artifacts(
    target_dir: str,
    run_config: dict,
    ae_state: dict,
    dec_state: dict,
    val_rmse_deg: np.ndarray | None,
    val_loss: float,
    fixed_len_L: int,
    val_indices: list,
    val_indices_src_path: str,
    fl_dataset_path: str,
    template: np.ndarray,
    template_path: str,
    val_trajs: list,
    seed: int,
    device: str,
    bucket_axes: list,
    representation: str = 'sa',
) -> None:
    """
    Write a per-latent-dim model checkpoint into `target_dir`, plus run the same
    eval suite (reconstruction plot, per-phase error, per-bucket eval) as the
    top-level overall-best save.

    Used after the grid loop to emit one self-contained subdirectory per
    unique latent_dim value, so each can be loaded and inspected independently.
    """
    os.makedirs(target_dir, exist_ok=True)

    repr_spec      = get_representation(representation)
    n_channels     = repr_spec['n_channels']
    activation     = run_config.get('activation', 'gelu')
    dropout        = run_config.get('dropout', 0.0)
    base_channels  = run_config.get('base_channels', _BASE_CHANNELS)
    bottleneck_len = run_config.get('bottleneck_len', _BOTTLENECK_LEN)
    decoder_kernel = run_config.get('decoder_kernel_size', 5)
    channel_w      = run_config.get('channel_loss_weight')
    cycle_w        = run_config.get('cycle_weight', 0.0)
    endpoint_w     = run_config.get('endpoint_weight', 1.0)
    endpoint_s     = run_config.get('endpoint_samples', 3)
    derivative_w   = run_config.get('derivative_weight', 0.0)
    latent_dim     = int(run_config['latent_dim'])
    sa_scale_list  = [float(v) for v in repr_spec['input_scale']]
    val_rmse_list  = None if val_rmse_deg is None else [float(v) for v in val_rmse_deg]
    chan_labels    = list(repr_spec['channel_labels'])

    torch.save({
        'state_dict':          dec_state,
        'latent_dim':          latent_dim,
        'representation':      representation,
        'in_channels':         n_channels,
        'out_channels':        n_channels,
        'activation':          activation,
        'dropout':             dropout,
        'base_channels':       base_channels,
        'bottleneck_len':      bottleneck_len,
        'decoder_kernel_size': decoder_kernel,
        'output_len':          fixed_len_L,
        'sa_scale':            sa_scale_list,
        'val_loss':            val_loss,
        'val_rmse_deg':        val_rmse_list,
        'channel_labels':      chan_labels,
    }, os.path.join(target_dir, 'best_decoder.pt'))

    torch.save({
        'state_dict':          ae_state,
        'latent_dim':          latent_dim,
        'representation':      representation,
        'in_channels':         n_channels,
        'out_channels':        n_channels,
        'activation':          activation,
        'dropout':             dropout,
        'base_channels':       base_channels,
        'channel_loss_weight': channel_w,
        'cycle_weight':        cycle_w,
        'endpoint_weight':     endpoint_w,
        'endpoint_samples':    endpoint_s,
        'derivative_weight':   derivative_w,
        'bottleneck_len':      bottleneck_len,
        'decoder_kernel_size': decoder_kernel,
        'output_len':          fixed_len_L,
        'sa_scale':            sa_scale_list,
        'val_loss':            val_loss,
        'val_rmse_deg':        val_rmse_list,
        'channel_labels':      chan_labels,
    }, os.path.join(target_dir, 'best_autoencoder.pt'))

    with open(os.path.join(target_dir, 'best_config.json'), 'w') as f:
        json.dump(run_config, f, indent=2)

    # Copy the run's shared val_indices.json (written once at the sweep root) into
    # this subdir, so each per-combo model dir is self-contained: downstream tools
    # (build_regressor_dataset, body_latent_regressor, evaluate_body_to_wingbeat)
    # all expect val_indices.json next to best_autoencoder.pt. Copying the root
    # file verbatim guarantees the per-dim split can't silently diverge from it.
    if os.path.exists(val_indices_src_path):
        shutil.copy(val_indices_src_path, os.path.join(target_dir, 'val_indices.json'))

    model = WingbeatAutoencoder(
        latent_dim          = latent_dim,
        in_channels         = n_channels,
        activation          = activation,
        dropout             = dropout,
        base_channels       = base_channels,
        bottleneck_len      = bottleneck_len,
        decoder_kernel_size = decoder_kernel,
        output_len          = fixed_len_L,
    )
    model.load_state_dict(ae_state)
    model.to(device)

    eval_dir = os.path.join(target_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    if representation == 'single_wing':
        template3 = np.load(single_wing_template_path(template_path))
        _plot_reconstructed_trajectory_single_wing(
            model     = model,
            val_trajs = val_trajs,
            template3 = template3,
            save_path = os.path.join(eval_dir, "reconstruction.png"),
            device    = device,
            seed      = seed,
        )
    else:
        _plot_reconstructed_trajectory(
            model     = model,
            val_trajs = val_trajs,
            template  = template,
            save_path = os.path.join(eval_dir, "reconstruction.png"),
            device    = device,
            seed      = seed,
        )

    # The per-phase / per-bucket evals decode S/A → L/R and assume 6 channels;
    # they're only run for the 'sa' representation for now.
    if bucket_axes and representation == 'sa':
        per_phase = plot_per_phase_error(
            model              = model,
            npz_path           = fl_dataset_path,
            val_trajectory_ids = set(int(i) for i in val_indices),
            device             = device,
            save_dir           = eval_dir,
        )
        plot_phase_range_distributions(
            errors_deg   = per_phase["errors_deg"],
            phase_ranges = list(DEFAULT_PHASE_RANGES),
            save_dir     = eval_dir,
        )
        for axis in expand_score_axis(list(bucket_axes)):
            evaluate_by_maneuver_bucket(
                model              = model,
                npz_path           = fl_dataset_path,
                val_trajectory_ids = set(int(i) for i in val_indices),
                score_axis         = axis,
                device             = device,
                save_dir           = eval_dir,
                template_path      = template_path,
            )

    print(f"  ✓ latent_dim={latent_dim} → {target_dir}  (val_loss={val_loss:.6f})", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Wingbeat Autoencoder Grid Search")
    parser.add_argument("--config", default="code/autoencoder_config.json", help="Path to JSON config file")
    parser.add_argument(
        "--job_name",
        default=None,
        help="Name used as prefix for the per-run model and analysis directories. "
             "If unset/empty, defaults to 'run' (models) and 'gridsearch' (analysis).",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        raw_config = json.load(f)

    # Keys whose values are lists are grid-searched; all others are fixed across every run.
    # Some keys hold *complex* values (lists/tuples) as a single unit — for those, a bare
    # list-of-scalars is a single value, while a list-of-lists is a grid over multiple values.
    # Add entries here when adding new config keys that hold a list value.
    COMPLEX_VALUE_KEYS = {
        'channel_loss_weight',              # value is (alpha_s, alpha_a)
        'post_training_bucket_eval_axes',   # value is a list of score axes, not a sweep dim
    }

    fixed: dict = {}
    grid:  dict = {}
    for k, v in raw_config.items():
        if k in COMPLEX_VALUE_KEYS:
            # A bare list-of-scalars is a single value; only a list-of-lists is a grid.
            is_grid = isinstance(v, list) and len(v) > 0 and isinstance(v[0], (list, tuple))
        else:
            is_grid = isinstance(v, list)
        (grid if is_grid else fixed)[k] = v

    grid_keys = list(grid.keys())
    combos    = list(itertools.product(*grid.values()))
    n_runs    = len(combos)
    print(f"Grid search: {n_runs} run(s)" + (f" over {grid_keys}" if grid_keys else " (no grid params)"))

    seed = fixed.get('random_seed', 42)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Representation: 'sa' (default, 6-ch S/A) or 'single_wing' (3-ch one wing at a time).
    representation = fixed.get('representation', 'sa')
    repr_spec      = get_representation(representation)
    n_channels     = repr_spec['n_channels']
    repr_array_key = repr_spec['array_key']
    print(f"Representation: {representation}  (in/out channels = {n_channels}, npz key = {repr_array_key})")

    # --- Auto-build every fixed-length dataset the grid will reference ---
    # fixed_len may be a sweep dim, so collect all unique L values that any combo
    # will request, then build the ones that are missing or stale.
    unique_fixed_lens = sorted({
        int({**fixed, **dict(zip(grid_keys, c))}['fixed_len']) for c in combos
    })
    auto_build = bool(fixed.get('auto_build_dataset', True))
    data_dir   = os.path.dirname(os.path.abspath(fixed['data_path']))
    for L_val in unique_fixed_lens:
        path = fixed_len_dataset_path(data_dir, L_val, representation)
        scar = fixed_len_sidecar_path(data_dir, L_val, representation)
        is_valid, reason = fixed_len_dataset_is_valid(
            output_path       = path,
            sidecar_path      = scar,
            L                 = L_val,
            trajectories_path = fixed['data_path'],
            template_path     = fixed['template_path'],
            representation    = representation,
        )
        if not is_valid:
            if not auto_build:
                raise FileNotFoundError(
                    f"Fixed-length dataset at {path} is missing or stale ({reason}). "
                    f"Either set auto_build_dataset=true in the config, or run "
                    f"`python code/transform_data.py --fixed_len {L_val}` to build it."
                )
            print(f"Building fixed-length dataset (L={L_val}): {reason} → {path}", flush=True)
            build_fixed_len_dataset_from_disk(
                L                 = L_val,
                trajectories_path = fixed['data_path'],
                template_path     = fixed['template_path'],
                output_path       = path,
                representation    = representation,
            )

    # --- L-agnostic state: template + trajectory list + train/val split (trajectory level) ---
    # The train/val split is on trajectory_ids, which are identical across L values
    # (CubicSpline resampling doesn't change which wingbeats exist, only how many
    # time samples each has). So one split is reused for every L.
    template     = np.load(fixed['template_path'])
    trajectories = np.load(fixed['data_path'], allow_pickle=True)
    n_trajs      = len(trajectories)

    n_val         = max(1, int(n_trajs * fixed.get('val_split', 0.15)))
    perm          = np.random.permutation(n_trajs)
    val_indices   = perm[:n_val].tolist()
    train_indices = perm[n_val:].tolist()
    val_traj_set  = set(int(i) for i in val_indices)
    val_trajs     = [trajectories[i] for i in val_indices]

    # --- Lazy per-L dataset cache. Each unique L value's wingbeats are loaded
    # at most once across the entire grid (e.g., when L=45 is shared between
    # combos sweeping latent_dim, the npz is loaded a single time).
    _dataset_cache: dict[int, tuple['FixedLenWingbeatDataset', 'FixedLenWingbeatDataset', str]] = {}

    def _get_datasets_for_L(L_val: int) -> tuple['FixedLenWingbeatDataset', 'FixedLenWingbeatDataset', str]:
        if L_val not in _dataset_cache:
            path           = fixed_len_dataset_path(data_dir, L_val, representation)
            fl_data        = np.load(path)
            wingbeats      = fl_data[repr_array_key]        # (N, C, L) float32, normalized
            trajectory_ids = fl_data['trajectory_ids']
            val_mask       = np.array([int(t) in val_traj_set for t in trajectory_ids], dtype=bool)
            train_ds       = FixedLenWingbeatDataset(wingbeats[~val_mask])
            val_ds         = FixedLenWingbeatDataset(wingbeats[ val_mask])
            _dataset_cache[L_val] = (train_ds, val_ds, path)
        return _dataset_cache[L_val]

    print(f"Trajectories : {len(train_indices)} train / {n_val} val (out of {n_trajs} total)")
    print(f"Fixed-L values requested: {unique_fixed_lens}")

    device = fixed.get('device', 'auto')
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Shared timestamp so models and analysis plots from the same run are paired by name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Directory prefixes: use the job name if provided, otherwise the original defaults.
    job_name        = (args.job_name or "").strip()
    models_prefix   = job_name if job_name else "run"
    analysis_prefix = job_name if job_name else "gridsearch"

    # Per-run model directory so previous checkpoints are never overwritten
    base_save_dir = fixed.get('save_dir', 'data/models/autoencoder')
    save_dir      = os.path.join(base_save_dir, f"{models_prefix}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    # Matching analysis directory for plots
    analysis_dir = os.path.join("data/analysis", f"{analysis_prefix}_{timestamp}")
    os.makedirs(analysis_dir, exist_ok=True)

    print(f"Models   → {save_dir}",     flush=True)
    print(f"Analysis → {analysis_dir}", flush=True)

    # --- Grid search ---
    summary              = []
    best_val_loss        = float('inf')
    best_run_config      = None
    best_decoder_state   = None
    best_autoencoder_state = None
    best_run_val_rmse_deg: np.ndarray | None = None
    # Per-(fixed_len, latent_dim) best across the grid: each entry is the winner
    # among all grid combos that share that (L, latent_dim) pair. After the grid
    # loop, these are persisted into nested subdirectories so interpretability
    # experiments can compare across either axis cleanly. The key is always a
    # (L, latent_dim) tuple; when one or both dims are scalar in the config,
    # the resulting subdir layout collapses appropriately (see the save block).
    per_combo_best: dict[tuple[int, int], dict] = {}

    def _build_model(rcfg: dict) -> WingbeatAutoencoder:
        return WingbeatAutoencoder(
            latent_dim=rcfg['latent_dim'],
            in_channels=n_channels,
            activation=rcfg.get('activation', 'gelu'),
            dropout=rcfg.get('dropout', 0.0),
            base_channels=rcfg.get('base_channels', _BASE_CHANNELS),
            bottleneck_len=rcfg.get('bottleneck_len', _BOTTLENECK_LEN),
            decoder_kernel_size=rcfg.get('decoder_kernel_size', 5),
            output_len=rcfg['fixed_len'],
        )

    for run_idx, combo in enumerate(combos):
        run_config = {**fixed, **dict(zip(grid_keys, combo))}
        print(f"\n--- Run {run_idx + 1}/{n_runs} | {dict(zip(grid_keys, combo))} ---", flush=True)

        # Fetch this combo's fixed-L datasets (cached so identical-L combos reuse them).
        combo_L = int(run_config['fixed_len'])
        train_dataset, val_dataset, fl_dataset_path = _get_datasets_for_L(combo_L)
        print(f"  L={combo_L}: {len(train_dataset)} train / {len(val_dataset)} val wingbeats", flush=True)

        # Restart-on-plateau settings: when enabled, training aborts early at
        # `restart_check_epoch` if the best val loss is still above `restart_threshold`,
        # and we retry with a fresh init (different sub-seed). Disabled by default.
        restart_on_plateau   = bool(run_config.get('restart_on_plateau', False))
        restart_check_epoch  = run_config.get('restart_check_epoch', 25) if restart_on_plateau else None
        restart_threshold    = run_config.get('restart_threshold', 0.0015) if restart_on_plateau else None
        restart_max_attempts = int(run_config.get('restart_max_attempts', 3)) if restart_on_plateau else 0

        run_seed = int(run_config.get('random_seed', seed))
        run_label = "_".join(f"{k}{v}" for k, v in zip(grid_keys, combo)) if grid_keys else "single"

        attempt = 0
        train_losses = val_losses = []
        best_state   = None
        was_stuck    = False
        best_val_rmse_deg: np.ndarray | None = None
        # Wall clock for this config — covers every restart attempt so retried
        # configs show their real cost. Stored on the summary entry below.
        run_t0 = time.perf_counter()
        while True:
            # Re-seed before each attempt so initial inits are deterministic per (seed, attempt).
            sub_seed = run_seed if attempt == 0 else run_seed * 1000 + attempt
            torch.manual_seed(sub_seed)
            np.random.seed(sub_seed)

            if attempt > 0:
                print(f"  Restart attempt {attempt}/{restart_max_attempts} (sub_seed={sub_seed})", flush=True)

            model = _build_model(run_config)

            loss_fig_path = os.path.join(
                analysis_dir,
                f"losses_run{run_idx + 1}_{run_label}"
                + (f"_attempt{attempt}" if attempt > 0 else "")
                + ".png",
            )

            train_losses, val_losses, best_state, was_stuck, best_val_rmse_deg = train_autoencoder(
                model               = model,
                train_dataset       = train_dataset,
                val_dataset         = val_dataset,
                n_epochs            = run_config.get('n_epochs', 100),
                lr                  = run_config.get('lr', 1e-3),
                batch_size          = run_config.get('batch_size', 32),
                device              = device,
                loss_fig_path       = loss_fig_path,
                optimizer_name      = run_config.get('optimizer', 'adam'),
                weight_decay        = run_config.get('weight_decay', 0.0),
                channel_loss_weight = run_config.get('channel_loss_weight'),
                cycle_weight        = run_config.get('cycle_weight', 0.0),
                endpoint_weight     = run_config.get('endpoint_weight', 1.0),
                endpoint_samples    = run_config.get('endpoint_samples', 3),
                derivative_weight   = run_config.get('derivative_weight', 0.0),
                restart_check_epoch = restart_check_epoch,
                restart_threshold   = restart_threshold,
                representation      = representation,
            )

            if not was_stuck or attempt >= restart_max_attempts:
                break
            attempt += 1

        attempts_used  = attempt + 1
        final_was_stuck = was_stuck  # True only if all attempts were exhausted while stuck
        train_seconds   = time.perf_counter() - run_t0

        model.load_state_dict(best_state)
        run_best = min(val_losses) if val_losses else min(train_losses)
        rmse_msg = f"  {_format_rmse_degrees(best_val_rmse_deg, labels=repr_spec['channel_labels'])}" if best_val_rmse_deg is not None else ""
        h, rem = divmod(int(train_seconds), 3600)
        m, s   = divmod(rem, 60)
        train_time_hms = f"{h:d}:{m:02d}:{s:02d}"
        print(
            f"  Best val loss: {run_best:.6f}  "
            f"(attempts={attempts_used}, final_stuck={final_was_stuck}, "
            f"train_time={train_time_hms}){rmse_msg}"
        )

        summary.append({
            **dict(zip(grid_keys, combo)),
            'best_val_loss':       run_best,
            'best_val_rmse_deg':   None if best_val_rmse_deg is None else [float(v) for v in best_val_rmse_deg],
            'attempts':            attempts_used,
            'final_was_stuck':     final_was_stuck,
            'train_seconds':       float(train_seconds),
            'train_time_hms':      train_time_hms,
        })

        if run_best < best_val_loss:
            best_val_loss          = run_best
            best_run_config        = run_config
            best_decoder_state     = copy.deepcopy(model.decoder.state_dict())
            best_autoencoder_state = copy.deepcopy(best_state)
            best_run_val_rmse_deg  = best_val_rmse_deg

        # Track the best across all configs sharing this (fixed_len, latent_dim)
        # pair. After the grid completes we persist one checkpoint per pair into
        # an appropriately nested subdir.
        combo_key = (combo_L, int(run_config['latent_dim']))
        prev = per_combo_best.get(combo_key)
        if prev is None or run_best < prev['val_loss']:
            per_combo_best[combo_key] = {
                'val_loss':          run_best,
                'run_config':        run_config,
                'autoencoder_state': copy.deepcopy(best_state),
                'decoder_state':     copy.deepcopy(model.decoder.state_dict()),
                'val_rmse_deg':      best_val_rmse_deg,
                'fl_dataset_path':   fl_dataset_path,
            }

    # --- Persist results ---

    # The decoder checkpoint stores everything needed to reconstruct the architecture.
    best_activation          = best_run_config.get('activation', 'gelu')
    best_dropout             = best_run_config.get('dropout', 0.0)
    best_base_channels       = best_run_config.get('base_channels', _BASE_CHANNELS)
    best_bottleneck_len      = best_run_config.get('bottleneck_len', _BOTTLENECK_LEN)
    best_decoder_kernel_size = best_run_config.get('decoder_kernel_size', 5)
    best_channel_loss_weight = best_run_config.get('channel_loss_weight')
    best_cycle_weight        = best_run_config.get('cycle_weight', 0.0)
    best_endpoint_weight     = best_run_config.get('endpoint_weight', 1.0)
    best_endpoint_samples    = best_run_config.get('endpoint_samples', 3)
    best_derivative_weight   = best_run_config.get('derivative_weight', 0.0)
    # The input scale (per-representation) is included so a loader can recover physical
    # units without re-importing the constant.
    sa_scale_list = [float(v) for v in repr_spec['input_scale']]
    best_channel_labels = list(repr_spec['channel_labels'])
    val_rmse_deg_list = (
        None if best_run_val_rmse_deg is None else [float(v) for v in best_run_val_rmse_deg]
    )
    # Overall-best L may differ from any individual combo's L when fixed_len is swept.
    best_fixed_len_L     = int(best_run_config['fixed_len'])
    best_fl_dataset_path = fixed_len_dataset_path(data_dir, best_fixed_len_L, representation)
    torch.save(
        {
            'state_dict':          best_decoder_state,
            'latent_dim':          best_run_config['latent_dim'],
            'representation':      representation,
            'in_channels':         n_channels,
            'out_channels':        n_channels,
            'activation':          best_activation,
            'dropout':             best_dropout,
            'base_channels':       best_base_channels,
            'bottleneck_len':      best_bottleneck_len,
            'decoder_kernel_size': best_decoder_kernel_size,
            'output_len':          best_fixed_len_L,
            'sa_scale':            sa_scale_list,
            'val_loss':            best_val_loss,
            'val_rmse_deg':        val_rmse_deg_list,
            'channel_labels':      best_channel_labels,
        },
        os.path.join(save_dir, 'best_decoder.pt'),
    )

    # Full autoencoder in case the encoder is useful later.
    torch.save(
        {
            'state_dict':          best_autoencoder_state,
            'latent_dim':          best_run_config['latent_dim'],
            'representation':      representation,
            'in_channels':         n_channels,
            'out_channels':        n_channels,
            'activation':          best_activation,
            'dropout':             best_dropout,
            'base_channels':       best_base_channels,
            'channel_loss_weight': best_channel_loss_weight,
            'cycle_weight':        best_cycle_weight,
            'endpoint_weight':     best_endpoint_weight,
            'endpoint_samples':    best_endpoint_samples,
            'derivative_weight':   best_derivative_weight,
            'bottleneck_len':      best_bottleneck_len,
            'decoder_kernel_size': best_decoder_kernel_size,
            'output_len':          best_fixed_len_L,
            'sa_scale':            sa_scale_list,
            'val_loss':            best_val_loss,
            'val_rmse_deg':        val_rmse_deg_list,
            'channel_labels':      best_channel_labels,
        },
        os.path.join(save_dir, 'best_autoencoder.pt'),
    )

    with open(os.path.join(save_dir, 'best_config.json'), 'w') as f:
        json.dump(best_run_config, f, indent=2)

    # All runs sorted by val loss — useful for manual inspection.
    with open(os.path.join(save_dir, 'grid_search_summary.json'), 'w') as f:
        json.dump(sorted(summary, key=lambda r: r['best_val_loss']), f, indent=2)

    # Validation-set membership — needed by any downstream script that wants to plot
    # reconstructions without leaking training trajectories
    with open(os.path.join(save_dir, 'val_indices.json'), 'w') as f:
        json.dump({
            'val_indices':   [int(i) for i in val_indices],
            'train_indices': [int(i) for i in train_indices],
            'n_total':       n_trajs,
            'val_split':     fixed.get('val_split', 0.15),
            'random_seed':   seed,
            'data_path':     fixed['data_path'],
        }, f, indent=2)

    # --- Reconstruction plot using the best model across the entire grid search ---
    best_model = WingbeatAutoencoder(
        latent_dim=best_run_config['latent_dim'],
        in_channels=n_channels,
        activation=best_activation,
        dropout=best_dropout,
        base_channels=best_base_channels,
        bottleneck_len=best_bottleneck_len,
        decoder_kernel_size=best_decoder_kernel_size,
        output_len=best_fixed_len_L,
    )
    best_model.load_state_dict(best_autoencoder_state)
    best_model.to(device)
    if representation == 'single_wing':
        _plot_reconstructed_trajectory_single_wing(
            model      = best_model,
            val_trajs  = val_trajs,
            template3  = np.load(single_wing_template_path(fixed['template_path'])),
            save_path  = os.path.join(analysis_dir, "best_model_reconstruction.png"),
            device     = device,
            seed       = seed,
        )
    else:
        _plot_reconstructed_trajectory(
            model      = best_model,
            val_trajs  = val_trajs,
            template   = template,
            save_path  = os.path.join(analysis_dir, "best_model_reconstruction.png"),
            device     = device,
            seed       = seed,
        )

    # --- Bucket eval on the overall best model. Same axes as the per-config eval.
    # Output files have no run-label prefix so this matches the manual evaluator.
    # The per-phase/per-bucket evals are S/A-only for now (gated on representation).
    bucket_axes = fixed.get('post_training_bucket_eval_axes', ['max'])
    if bucket_axes and representation == 'sa':
        eval_dir = os.path.join(save_dir, "eval")
        os.makedirs(eval_dir, exist_ok=True)
        # Per-phase error plot is independent of score_axis, so run it once.
        # We pass best_fl_dataset_path — the npz for the best model's L.
        per_phase = plot_per_phase_error(
            model              = best_model,
            npz_path           = best_fl_dataset_path,
            val_trajectory_ids = set(int(i) for i in val_indices),
            device             = device,
            save_dir           = eval_dir,
        )
        # Phase-window error histograms — driven by the same raw-error tensor
        # plot_per_phase_error already produced, so no extra model pass.
        plot_phase_range_distributions(
            errors_deg   = per_phase["errors_deg"],
            phase_ranges = list(DEFAULT_PHASE_RANGES),
            save_dir     = eval_dir,
        )
        for axis in expand_score_axis(list(bucket_axes)):
            evaluate_by_maneuver_bucket(
                model              = best_model,
                npz_path           = best_fl_dataset_path,
                val_trajectory_ids = set(int(i) for i in val_indices),
                score_axis         = axis,
                device             = device,
                save_dir           = eval_dir,
                file_prefix        = "",
                print_table        = True,
                template_path      = fixed['template_path'],
            )

    # --- Per-(fixed_len, latent_dim) subdirectories ------------------------
    # Layout depends on which dimensions were swept:
    #   * Both L and latent_dim swept  → nested fixed_len_<L>/latent_dim_<k>/
    #   * Only fixed_len swept         → flat   fixed_len_<L>/
    #   * Only latent_dim swept        → flat   latent_dim_<k>/      (legacy layout)
    #   * Neither swept                → no subdirs (only the top-level overall best)
    # The "winner" for each cell is the best run across every other grid dim
    # (e.g., seed, base_channels, ...) for that (L, latent_dim) pair.
    unique_L_in_grid  = sorted({L for (L, _)  in per_combo_best.keys()})
    unique_LD_in_grid = sorted({ld for (_, ld) in per_combo_best.keys()})
    L_swept           = len(unique_L_in_grid)  > 1
    LD_swept          = len(unique_LD_in_grid) > 1

    if L_swept or LD_swept:
        print(
            f"\nSaving per-combo winners under {save_dir}  "
            f"({len(per_combo_best)} cell(s), L_swept={L_swept}, LD_swept={LD_swept})",
            flush=True,
        )
        for (L_val, ld_val) in sorted(per_combo_best.keys()):
            info = per_combo_best[(L_val, ld_val)]
            if L_swept and LD_swept:
                target_dir = os.path.join(save_dir, f"fixed_len_{L_val}", f"latent_dim_{ld_val}")
            elif L_swept:
                target_dir = os.path.join(save_dir, f"fixed_len_{L_val}")
            else:
                target_dir = os.path.join(save_dir, f"latent_dim_{ld_val}")
            _save_per_dim_artifacts(
                target_dir       = target_dir,
                run_config       = info['run_config'],
                ae_state         = info['autoencoder_state'],
                dec_state        = info['decoder_state'],
                val_rmse_deg     = info['val_rmse_deg'],
                val_loss         = info['val_loss'],
                fixed_len_L      = L_val,
                val_indices      = val_indices,
                val_indices_src_path = os.path.join(save_dir, 'val_indices.json'),
                fl_dataset_path  = info['fl_dataset_path'],
                template         = template,
                template_path    = fixed['template_path'],
                val_trajs        = val_trajs,
                seed             = seed,
                device           = device,
                bucket_axes      = bucket_axes,
                representation   = representation,
            )

    # --- Cross-run summary: fixed_len vs performance.
    # Only meaningful when more than one fixed_len was actually swept; otherwise
    # the strip plot would be a single column with no comparison to make.
    if L_swept:
        try:
            plot_fixed_len_vs_performance(
                summary_entries = summary,
                out_path        = os.path.join(save_dir, "fixed_len_vs_rmse_deg.png"),
                metric          = "rmse_deg",
            )
        except Exception as exc:
            print(f"  Skipping fixed_len-vs-performance plot: {exc}", flush=True)

    print(f"\n{'=' * 45}")
    print(f"Grid search complete.")
    print(f"Best val loss : {best_val_loss:.6f}")
    if grid_keys:
        print(f"Best config   : {dict(zip(grid_keys, [best_run_config[k] for k in grid_keys]))}")
    print(f"Saved to      : {save_dir}")
    print(f"Plots saved to: {analysis_dir}")


if __name__ == '__main__':
    main()
