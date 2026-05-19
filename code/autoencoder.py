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
import sys
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
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from transform_data import _wingbeat_peaks, _segment_to_sa, _sa_to_segment, SA_PHYSICAL_SCALE

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
        # Channel growth doubles each layer: base/4 → base/2 → base.
        c1 = base_channels // 4
        c2 = base_channels // 2
        c3 = base_channels
        self.base_channels  = base_channels
        self.bottleneck_len = bottleneck_len

        self.convs = nn.Sequential(
            nn.Conv1d(in_channels, c1, kernel_size=5, padding=2, padding_mode='replicate'),
            nn.GroupNorm(_NORM_GROUPS, c1),
            _make_activation(activation),
            nn.Conv1d(c1, c2, kernel_size=5, padding=2, stride=2, padding_mode='replicate'),
            nn.GroupNorm(_NORM_GROUPS, c2),
            _make_activation(activation),
            nn.Conv1d(c2, c3, kernel_size=3, padding=1, stride=2, padding_mode='replicate'),
            nn.GroupNorm(_NORM_GROUPS, c3),
            _make_activation(activation),
        )
        # Pool to bottleneck_len time-steps so the latent layer sees temporal layout
        self.pool    = nn.AdaptiveAvgPool1d(bottleneck_len)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(c3 * bottleneck_len, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convs(x)             # (B, _BASE_CHANNELS, ~n/4)
        x = self.pool(x)              # (B, _BASE_CHANNELS, _BOTTLENECK_LEN)
        x = x.flatten(start_dim=1)    # (B, _BASE_CHANNELS * _BOTTLENECK_LEN)
        x = self.dropout(x)
        return self.fc(x)             # (B, latent_dim)


class WingbeatDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        out_channels: int = 6,
        activation: str = 'gelu',
        dropout: float = 0.0,
        base_channels: int = _BASE_CHANNELS,
        bottleneck_len: int = _BOTTLENECK_LEN,
        decoder_kernel_size: int = 5,
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

        kw = decoder_kernel_size
        pad = kw // 2  # 'same' padding for stride=1
        self.fc      = nn.Linear(latent_dim, c3 * bottleneck_len)
        self.dropout = nn.Dropout(dropout)
        self.convs = nn.Sequential(
            nn.Conv1d(c3, c2, kernel_size=kw, padding=pad, padding_mode='replicate'),
            nn.GroupNorm(_NORM_GROUPS, c2),
            _make_activation(activation),
            nn.Conv1d(c2, c1, kernel_size=kw, padding=pad, padding_mode='replicate'),
            nn.GroupNorm(_NORM_GROUPS, c1),
            _make_activation(activation),
            nn.Conv1d(c1, out_channels, kernel_size=kw, padding=pad, padding_mode='replicate'),
        )

    def forward(self, z: torch.Tensor, target_len: int) -> torch.Tensor:
        x = self.fc(z)                                                                    # (B, C*L)
        x = self.dropout(x)
        x = x.view(x.size(0), self.base_channels, self.bottleneck_len)                    # (B, C, L)
        x = F.interpolate(x, size=target_len, mode='linear', align_corners=False)         # (B, C, n)
        return self.convs(x)                                                              # (B, 6, n)


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
    ):
        super().__init__()
        self.encoder = WingbeatEncoder(
            latent_dim, in_channels, activation, dropout, base_channels, bottleneck_len,
        )
        self.decoder = WingbeatDecoder(
            latent_dim, in_channels, activation, dropout, base_channels, bottleneck_len, decoder_kernel_size,
        )
        self.latent_dim          = latent_dim
        self.activation          = activation
        self.dropout             = dropout
        self.base_channels       = base_channels
        self.bottleneck_len      = bottleneck_len
        self.decoder_kernel_size = decoder_kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_len = x.shape[-1]
        z = self.encoder(x)
        return self.decoder(z, target_len)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor, target_len: int) -> torch.Tensor:
        return self.decoder(z, target_len)


class WingbeatDataset(Dataset):
    """
    Segments continuous flight trajectories into individual wingbeat cycles,
    applies the S/A transformation relative to a golden template, and
    normalizes by SA_PHYSICAL_SCALE so all 6 channels have comparable range.

    Each sample is a (6, n) tensor: [S_phi, S_theta, S_psi, A_phi, A_theta, A_psi],
    normalized to roughly [-1, 1] per channel.
    """

    def __init__(self, trajectories: list, template: np.ndarray):
        # (6, 1) so it broadcasts against (6, n) samples
        self.scale = torch.from_numpy(SA_PHYSICAL_SCALE).view(6, 1)
        self.samples = []
        for traj in trajectories:
            peaks = _wingbeat_peaks(traj)
            for i in range(len(peaks) - 1):
                start, end = peaks[i], peaks[i + 1]
                sa = _segment_to_sa(traj[start:end], template)  # (n, 6)
                sample = torch.from_numpy(sa.T)                 # (6, n)
                self.samples.append(sample / self.scale)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.samples[idx]


def _collate_fn(batch: list) -> tuple[torch.Tensor, torch.Tensor]:
    """Pads variable-length wingbeats by replicating each sample's last value."""
    lengths = torch.tensor([x.shape[-1] for x in batch])
    max_n   = int(lengths.max())
    padded  = torch.empty(len(batch), batch[0].shape[0], max_n, dtype=batch[0].dtype)
    for i, x in enumerate(batch):
        n = x.shape[-1]
        padded[i, :, :n] = x
        if n < max_n:
            padded[i, :, n:] = x[:, -1:]   # broadcast last sample across the pad
    return padded, lengths

def _masked_mse(
    recon: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor,
    channel_weight: torch.Tensor | None = None,
    endpoint_weight: float = 1.0,
    endpoint_samples: int = 3,
) -> torch.Tensor:
    """
    MSE computed only over the valid (non-padded) time steps.

    `channel_weight` (shape (C,)): per-channel multiplier on squared errors.
        Used to up-weight A channels so the model can't trivially minimize loss
        by predicting symmetric (A=0) outputs.
    `endpoint_weight` (scalar): multiplier applied to the first and last
        `endpoint_samples` valid time-steps of each wingbeat. Forces the model
        to fit the small-magnitude residuals near the stroke peaks instead of
        letting them drift. Pass 1.0 to disable.

    Denominator stays `mask.sum()` (unweighted timestep count) so the weights
    act as relative scales, not re-normalizations.
    """
    mask = torch.zeros_like(target)
    pos_weight = torch.ones_like(target) if endpoint_weight != 1.0 else None
    for i, l in enumerate(lengths):
        l_int = int(l)
        mask[i, :, :l_int] = 1.0
        if pos_weight is not None:
            k = min(endpoint_samples, l_int)
            pos_weight[i, :, :k]              = endpoint_weight
            pos_weight[i, :, l_int - k:l_int] = endpoint_weight

    sq_err = (recon - target) ** 2 * mask                       # (B, C, T)
    if channel_weight is not None:
        sq_err = sq_err * channel_weight.view(1, -1, 1).to(sq_err.device, sq_err.dtype)
    if pos_weight is not None:
        sq_err = sq_err * pos_weight
    return sq_err.sum() / mask.sum()


def _cycle_consistency_loss(recon: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """
    Mean squared difference between each wingbeat's first valid sample and its
    last valid sample, across all channels. Pulls `recon[:, :, 0]` toward
    `recon[:, :, l-1]` so the reconstructed SA residual is cyclic — and since
    the template is cyclic by construction, this makes the physical seam between
    consecutive wingbeats continuous.
    """
    batch_size = recon.size(0)
    last_idx   = (lengths - 1).clamp(min=0).long().to(recon.device)
    first = recon[:, :, 0]                                                    # (B, C)
    last  = recon[torch.arange(batch_size, device=recon.device), :, last_idx] # (B, C)
    return ((first - last) ** 2).mean()


def _derivative_loss(recon: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """
    MSE between the time-derivatives of recon and target, masked to valid steps.

    Plain MSE is dominated by large smooth features and is nearly blind to small
    high-frequency wiggles — a reconstruction that smooths them out has tiny extra
    error. The derivative magnifies high-frequency content (each wiggle adds two
    opposite-sign diffs), so the model is forced to render it.

    For a wingbeat of valid length `l`, there are `l - 1` valid diff positions.
    """
    d_recon  = recon[:, :, 1:] - recon[:, :, :-1]    # (B, C, T-1)
    d_target = target[:, :, 1:] - target[:, :, :-1]  # (B, C, T-1)
    mask = torch.zeros_like(d_target)
    for i, l in enumerate(lengths):
        v = max(int(l) - 1, 0)
        mask[i, :, :v] = 1.0
    sq_err = (d_recon - d_target) ** 2 * mask
    return sq_err.sum() / mask.sum().clamp(min=1.0)


def train_autoencoder(
    model: WingbeatAutoencoder,
    train_dataset: WingbeatDataset,
    val_dataset: WingbeatDataset | None = None,
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
) -> tuple[list[float], list[float], dict]:
    """
    Trains the autoencoder.

    Returns:
        train_losses: per-epoch average training loss
        val_losses:   per-epoch average validation loss (empty list if no val_dataset)
        best_state:   state_dict of the epoch with the lowest monitored loss
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=_collate_fn)
    val_loader = (
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate_fn)
        if val_dataset else None
    )

    optimizer = _make_optimizer(optimizer_name, model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, patience=4, factor=0.5)

    # channel_loss_weight is now (alpha_s, alpha_a): a 2-tuple that the code broadcasts to
    # the 6 channels [S_phi, S_theta, S_psi, A_phi, A_theta, A_psi] as [αs, αs, αs, αa, αa, αa].
    # None or all-ones → unweighted MSE.
    channel_weight_tensor = None
    if channel_loss_weight is not None:
        if len(channel_loss_weight) != 2:
            raise ValueError(
                f"channel_loss_weight must be a 2-element sequence (alpha_s, alpha_a); "
                f"got {channel_loss_weight}"
            )
        alpha_s, alpha_a = float(channel_loss_weight[0]), float(channel_loss_weight[1])
        channel_weight_tensor = torch.tensor(
            [alpha_s, alpha_s, alpha_s, alpha_a, alpha_a, alpha_a], dtype=torch.float32,
        )

    model.to(device)
    train_losses, val_losses = [], []
    best_monitor = float('inf')
    best_state = None

    for epoch in range(n_epochs):
        model.train()
        total_train = 0.0
        for x, lengths in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{n_epochs}", leave=False, file=sys.stdout, disable=not sys.stdout.isatty()):
            x = x.to(device)
            recon = model(x)
            mse = _masked_mse(
                recon, x, lengths,
                channel_weight   = channel_weight_tensor,
                endpoint_weight  = endpoint_weight,
                endpoint_samples = endpoint_samples,
            )
            loss = mse
            if cycle_weight > 0.0:
                loss = loss + cycle_weight * _cycle_consistency_loss(recon, lengths)
            if derivative_weight > 0.0:
                loss = loss + derivative_weight * _derivative_loss(recon, x, lengths)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train += loss.item()

        avg_train = total_train / len(train_loader)
        train_losses.append(avg_train)

        avg_val = None
        if val_loader:
            model.eval()
            total_val = 0.0
            with torch.no_grad():
                for x, lengths in val_loader:
                    x = x.to(device)
                    # Val loss is unweighted (no channel/endpoint/cycle terms) so it
                    # stays comparable across runs with different loss-weight settings.
                    total_val += _masked_mse(model(x), x, lengths).item()
            avg_val = total_val / len(val_loader)
            val_losses.append(avg_val)

        monitor = avg_val if avg_val is not None else avg_train
        scheduler.step(monitor)

        if monitor < best_monitor:
            best_monitor = monitor
            best_state = copy.deepcopy(model.state_dict())

        msg = f"Epoch {epoch + 1:>4}/{n_epochs}  train={avg_train:.6f}"
        if avg_val is not None:
            msg += f"  val={avg_val:.6f}"
        print(msg, flush=True)

    if loss_fig_path:
        _save_loss_figure(train_losses, val_losses, loss_fig_path)

    return train_losses, val_losses, best_state

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
    Loads a saved decoder from a checkpoint file.

    Usage:
        decoder = load_decoder("data/models/autoencoder/best_decoder.pt")
        wings = decoder(z, target_len=68)
    """
    ckpt = torch.load(path, map_location=device)
    decoder = WingbeatDecoder(
        latent_dim=ckpt['latent_dim'],
        activation=ckpt.get('activation', 'gelu'),
        dropout=ckpt.get('dropout', 0.0),
        base_channels=ckpt.get('base_channels', _BASE_CHANNELS),
        bottleneck_len=ckpt.get('bottleneck_len', _BOTTLENECK_LEN),
        decoder_kernel_size=ckpt.get('decoder_kernel_size', 5),
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

    for i in range(start_beat, start_beat + n_beats):
        start, end = int(peaks[i]), int(peaks[i + 1])
        segment = traj[start:end]                                    # (n, 6)
        n       = segment.shape[0]

        # Template matched to this segment length (S=A=0 → hat=0 → output = matched template)
        tmpl_matched = _sa_to_segment(np.zeros((n, 6), dtype=np.float32), template)

        # Encode → decode (model operates in SA_PHYSICAL_SCALE-normalized space)
        sa = _segment_to_sa(segment, template) / SA_PHYSICAL_SCALE   # (n, 6) normalized
        x  = torch.from_numpy(sa.T).unsqueeze(0).to(device)          # (1, 6, n)
        with torch.no_grad():
            recon_sa = model(x).squeeze(0).T.cpu().numpy()           # (n, 6) still normalized
        reconstruction = _sa_to_segment(recon_sa * SA_PHYSICAL_SCALE, template)

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
    COMPLEX_VALUE_KEYS = {'channel_loss_weight'}  # value is (alpha_s, alpha_a)

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

    # --- Load data ---
    # trajectories.npy is treated as the authoritative, already-validated dataset —
    # garbage filtering is the responsibility of transform_data.py at data-prep time.
    trajectories = np.load(fixed['data_path'], allow_pickle=True)
    template     = np.load(fixed['template_path'])
    n_trajs      = len(trajectories)

    # Split at trajectory level to prevent leakage between wingbeats of the same flight
    n_val       = max(1, int(n_trajs * fixed.get('val_split', 0.15)))
    perm        = np.random.permutation(n_trajs)
    val_indices   = perm[:n_val].tolist()      # persisted so eval scripts don't accidentally pick a training trajectory
    train_indices = perm[n_val:].tolist()
    val_trajs   = [trajectories[i] for i in val_indices]
    train_trajs = [trajectories[i] for i in train_indices]

    train_dataset = WingbeatDataset(train_trajs, template)
    val_dataset   = WingbeatDataset(val_trajs,   template)

    print(f"Trajectories : {len(train_trajs)} train / {n_val} val (out of {n_trajs} total)")
    print(f"Wingbeats    : {len(train_dataset)} train / {len(val_dataset)} val")

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

    for run_idx, combo in enumerate(combos):
        run_config = {**fixed, **dict(zip(grid_keys, combo))}
        print(f"\n--- Run {run_idx + 1}/{n_runs} | {dict(zip(grid_keys, combo))} ---", flush=True)

        model = WingbeatAutoencoder(
            latent_dim=run_config['latent_dim'],
            activation=run_config.get('activation', 'gelu'),
            dropout=run_config.get('dropout', 0.0),
            base_channels=run_config.get('base_channels', _BASE_CHANNELS),
            bottleneck_len=run_config.get('bottleneck_len', _BOTTLENECK_LEN),
            decoder_kernel_size=run_config.get('decoder_kernel_size', 5),
        )

        run_label = "_".join(f"{k}{v}" for k, v in zip(grid_keys, combo)) if grid_keys else "single"
        loss_fig_path = os.path.join(analysis_dir, f"losses_run{run_idx + 1}_{run_label}.png")

        train_losses, val_losses, best_state = train_autoencoder(
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
        )

        model.load_state_dict(best_state)
        run_best = min(val_losses) if val_losses else min(train_losses)
        print(f"  Best val loss: {run_best:.6f}")

        summary.append({**dict(zip(grid_keys, combo)), 'best_val_loss': run_best})

        if run_best < best_val_loss:
            best_val_loss          = run_best
            best_run_config        = run_config
            best_decoder_state     = copy.deepcopy(model.decoder.state_dict())
            best_autoencoder_state = copy.deepcopy(best_state)

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
    # SA scale is included so a loader can recover physical units without re-importing the constant.
    sa_scale_list = SA_PHYSICAL_SCALE.tolist()
    torch.save(
        {
            'state_dict':          best_decoder_state,
            'latent_dim':          best_run_config['latent_dim'],
            'activation':          best_activation,
            'dropout':             best_dropout,
            'base_channels':       best_base_channels,
            'bottleneck_len':      best_bottleneck_len,
            'decoder_kernel_size': best_decoder_kernel_size,
            'sa_scale':            sa_scale_list,
            'val_loss':            best_val_loss,
        },
        os.path.join(save_dir, 'best_decoder.pt'),
    )

    # Full autoencoder in case the encoder is useful later.
    torch.save(
        {
            'state_dict':          best_autoencoder_state,
            'latent_dim':          best_run_config['latent_dim'],
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
            'sa_scale':            sa_scale_list,
            'val_loss':            best_val_loss,
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
        activation=best_activation,
        dropout=best_dropout,
        base_channels=best_base_channels,
        bottleneck_len=best_bottleneck_len,
        decoder_kernel_size=best_decoder_kernel_size,
    )
    best_model.load_state_dict(best_autoencoder_state)
    _plot_reconstructed_trajectory(
        model      = best_model,
        val_trajs  = val_trajs,
        template   = template,
        save_path  = os.path.join(analysis_dir, "best_model_reconstruction.png"),
        device     = device,
        seed       = seed,
    )

    print(f"\n{'=' * 45}")
    print(f"Grid search complete.")
    print(f"Best val loss : {best_val_loss:.6f}")
    if grid_keys:
        print(f"Best config   : {dict(zip(grid_keys, [best_run_config[k] for k in grid_keys]))}")
    print(f"Saved to      : {save_dir}")
    print(f"Plots saved to: {analysis_dir}")


if __name__ == '__main__':
    main()
