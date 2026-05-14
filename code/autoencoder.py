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
_BASE_CHANNELS  = 128  # channels at the deepest conv layer
_BOTTLENECK_LEN = 4    # temporal length retained at the encoder bottleneck (was 1)
_NORM_GROUPS    = 8    # GroupNorm groups; must divide every conv output channel count (64, 128, 256)


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
    def __init__(self, latent_dim: int, in_channels: int = 6, activation: str = 'gelu', dropout: float = 0.0):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2, padding_mode='replicate'),
            nn.GroupNorm(_NORM_GROUPS, 32),
            _make_activation(activation),
            nn.Conv1d(32, 64, kernel_size=5, padding=2, stride=2, padding_mode='replicate'),
            nn.GroupNorm(_NORM_GROUPS, 64),
            _make_activation(activation),
            nn.Conv1d(64, _BASE_CHANNELS, kernel_size=3, padding=1, stride=2, padding_mode='replicate'),
            nn.GroupNorm(_NORM_GROUPS, _BASE_CHANNELS),
            _make_activation(activation),
        )
        # Keep _BOTTLENECK_LEN time-steps so the latent layer sees temporal layout
        self.pool    = nn.AdaptiveAvgPool1d(_BOTTLENECK_LEN)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(_BASE_CHANNELS * _BOTTLENECK_LEN, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convs(x)             # (B, _BASE_CHANNELS, ~n/4)
        x = self.pool(x)              # (B, _BASE_CHANNELS, _BOTTLENECK_LEN)
        x = x.flatten(start_dim=1)    # (B, _BASE_CHANNELS * _BOTTLENECK_LEN)
        x = self.dropout(x)
        return self.fc(x)             # (B, latent_dim)


class WingbeatDecoder(nn.Module):
    def __init__(self, latent_dim: int, out_channels: int = 6, activation: str = 'gelu', dropout: float = 0.0):
        super().__init__()
        self.fc      = nn.Linear(latent_dim, _BASE_CHANNELS * _BOTTLENECK_LEN)
        self.dropout = nn.Dropout(dropout)
        self.convs = nn.Sequential(
            nn.Conv1d(_BASE_CHANNELS, 64, kernel_size=5, padding=2, padding_mode='replicate'),
            nn.GroupNorm(_NORM_GROUPS, 64),
            _make_activation(activation),
            nn.Conv1d(64, 32, kernel_size=5, padding=2, padding_mode='replicate'),
            nn.GroupNorm(_NORM_GROUPS, 32),
            _make_activation(activation),
            nn.Conv1d(32, out_channels, kernel_size=5, padding=2, padding_mode='replicate'),
        )

    def forward(self, z: torch.Tensor, target_len: int) -> torch.Tensor:
        x = self.fc(z)                                                                    # (B, C*L)
        x = self.dropout(x)
        x = x.view(x.size(0), _BASE_CHANNELS, _BOTTLENECK_LEN)                            # (B, C, L)
        x = F.interpolate(x, size=target_len, mode='linear', align_corners=False)         # (B, C, n)
        return self.convs(x)                                                              # (B, 6, n)


class WingbeatAutoencoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 16,
        in_channels: int = 6,
        activation: str = 'gelu',
        dropout: float = 0.0,
        input_noise_std: float = 0.0,
    ):
        super().__init__()
        self.encoder         = WingbeatEncoder(latent_dim, in_channels, activation, dropout)
        self.decoder         = WingbeatDecoder(latent_dim, in_channels, activation, dropout)
        self.latent_dim      = latent_dim
        self.activation      = activation
        self.dropout         = dropout
        self.input_noise_std = input_noise_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_len = x.shape[-1]
        if self.training and self.input_noise_std > 0:
            x_in = x + torch.randn_like(x) * self.input_noise_std
        else:
            x_in = x
        z = self.encoder(x_in)
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
    """Pads variable-length wingbeats to the same length within a batch."""
    lengths = torch.tensor([x.shape[-1] for x in batch])
    padded = pad_sequence([x.T for x in batch], batch_first=True, padding_value=0.0)
    return padded.permute(0, 2, 1).contiguous(), lengths  # (B, 6, max_n), (B,)


def _masked_mse(recon: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """MSE computed only over the valid (non-padded) time steps."""
    mask = torch.zeros_like(target)
    for i, l in enumerate(lengths):
        mask[i, :, :l] = 1.0
    return F.mse_loss(recon * mask, target * mask, reduction='sum') / mask.sum()


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

    model.to(device)
    train_losses, val_losses = [], []
    best_monitor = float('inf')
    best_state = None

    for epoch in range(n_epochs):
        model.train()
        total_train = 0.0
        for x, lengths in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{n_epochs}", leave=False, file=sys.stdout, disable=not sys.stdout.isatty()):
            x = x.to(device)
            loss = _masked_mse(model(x), x, lengths)
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

    traj, peaks = candidates[rng.integers(len(candidates))]
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
    args = parser.parse_args()

    with open(args.config) as f:
        raw_config = json.load(f)

    # Keys whose values are lists are grid-searched; all others are fixed across every run.
    fixed = {k: v for k, v in raw_config.items() if not isinstance(v, list)}
    grid  = {k: v for k, v in raw_config.items() if isinstance(v, list)}

    grid_keys = list(grid.keys())
    combos    = list(itertools.product(*grid.values()))
    n_runs    = len(combos)
    print(f"Grid search: {n_runs} run(s)" + (f" over {grid_keys}" if grid_keys else " (no grid params)"))

    seed = fixed.get('random_seed', 42)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # --- Load data ---
    trajectories = np.load(fixed['data_path'], allow_pickle=True)
    template     = np.load(fixed['template_path'])

    # Split at trajectory level to prevent leakage between wingbeats of the same flight
    n_trajs = len(trajectories)
    n_val   = max(1, int(n_trajs * fixed.get('val_split', 0.15)))
    perm        = np.random.permutation(n_trajs)
    val_indices = perm[:n_val].tolist()    # persisted so eval scripts don't accidentally pick a training trajectory
    val_trajs   = [trajectories[i] for i in val_indices]
    train_trajs = [trajectories[i] for i in perm[n_val:]]

    train_dataset = WingbeatDataset(train_trajs, template)
    val_dataset   = WingbeatDataset(val_trajs,   template)

    print(f"Trajectories : {len(train_trajs)} train / {n_val} val")
    print(f"Wingbeats    : {len(train_dataset)} train / {len(val_dataset)} val")

    device = fixed.get('device', 'auto')
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Shared timestamp so models and analysis plots from the same run are paired by name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Per-run model directory so previous checkpoints are never overwritten
    base_save_dir = fixed.get('save_dir', 'data/models/autoencoder')
    save_dir      = os.path.join(base_save_dir, f"run_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    # Matching analysis directory for plots
    analysis_dir = os.path.join("data/analysis", f"gridsearch_{timestamp}")
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
            input_noise_std=run_config.get('input_noise_std', 0.0),
        )

        run_label = "_".join(f"{k}{v}" for k, v in zip(grid_keys, combo)) if grid_keys else "single"
        loss_fig_path = os.path.join(analysis_dir, f"losses_run{run_idx + 1}_{run_label}.png")

        train_losses, val_losses, best_state = train_autoencoder(
            model          = model,
            train_dataset  = train_dataset,
            val_dataset    = val_dataset,
            n_epochs       = run_config.get('n_epochs', 100),
            lr             = run_config.get('lr', 1e-3),
            batch_size     = run_config.get('batch_size', 32),
            device         = device,
            loss_fig_path  = loss_fig_path,
            optimizer_name = run_config.get('optimizer', 'adam'),
            weight_decay   = run_config.get('weight_decay', 0.0),
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
    best_activation       = best_run_config.get('activation', 'gelu')
    best_dropout          = best_run_config.get('dropout', 0.0)
    best_input_noise_std  = best_run_config.get('input_noise_std', 0.0)
    # SA scale is included so a loader can recover physical units without re-importing the constant.
    sa_scale_list = SA_PHYSICAL_SCALE.tolist()
    torch.save(
        {
            'state_dict':       best_decoder_state,
            'latent_dim':       best_run_config['latent_dim'],
            'activation':       best_activation,
            'dropout':          best_dropout,
            'sa_scale':         sa_scale_list,
            'val_loss':         best_val_loss,
        },
        os.path.join(save_dir, 'best_decoder.pt'),
    )

    # Full autoencoder in case the encoder is useful later.
    torch.save(
        {
            'state_dict':      best_autoencoder_state,
            'latent_dim':      best_run_config['latent_dim'],
            'activation':      best_activation,
            'dropout':         best_dropout,
            'input_noise_std': best_input_noise_std,
            'sa_scale':        sa_scale_list,
            'val_loss':        best_val_loss,
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
            'val_indices': val_indices,
            'n_total':     n_trajs,
            'val_split':   fixed.get('val_split', 0.15),
            'random_seed': seed,
            'data_path':   fixed['data_path'],
        }, f, indent=2)

    # --- Reconstruction plot using the best model across the entire grid search ---
    best_model = WingbeatAutoencoder(
        latent_dim=best_run_config['latent_dim'],
        activation=best_activation,
        dropout=best_dropout,
        input_noise_std=best_input_noise_std,
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
