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

from transform_data import _wingbeat_peaks, _segment_to_sa, _sa_to_segment

class WingbeatEncoder(nn.Module):
    def __init__(self, latent_dim: int, in_channels: int = 6):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm1d(128),
            nn.ELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convs(x)      # (B, 128, ~n/4)
        x = self.pool(x)       # (B, 128, 1)
        x = x.squeeze(-1)      # (B, 128)
        return self.fc(x)      # (B, latent_dim)


class WingbeatDecoder(nn.Module):
    def __init__(self, latent_dim: int, out_channels: int = 6):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128)
        self.convs = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Conv1d(64, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Conv1d(32, out_channels, kernel_size=5, padding=2),
        )

    def forward(self, z: torch.Tensor, target_len: int) -> torch.Tensor:
        x = self.fc(z)                                                             # (B, 128)
        x = x.unsqueeze(-1)                                                        # (B, 128, 1)
        x = F.interpolate(x, size=target_len, mode='linear', align_corners=False)  # (B, 128, n)
        return self.convs(x)                                                       # (B, 6, n)


class WingbeatAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 16, in_channels: int = 6):
        super().__init__()
        self.encoder = WingbeatEncoder(latent_dim, in_channels)
        self.decoder = WingbeatDecoder(latent_dim, in_channels)
        self.latent_dim = latent_dim

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
    Segments continuous flight trajectories into individual wingbeat cycles
    and applies the S/A transformation relative to a golden template.

    Each sample is a (6, n) tensor: [S_phi, S_theta, S_psi, A_phi, A_theta, A_psi].
    """

    def __init__(self, trajectories: list, template: np.ndarray):
        self.samples = []
        for traj in trajectories:
            peaks = _wingbeat_peaks(traj)
            for i in range(len(peaks) - 1):
                start, end = peaks[i], peaks[i + 1]
                sa = _segment_to_sa(traj[start:end], template)  # (n, 6)
                self.samples.append(torch.from_numpy(sa.T))     # (6, n)

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

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

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
    decoder = WingbeatDecoder(latent_dim=ckpt['latent_dim'])
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

        # Encode → decode
        sa = _segment_to_sa(segment, template)
        x  = torch.from_numpy(sa.T).unsqueeze(0).to(device)         # (1, 6, n)
        with torch.no_grad():
            recon_sa = model(x).squeeze(0).T.cpu().numpy()           # (n, 6)
        reconstruction = _sa_to_segment(recon_sa, template)

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
    perm    = np.random.permutation(n_trajs)
    val_trajs   = [trajectories[i] for i in perm[:n_val]]
    train_trajs = [trajectories[i] for i in perm[n_val:]]

    train_dataset = WingbeatDataset(train_trajs, template)
    val_dataset   = WingbeatDataset(val_trajs,   template)

    print(f"Trajectories : {len(train_trajs)} train / {n_val} val")
    print(f"Wingbeats    : {len(train_dataset)} train / {len(val_dataset)} val")

    device = fixed.get('device', 'auto')
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    save_dir = fixed.get('save_dir', 'data/models/autoencoder')
    os.makedirs(save_dir, exist_ok=True)

    # One analysis sub-directory per grid-search invocation so plots don't overwrite each other
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_dir = os.path.join("data/analysis", f"gridsearch_{timestamp}")
    os.makedirs(analysis_dir, exist_ok=True)
    print(f"Analysis plots → {analysis_dir}", flush=True)

    # --- Grid search ---
    summary              = []
    best_val_loss        = float('inf')
    best_run_config      = None
    best_decoder_state   = None
    best_autoencoder_state = None

    for run_idx, combo in enumerate(combos):
        run_config = {**fixed, **dict(zip(grid_keys, combo))}
        print(f"\n--- Run {run_idx + 1}/{n_runs} | {dict(zip(grid_keys, combo))} ---", flush=True)

        model = WingbeatAutoencoder(latent_dim=run_config['latent_dim'])

        run_label = "_".join(f"{k}{v}" for k, v in zip(grid_keys, combo)) if grid_keys else "single"
        loss_fig_path = os.path.join(analysis_dir, f"losses_run{run_idx + 1}_{run_label}.png")

        train_losses, val_losses, best_state = train_autoencoder(
            model         = model,
            train_dataset = train_dataset,
            val_dataset   = val_dataset,
            n_epochs      = run_config.get('n_epochs', 100),
            lr            = run_config.get('lr', 1e-3),
            batch_size    = run_config.get('batch_size', 32),
            device        = device,
            loss_fig_path = loss_fig_path,
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
    torch.save(
        {'state_dict': best_decoder_state, 'latent_dim': best_run_config['latent_dim'], 'val_loss': best_val_loss},
        os.path.join(save_dir, 'best_decoder.pt'),
    )

    # Full autoencoder in case the encoder is useful later.
    torch.save(
        {'state_dict': best_autoencoder_state, 'latent_dim': best_run_config['latent_dim'], 'val_loss': best_val_loss},
        os.path.join(save_dir, 'best_autoencoder.pt'),
    )

    with open(os.path.join(save_dir, 'best_config.json'), 'w') as f:
        json.dump(best_run_config, f, indent=2)

    # All runs sorted by val loss — useful for manual inspection.
    with open(os.path.join(save_dir, 'grid_search_summary.json'), 'w') as f:
        json.dump(sorted(summary, key=lambda r: r['best_val_loss']), f, indent=2)

    # --- Reconstruction plot using the best model across the entire grid search ---
    best_model = WingbeatAutoencoder(latent_dim=best_run_config['latent_dim'])
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
