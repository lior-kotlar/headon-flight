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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils.rnn import pad_sequence
from scipy.interpolate import interp1d
from tqdm import tqdm

from transform_data import _wingbeat_peaks  # Reuse the same peak detection logic for consistency

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
        phase_grid_template = np.linspace(0, 1, template.shape[0])

        for traj in trajectories:
            peaks = _wingbeat_peaks(traj)
            for i in range(len(peaks) - 1):
                start, end = peaks[i], peaks[i + 1]
                segment = traj[start:end]  # (n, 6)
                n = end - start

                phase_segment = np.linspace(0, 1, n)
                matched = interp1d(phase_grid_template, template, axis=0, kind='cubic')(phase_segment)

                hat = segment - matched
                S = (hat[:, :3] + hat[:, 3:]) / 2.0
                A = (hat[:, :3] - hat[:, 3:]) / 2.0
                transformed = np.concatenate([S, A], axis=1).astype(np.float32)  # (n, 6)

                self.samples.append(torch.from_numpy(transformed.T))  # (6, n)

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
        for x, lengths in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{n_epochs}", leave=False):
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

        if (epoch + 1) % 10 == 0:
            msg = f"Epoch {epoch + 1:>4}/{n_epochs}  train={avg_train:.6f}"
            if avg_val is not None:
                msg += f"  val={avg_val:.6f}"
            print(msg)

    return train_losses, val_losses, best_state


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

    # --- Grid search ---
    summary              = []
    best_val_loss        = float('inf')
    best_run_config      = None
    best_decoder_state   = None
    best_autoencoder_state = None

    for run_idx, combo in enumerate(combos):
        run_config = {**fixed, **dict(zip(grid_keys, combo))}
        print(f"\n--- Run {run_idx + 1}/{n_runs} | {dict(zip(grid_keys, combo))} ---")

        model = WingbeatAutoencoder(latent_dim=run_config['latent_dim'])

        train_losses, val_losses, best_state = train_autoencoder(
            model         = model,
            train_dataset = train_dataset,
            val_dataset   = val_dataset,
            n_epochs      = run_config.get('n_epochs', 100),
            lr            = run_config.get('lr', 1e-3),
            batch_size    = run_config.get('batch_size', 32),
            device        = device,
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

    print(f"\n{'=' * 45}")
    print(f"Grid search complete.")
    print(f"Best val loss : {best_val_loss:.6f}")
    if grid_keys:
        print(f"Best config   : {dict(zip(grid_keys, [best_run_config[k] for k in grid_keys]))}")
    print(f"Saved to      : {save_dir}")


if __name__ == '__main__':
    main()
