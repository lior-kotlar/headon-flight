"""
Standalone reconstruction pipeline for a trained autoencoder.

Loads a saved checkpoint and produces the reconstruction plot (PNG + interactive HTML)
WITHOUT retraining. Uses only validation trajectories so the visualization isn't biased
by training-set memorization.

Run from the project root:
    python code/evaluate_autoencoder.py
    python code/evaluate_autoencoder.py --model_dir data/models/autoencoder --n_beats 5
    python code/evaluate_autoencoder.py --seed 7         # reproducible trajectory choice
"""

import argparse
import json
import os

import numpy as np
import torch

from autoencoder import WingbeatAutoencoder, _plot_reconstructed_trajectory


def _resolve_model_dir(model_dir: str) -> str:
    """
    If model_dir holds a checkpoint directly, return it.
    Otherwise look for run_* subdirectories and return the most recent one.
    Timestamp format YYYYMMDD_HHMMSS sorts lexically the same as chronologically.
    """
    if os.path.exists(os.path.join(model_dir, "best_autoencoder.pt")):
        return model_dir

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"{model_dir} does not exist.")

    runs = sorted(
        d for d in os.listdir(model_dir)
        if d.startswith("run_") and os.path.isdir(os.path.join(model_dir, d))
    )
    if not runs:
        raise FileNotFoundError(
            f"No checkpoint or run_* subdirectory found in {model_dir}."
        )

    latest = os.path.join(model_dir, runs[-1])
    print(f"No checkpoint directly in {model_dir} — using latest run: {latest}", flush=True)
    return latest


def _load_val_trajectories(model_dir: str, trajectories: list, config: dict) -> list:
    """
    Returns the validation trajectories the model was trained against.

    Preferred path: val_indices.json (saved by autoencoder.py at training time).
    Fallback: re-derive the split deterministically from the config's seed + val_split.
    """
    val_path = os.path.join(model_dir, "val_indices.json")
    if os.path.exists(val_path):
        with open(val_path) as f:
            saved = json.load(f)
        val_indices = saved['val_indices']
        if saved.get('n_total') != len(trajectories):
            print(
                f"WARNING: trajectory count changed since training "
                f"({saved.get('n_total')} → {len(trajectories)}). Indices may be stale.",
                flush=True,
            )
        print(f"Loaded {len(val_indices)} validation indices from val_indices.json", flush=True)
        return [trajectories[i] for i in val_indices]

    # Fallback: deterministic re-derivation
    print(
        "val_indices.json not found — re-deriving the train/val split from "
        f"seed={config.get('random_seed', 42)}, val_split={config.get('val_split', 0.15)}.",
        flush=True,
    )
    np.random.seed(config.get('random_seed', 42))
    n_trajs = len(trajectories)
    n_val   = max(1, int(n_trajs * config.get('val_split', 0.15)))
    perm    = np.random.permutation(n_trajs)
    return [trajectories[i] for i in perm[:n_val]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot autoencoder reconstructions from a trained checkpoint.")
    parser.add_argument(
        "--model_dir",
        default="data/models/autoencoder",
        help="Directory containing best_autoencoder.pt, best_config.json, val_indices.json",
    )
    parser.add_argument("--n_beats",  type=int, default=5, help="Number of consecutive wingbeats to plot")
    parser.add_argument("--seed",     type=int, default=None,
                        help="Random seed for trajectory choice. If unset, a fresh random seed is drawn "
                             "and recorded in the output filename so the plot stays reproducible.")
    parser.add_argument("--save_dir", default=None,
                        help="Where to save plots. Default: the model's own directory (same dir as the checkpoint).")
    args = parser.parse_args()

    # Resolve to the latest run_<timestamp>/ if the parent directory was given
    model_dir   = _resolve_model_dir(args.model_dir)

    # --- Load checkpoint and the config it was trained with ---
    ckpt_path   = os.path.join(model_dir, "best_autoencoder.pt")
    config_path = os.path.join(model_dir, "best_config.json")
    ckpt        = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    with open(config_path) as f:
        config = json.load(f)

    # --- Load data ---
    trajectories = np.load(config['data_path'], allow_pickle=True)
    template     = np.load(config['template_path'])

    # --- Validation trajectories only (avoids training-set leakage) ---
    val_trajs = _load_val_trajectories(model_dir, trajectories, config)

    # --- Rebuild the model from the checkpoint's architecture metadata ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = WingbeatAutoencoder(
        latent_dim          = ckpt['latent_dim'],
        activation          = ckpt.get('activation', 'gelu'),
        dropout             = ckpt.get('dropout', 0.0),
        base_channels       = ckpt.get('base_channels', 128),       # legacy default
        bottleneck_len      = ckpt.get('bottleneck_len', 12),       # legacy default
        decoder_kernel_size = ckpt.get('decoder_kernel_size', 5),   # legacy default
        output_len          = ckpt['output_len'],
    )
    model.load_state_dict(ckpt['state_dict'])
    model.to(device)
    print(
        f"Loaded model: latent_dim={ckpt['latent_dim']} "
        f"activation={ckpt.get('activation', 'gelu')} val_loss={ckpt.get('val_loss', 'unknown')}",
        flush=True,
    )

    # --- Output directory ---
    # Default to the model's own directory so plots live alongside the checkpoint they came from.
    save_dir = args.save_dir if args.save_dir is not None else model_dir
    os.makedirs(save_dir, exist_ok=True)

    # Resolve the seed. If the user didn't pass one, draw a random one and record it so
    # the filename always carries the seed that produced the plot.
    seed = args.seed if args.seed is not None else int(np.random.randint(0, 2**31 - 1))
    if args.seed is None:
        print(f"No --seed given; using random seed={seed}", flush=True)

    save_path = os.path.join(save_dir, f"reconstruction_seed{seed}.png")
    _plot_reconstructed_trajectory(
        model     = model,
        val_trajs = val_trajs,
        template  = template,
        save_path = save_path,
        device    = device,
        n_beats   = args.n_beats,
        seed      = seed,
    )
    print(f"\nDone. Plots saved in: {save_dir}", flush=True)


if __name__ == '__main__':
    main()
