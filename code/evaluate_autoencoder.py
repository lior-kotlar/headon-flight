"""
Standalone evaluation pipeline for a trained autoencoder.

Loads a saved checkpoint and produces two outputs WITHOUT retraining:

  1. Per-maneuver-bucket reconstruction report. For each maneuver score bucket
     (zero / low / mid / high / peak), computes normalized MSE and L/R per-angle
     RMSE in degrees on the validation wingbeats. Prints a table, writes a JSON
     sidecar, and saves a bar chart. Runs automatically when the npz carries
     maneuver_scores; skip it with --no_bucket_eval to recover the old behavior.

  2. Reconstruction plot of randomly-chosen consecutive wingbeats from a
     validation trajectory (this is the original behavior of the script).

Run from the project root:
    python code/evaluate_autoencoder.py
    python code/evaluate_autoencoder.py --score_axis all        # bucket eval on each axis
    python code/evaluate_autoencoder.py --no_bucket_eval        # just the reconstruction plot
    python code/evaluate_autoencoder.py --model_dir data/models/autoencoder --n_beats 5
    python code/evaluate_autoencoder.py --seed 7                # reproducible trajectory choice
"""

import argparse
import json
import os

import numpy as np
import torch

from autoencoder import WingbeatAutoencoder, _plot_reconstructed_trajectory
from data_handling.bucket_eval import (
    evaluate_by_maneuver_bucket,
    plot_per_phase_error,
    plot_phase_range_distributions,
    DEFAULT_PHASE_RANGES,
)
from data_handling.maneuver_scoring import SCORE_AXIS_CHOICES, expand_score_axis


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
    parser.add_argument(
        "--score_axis", nargs="+", default=["all"], choices=list(SCORE_AXIS_CHOICES),
        help="Which score axis to bucket on for the per-bucket eval. 'all' expands to "
             "max + yaw + pitch + roll (separate tables and bar charts). Default: max.",
    )
    parser.add_argument(
        "--no_bucket_eval", action="store_true",
        help="Skip the per-maneuver-bucket eval and only produce the reconstruction plot "
             "(the script's original behavior).",
    )
    parser.add_argument(
        "--npz_path", default=None,
        help="Path to the fixed-length wingbeat npz. Default: derived from the training "
             "config's data_path (data/wingbeats_L<output_len>.npz next to trajectories.npy).",
    )
    parser.add_argument(
        "--phase_ranges", nargs="+", default=None,
        help="One or more phase-window ranges as 'lo,hi' (each in [0, 1]) for the "
             "error-distribution histogram. Default: four equal quarters of the wingbeat.",
    )
    args = parser.parse_args()

    if args.phase_ranges:
        phase_ranges: list[tuple[float, float]] = []
        for r in args.phase_ranges:
            lo_str, hi_str = r.split(",")
            phase_ranges.append((float(lo_str), float(hi_str)))
    else:
        phase_ranges = list(DEFAULT_PHASE_RANGES)

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
    # Default to <model_dir>/eval/ so plots live next to (not on top of) the checkpoint.
    # An explicit --save_dir is taken as-is (no eval/ suffix added).
    save_dir = args.save_dir if args.save_dir is not None else os.path.join(model_dir, "eval")
    os.makedirs(save_dir, exist_ok=True)

    # --- Per-maneuver-bucket eval (default on; --no_bucket_eval to skip) ---
    if not args.no_bucket_eval:
        if args.npz_path is not None:
            npz_path = args.npz_path
        else:
            data_dir = os.path.dirname(os.path.abspath(config['data_path']))
            npz_path = os.path.join(data_dir, f"wingbeats_L{ckpt['output_len']}.npz")
        if not os.path.exists(npz_path):
            print(f"Skipping bucket eval — fixed-length dataset not found at {npz_path}.", flush=True)
        else:
            with np.load(npz_path) as probe:
                has_scores = "maneuver_scores" in probe.files
            if not has_scores:
                print(f"Skipping bucket eval — {npz_path} has no maneuver_scores. "
                      f"Rebuild via transform_data.py --fixed_len {ckpt['output_len']}.", flush=True)
            else:
                val_path = os.path.join(model_dir, "val_indices.json")
                if os.path.exists(val_path):
                    with open(val_path) as f:
                        val_trajectory_ids = set(int(i) for i in json.load(f)["val_indices"])
                else:
                    np.random.seed(config.get('random_seed', 42))
                    perm = np.random.permutation(len(trajectories))
                    n_val = max(1, int(len(trajectories) * config.get('val_split', 0.15)))
                    val_trajectory_ids = set(int(i) for i in perm[:n_val])
                # Per-phase error plot is independent of score_axis, so run it once.
                per_phase = plot_per_phase_error(
                    model              = model,
                    npz_path           = npz_path,
                    val_trajectory_ids = val_trajectory_ids,
                    device             = device,
                    save_dir           = save_dir,
                )
                plot_phase_range_distributions(
                    errors_deg   = per_phase["errors_deg"],
                    phase_ranges = phase_ranges,
                    save_dir     = save_dir,
                )
                for axis in expand_score_axis(args.score_axis):
                    evaluate_by_maneuver_bucket(
                        model              = model,
                        npz_path           = npz_path,
                        val_trajectory_ids = val_trajectory_ids,
                        score_axis         = axis,
                        device             = device,
                        save_dir           = save_dir,
                        template_path      = config['template_path'],
                    )

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
