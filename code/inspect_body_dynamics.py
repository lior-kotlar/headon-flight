"""
Visual sanity check of the per-wingbeat body kinematics the regressor consumes.

Plots the mean body-dynamics vector (the same `body_means` stored in the regressor
dataset) across each trajectory — one trace per channel, grouped by physical vector
(v, a, ω, α) — so spikes or physically implausible jumps are visible at a glance.
This is the cheapest noise check: the accelerations are 2nd-derivative quantities
(see process_data.compute_*_kinematics, savgol window 351), so they're where
tracking noise shows up first.

Outputs (PNG, into data/analysis/body_dynamics_<timestamp>/ by default):
  * per-trajectory panel: v / a / ω / α vs physical time, spikes flagged
  * global histograms of the 6 acceleration channels with robust tail thresholds

Run from project root:
    python code/inspect_body_dynamics.py
    python code/inspect_body_dynamics.py --n_trajectories 8 --select longest
    python code/inspect_body_dynamics.py --trajectory_ids 12 47 103
    python code/inspect_body_dynamics.py --dataset_path data/wingbeat_regressor_dataset_dim16_L69.npz
"""

import argparse
import glob
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data_handling.body_features import BODY_CHANNEL_NAMES

SAMPLING_RATE = 16000  # Hz; matches process_data.SAMPLING_RATE

# The 12-d body vector groups into four physical 3-vectors.
VECTOR_GROUPS = [
    ("linear velocity",      slice(0, 3)),
    ("linear acceleration",  slice(3, 6)),
    ("angular velocity",     slice(6, 9)),
    ("angular acceleration", slice(9, 12)),
]
# Per-component colors, reused across panels.
COMP_COLORS = ["tab:blue", "tab:orange", "tab:green"]
# The 6 acceleration channels (linear + angular) for the global tail histograms.
ACCEL_INDICES = [3, 4, 5, 9, 10, 11]


def _robust_thresholds(x: np.ndarray, k: float) -> tuple[float, float, float]:
    """Return (center, lo, hi) using median ± k·(1.4826·MAD). Robust so genuine
    maneuvers in the bulk don't inflate the spike threshold."""
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    sigma = 1.4826 * mad if mad > 0 else float(x.std())
    return med, med - k * sigma, med + k * sigma


def _default_dataset_path() -> str:
    cands = sorted(glob.glob("data/wingbeat_regressor_dataset_*.npz"))
    if not cands:
        raise FileNotFoundError(
            "No data/wingbeat_regressor_dataset_*.npz found. Pass --dataset_path."
        )
    return cands[-1]


def plot_trajectory(
    body: np.ndarray,
    durations: np.ndarray,
    traj_id: int,
    thresholds: dict[int, tuple[float, float, float]],
    save_path: str,
) -> int:
    """One figure for a single trajectory: four stacked panels (v/a/ω/α), each with
    its three components vs physical time. Flags points outside the robust band.
    Returns the number of flagged (spike) samples in this trajectory."""
    # Physical time axis: cumulative wingbeat duration, centered on each wingbeat.
    dt_ms = durations.astype(np.float64) / SAMPLING_RATE * 1000.0
    t_edges = np.concatenate([[0.0], np.cumsum(dt_ms)])
    t = 0.5 * (t_edges[:-1] + t_edges[1:])   # wingbeat-center time (ms)

    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    n_flagged = 0
    for ax, (group_name, sl) in zip(axes, VECTOR_GROUPS):
        cols = list(range(sl.start, sl.stop))
        for color, c in zip(COMP_COLORS, cols):
            ax.plot(t, body[:, c], color=color, lw=1.4, marker="o", ms=2.5,
                    label=BODY_CHANNEL_NAMES[c])
            _, lo, hi = thresholds[c]
            spike = (body[:, c] < lo) | (body[:, c] > hi)
            if spike.any():
                ax.scatter(t[spike], body[spike, c], color="red", s=40, zorder=5,
                           edgecolor="black", linewidth=0.5)
                n_flagged += int(spike.sum())
        ax.set_ylabel(group_name)
        ax.grid(True, alpha=0.4)
        ax.legend(loc="upper right", fontsize=8, ncol=3)
    axes[-1].set_xlabel("Time within trajectory [ms]")
    fig.suptitle(
        f"Trajectory {traj_id} — per-wingbeat body dynamics "
        f"({len(body)} wingbeats)   red = outside median±k·MAD (potential spike)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return n_flagged


def plot_accel_histograms(body: np.ndarray, k: float, save_path: str) -> None:
    """Global histograms (log count) of the 6 acceleration channels with the robust
    tail thresholds marked — shows how heavy the extreme-acceleration tail is."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, c in zip(axes.ravel(), ACCEL_INDICES):
        x = body[:, c]
        med, lo, hi = _robust_thresholds(x, k)
        ax.hist(x, bins=80, color="tab:blue", alpha=0.85, edgecolor="white", log=True)
        ax.axvline(med, color="black", lw=1.2, ls="-", label=f"median={med:.3g}")
        ax.axvline(lo, color="red", lw=1.0, ls="--")
        ax.axvline(hi, color="red", lw=1.0, ls="--", label=f"±{k:g}·MAD")
        frac = float(((x < lo) | (x > hi)).mean()) * 100.0
        ax.set_title(f"{BODY_CHANNEL_NAMES[c]}   (tail {frac:.2f}%)", fontsize=11)
        ax.set_ylabel("count (log)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Acceleration channels — distribution + robust tail thresholds (val+train)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  → wrote {save_path}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.strip(),
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset_path", default=None,
                   help="Regressor dataset npz (default: latest data/wingbeat_regressor_dataset_*.npz). "
                        "body_means is identical across latent dims, so any one works.")
    p.add_argument("--n_trajectories", type=int, default=6,
                   help="How many trajectories to plot (ignored if --trajectory_ids given).")
    p.add_argument("--select", choices=["longest", "random"], default="longest",
                   help="Pick the longest trajectories (clearest) or a random sample.")
    p.add_argument("--trajectory_ids", type=int, nargs="+", default=None,
                   help="Explicit trajectory ids to plot (overrides --select/--n_trajectories).")
    p.add_argument("--spike_k", type=float, default=5.0,
                   help="Robust spike threshold in units of 1.4826·MAD from the median.")
    p.add_argument("--seed", type=int, default=0, help="Seed for --select random.")
    p.add_argument("--save_dir", default=None,
                   help="Output dir (default: data/analysis/body_dynamics_<timestamp>/).")
    args = p.parse_args()

    dataset_path = args.dataset_path or _default_dataset_path()
    d = np.load(dataset_path)
    body = d["body_means"].astype(np.float64)
    durations = d["durations"].astype(np.int64)
    tids = d["trajectory_ids"].astype(np.int64)
    print(f"Dataset: {dataset_path}  ({len(body)} wingbeats, {len(np.unique(tids))} trajectories)")

    save_dir = args.save_dir or os.path.join(
        "data/analysis", f"body_dynamics_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(save_dir, exist_ok=True)
    print(f"Save dir: {save_dir}")

    # Robust per-channel thresholds, fit on ALL wingbeats so spikes are flagged
    # relative to the population (a real maneuver in one trajectory isn't a spike).
    thresholds = {c: _robust_thresholds(body[:, c], args.spike_k) for c in range(12)}

    # Choose trajectories to plot.
    uniq, counts = np.unique(tids, return_counts=True)
    if args.trajectory_ids is not None:
        chosen = args.trajectory_ids
    elif args.select == "longest":
        chosen = uniq[np.argsort(counts)[::-1][:args.n_trajectories]].tolist()
    else:
        rng = np.random.default_rng(args.seed)
        chosen = rng.choice(uniq, size=min(args.n_trajectories, len(uniq)), replace=False).tolist()

    total_flagged = 0
    for tid in chosen:
        mask = tids == tid
        if not mask.any():
            print(f"  trajectory {tid} not in dataset — skipping", flush=True)
            continue
        n_flag = plot_trajectory(
            body[mask], durations[mask], int(tid), thresholds,
            os.path.join(save_dir, f"trajectory_{int(tid):04d}.png"),
        )
        total_flagged += n_flag
        print(f"  trajectory {int(tid):>4d}: {int(mask.sum()):>3d} wingbeats, {n_flag} flagged samples", flush=True)

    plot_accel_histograms(body, args.spike_k, os.path.join(save_dir, "accel_histograms.png"))

    # Global tail summary per channel — a quick numeric read alongside the plots.
    print("\nPer-channel robust tail fraction (share of wingbeats outside median±"
          f"{args.spike_k:g}·MAD):")
    for c in range(12):
        _, lo, hi = thresholds[c]
        frac = float(((body[:, c] < lo) | (body[:, c] > hi)).mean()) * 100.0
        print(f"  {BODY_CHANNEL_NAMES[c]:>12s}: {frac:5.2f}%")
    print(f"\nDone. {total_flagged} flagged samples across {len(chosen)} trajectories. Plots in {save_dir}")


if __name__ == "__main__":
    main()
