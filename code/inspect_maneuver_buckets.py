"""
Visual sanity-check for maneuver scoring.

Buckets wingbeats by their maneuver score and plots a handful of randomly-chosen
wingbeats from each bucket on top of the golden template. Useful for "do high-
score wingbeats actually look like maneuvers?" inspection.

Reads the fixed-length dataset (data/wingbeats_L<L>.npz) and the golden template
(data/analysis/golden_template.npy) and emits one figure per requested axis:
    rows  = score buckets (zero → peak)
    cols  = wing angles (stroke φ, deviation θ, rotation ψ)
Each cell overlays N random wingbeats (left wing in blue, right wing in red)
on top of the golden template (single bold dark-gray line — the template's L/R
columns are near-identical so they're collapsed to one curve for clarity).

Run from the project root:
    python code/inspect_maneuver_buckets.py                          # default: --score_axis max
    python code/inspect_maneuver_buckets.py --score_axis all         # max + yaw + pitch + roll
    python code/inspect_maneuver_buckets.py --score_axis yaw pitch   # any subset
    python code/inspect_maneuver_buckets.py --n_per_bucket 8 --seed 7
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from transform_data import SA_PHYSICAL_SCALE, _cubic_resample
from data_handling.maneuver_scoring import BUCKETS, SCORE_AXIS_CHOICES, expand_score_axis, select_score

# (column-name, L-column-index, R-column-index) for the original wing-angle layout
# [L_phi, L_theta, L_psi, R_phi, R_theta, R_psi].
_ANGLES = [
    ("Stroke φ (deg)",    0, 3),
    ("Deviation θ (deg)", 1, 4),
    ("Rotation ψ (deg)",  2, 5),
]


def _reconstruct_wing_angles(
    sa_norm: np.ndarray,        # (6, L) normalized SA from the npz (channels-first)
    template_L: np.ndarray,     # (L, 6) golden template resampled to L (radians)
) -> np.ndarray:
    """
    Inverse of the fixed-L SA build: undo the SA_PHYSICAL_SCALE normalization,
    split S/A back into L/R residuals, add the L-aligned template. Returns wing
    angles (L, 6) = [L_phi, L_theta, L_psi, R_phi, R_theta, R_psi] in radians.
    """
    sa = sa_norm.T.astype(np.float64) * SA_PHYSICAL_SCALE        # (L, 6) rad
    S, A = sa[:, :3], sa[:, 3:]
    hat_L = S + A
    hat_R = S - A
    hat   = np.concatenate([hat_L, hat_R], axis=1)               # (L, 6)
    return hat + template_L


def plot_maneuver_buckets(
    npz_path: str,
    template_path: str,
    out_path: str,
    n_per_bucket: int = 5,
    seed: int = 0,
    score_axis: str = "max",
) -> None:
    d = np.load(npz_path)
    sa_all          = d["sa_wingbeats"]          # (N, 6, L)
    maneuver_scores = d["maneuver_scores"]       # (N, C)

    sidecar_path = os.path.splitext(npz_path)[0] + ".json"
    with open(sidecar_path) as f:
        sidecar = json.load(f)
    channels = list(sidecar["maneuver_channel_labels"])
    W        = int(sidecar["maneuver_W"])
    L        = int(sidecar["L"])

    template_native = np.load(template_path)                          # (template_res, 6) rad
    template_L      = _cubic_resample(template_native, L).astype(np.float64)  # (L, 6) rad

    score = select_score(maneuver_scores, score_axis, channels)
    rng = np.random.default_rng(seed)

    n_rows = len(BUCKETS)
    n_cols = len(_ANGLES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 2.4 * n_rows), sharex=True)
    fig.suptitle(
        f"Random wingbeats by maneuver bucket  "
        f"(score_axis={score_axis!r}, W={W}, n_per_bucket={n_per_bucket}, seed={seed})",
        fontsize=14,
    )

    phase = np.linspace(0.0, 1.0, L)
    template_deg = np.rad2deg(template_L)

    for row, (name, predicate, description) in enumerate(BUCKETS):
        mask = predicate(score)
        idx_pool = np.flatnonzero(mask)
        n_total = idx_pool.size
        if n_total == 0:
            for col in range(n_cols):
                axes[row, col].text(0.5, 0.5, "(no wingbeats in bucket)",
                                    ha="center", va="center", transform=axes[row, col].transAxes,
                                    fontsize=10, color="gray")
                axes[row, col].set_yticks([])
            axes[row, 0].set_ylabel(f"{name}\n{description}\nn=0")
            continue

        k = min(n_per_bucket, n_total)
        pick = rng.choice(idx_pool, size=k, replace=False)
        wings = [_reconstruct_wing_angles(sa_all[i], template_L) for i in pick]  # list of (L, 6)

        for col, (angle_label, L_col, R_col) in enumerate(_ANGLES):
            ax = axes[row, col]
            # Template — single bold dark-gray line per angle (L/R columns are
            # essentially identical after averaging across wingbeats, so we
            # collapse them to one curve for legibility).
            template_curve = 0.5 * (template_deg[:, L_col] + template_deg[:, R_col])
            ax.plot(phase, template_curve, color="0.25", lw=2.2, label="Template")
            # Sampled wingbeats — translucent blue for L, translucent red for R.
            for w in wings:
                ax.plot(phase, np.rad2deg(w[:, L_col]), color="tab:blue", alpha=0.55, lw=1.1)
                ax.plot(phase, np.rad2deg(w[:, R_col]), color="tab:red",  alpha=0.55, lw=1.1)
            ax.grid(True, alpha=0.4)
            if row == 0:
                ax.set_title(angle_label, fontsize=11)
            if row == n_rows - 1:
                ax.set_xlabel("Normalized phase")
            if col == 0:
                ax.set_ylabel(f"{name}\n{description}\nn={n_total}", fontsize=10)

    # Single legend, top-right of the figure.
    legend_handles = [
        plt.Line2D([], [], color="0.25",     lw=2.2,              label="Template"),
        plt.Line2D([], [], color="tab:blue", lw=1.5, alpha=0.8,   label="L sample"),
        plt.Line2D([], [], color="tab:red",  lw=1.5, alpha=0.8,   label="R sample"),
    ]
    fig.legend(handles=legend_handles, loc="upper right",
               bbox_to_anchor=(0.995, 0.985), fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip(), formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--npz_path",      default="data/wingbeats_L80.npz")
    parser.add_argument("--template_path", default="data/analysis/golden_template.npy")
    parser.add_argument("--out_dir",       default=os.path.join("data", "analysis"),
                        help="Where to write maneuver_buckets_<axis>.png files. Default: data/analysis/")
    parser.add_argument("--n_per_bucket",  type=int, default=1)
    parser.add_argument("--seed",          type=int, default=0)
    parser.add_argument(
        "--score_axis", nargs="+", default=["max"], choices=list(SCORE_AXIS_CHOICES),
        help="One or more of {max, yaw, pitch, roll, all}. 'all' expands to "
             "the other four. Default: max.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for axis in expand_score_axis(args.score_axis):
        out_path = os.path.join(args.out_dir, f"maneuver_buckets_{axis}.png")
        plot_maneuver_buckets(
            npz_path      = args.npz_path,
            template_path = args.template_path,
            out_path      = out_path,
            n_per_bucket  = args.n_per_bucket,
            seed          = args.seed,
            score_axis    = axis,
        )


if __name__ == "__main__":
    main()
