"""
Cross-run comparison: fixed_len vs autoencoder performance.

Reads <model_dir>/grid_search_summary.json (written by autoencoder.py at the
end of every grid search) and produces a strip plot: one column per fixed_len
value, one dot per run (one run = one seed × every other grid combo), short
horizontal tick at the per-L mean, dashed line connecting the means so the
trend across L is visible at a glance.

Y axis defaults to mean validation RMSE in degrees across the six wing-angle
channels — physically interpretable. Pass --metric val_loss for the raw
normalized MSE used during training.

Run from the project root:

    python code/plot_fixed_len_vs_performance.py --model_dir <run_dir>
    python code/plot_fixed_len_vs_performance.py --model_dir <run_dir> --metric val_loss
    python code/plot_fixed_len_vs_performance.py --model_dir <run_dir> --out_path /tmp/plot.png
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_METRIC_CHOICES = ("rmse_deg", "val_loss")


def _extract_metric(entry: dict, metric: str) -> float | None:
    """
    Pull the per-run metric value out of a grid_search_summary.json entry.

    rmse_deg → mean across the six channels of best_val_rmse_deg (a length-6 list).
               Returns None for legacy runs that didn't record per-channel RMSE.
    val_loss → best_val_loss directly (always present).
    """
    if metric == "val_loss":
        v = entry.get("best_val_loss")
        return None if v is None else float(v)
    if metric == "rmse_deg":
        rmse = entry.get("best_val_rmse_deg")
        if rmse is None:
            return None
        return float(np.mean(rmse))
    raise ValueError(f"Unknown metric: {metric}")


def _metric_label(metric: str) -> str:
    return {
        "rmse_deg": "Mean val RMSE [deg] across 6 wing-angle channels",
        "val_loss": "Best val loss (normalized MSE)",
    }[metric]


def plot_fixed_len_vs_performance(
    summary_entries: list[dict],
    out_path:        str,
    metric:          str = "rmse_deg",
    jitter:          float = 0.08,
    seed:            int = 0,
) -> dict[int, list[float]]:
    """
    Group entries by fixed_len, draw the strip plot, return the grouped values
    (so callers can also dump them as JSON if they want).
    """
    grouped: dict[int, list[float]] = defaultdict(list)
    skipped_no_metric = 0
    skipped_no_L      = 0
    for entry in summary_entries:
        L = entry.get("fixed_len")
        if L is None:
            skipped_no_L += 1
            continue
        v = _extract_metric(entry, metric)
        if v is None:
            skipped_no_metric += 1
            continue
        grouped[int(L)].append(v)

    if not grouped:
        raise RuntimeError(
            f"No usable entries for metric={metric!r}. "
            f"Skipped: {skipped_no_L} without fixed_len, "
            f"{skipped_no_metric} without the metric field."
        )

    fixed_lens = sorted(grouped.keys())
    means      = [float(np.mean(grouped[L])) for L in fixed_lens]

    rng = np.random.default_rng(seed)

    fig, ax = plt.subplots(figsize=(max(6.0, 1.4 * len(fixed_lens) + 2.5), 4.8))
    for x, L in enumerate(fixed_lens):
        ys = grouped[L]
        xs = x + rng.uniform(-jitter, +jitter, size=len(ys))
        ax.scatter(xs, ys, s=42, color="tab:blue", alpha=0.78,
                   edgecolor="white", linewidth=0.6, zorder=3)
        # Mean tick — short horizontal segment centered on the column.
        ax.hlines(means[x], xmin=x - 0.22, xmax=x + 0.22,
                  colors="black", lw=2.2, zorder=4)
        # Per-column n annotation.
        ax.annotate(f"n={len(ys)}", xy=(x, means[x]),
                    xytext=(0, 12), textcoords="offset points",
                    ha="center", fontsize=8, color="0.35")

    # Trend line through the means.
    ax.plot(range(len(fixed_lens)), means, color="0.4", linestyle="--", lw=1.0,
            alpha=0.7, zorder=2, label="per-L mean")

    ax.set_xticks(range(len(fixed_lens)))
    ax.set_xticklabels([str(L) for L in fixed_lens])
    ax.set_xlabel("fixed_len (wingbeat length, samples)")
    ax.set_ylabel(_metric_label(metric))
    ax.set_title(
        f"Autoencoder performance vs fixed_len   "
        f"(n_total={sum(len(v) for v in grouped.values())} runs across {len(fixed_lens)} L values)"
    )
    ax.grid(True, axis="y", alpha=0.4)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return dict(grouped)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.strip(), formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model_dir", required=True,
                        help="Top-level grid-search dir, containing grid_search_summary.json "
                             "and the fixed_len_<L>/ subdirs.")
    parser.add_argument("--metric", default="rmse_deg", choices=list(_METRIC_CHOICES),
                        help="Y-axis metric: rmse_deg = mean per-channel val RMSE in degrees "
                             "(default), val_loss = raw normalized MSE used in training.")
    parser.add_argument("--out_path", default=None,
                        help="Where to save the PNG. Default: <model_dir>/fixed_len_vs_<metric>.png")
    parser.add_argument("--jitter", type=float, default=0.08,
                        help="X-jitter half-width for the strip plot (default: 0.08).")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for reproducible jitter (default: 0).")
    args = parser.parse_args()

    summary_path = os.path.join(args.model_dir, "grid_search_summary.json")
    if not os.path.exists(summary_path):
        sys.exit(f"ERROR: no grid_search_summary.json in {args.model_dir}.")
    with open(summary_path) as f:
        summary_entries = json.load(f)

    out_path = args.out_path or os.path.join(args.model_dir, f"fixed_len_vs_{args.metric}.png")

    grouped = plot_fixed_len_vs_performance(
        summary_entries = summary_entries,
        out_path        = out_path,
        metric          = args.metric,
        jitter          = args.jitter,
        seed            = args.seed,
    )

    print(f"  → wrote {out_path}", flush=True)
    print(f"\n  Per-L summary ({args.metric}):", flush=True)
    for L in sorted(grouped.keys()):
        vals = grouped[L]
        print(
            f"    L={L:>4d}  n={len(vals):>3d}  "
            f"mean={np.mean(vals):.5f}  std={np.std(vals):.5f}  "
            f"min={np.min(vals):.5f}  max={np.max(vals):.5f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
