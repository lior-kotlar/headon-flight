"""
End-to-end evaluation of the body→latent regressor composed with the frozen
autoencoder decoder. Loads a BodyToWingbeat checkpoint pair and produces a
suite of interpretable, visualization-friendly metrics on the regressor's
validation set (= the AE val trajectories the regressor inherited):

  1. Retrieval / percentile metrics: for each val example, the predicted latent
     is ranked against all val target latents; we report top-1/10/100 accuracy,
     median rank, median percentile, and a histogram of ranks.

  2. Decoded RMSE in degrees: decode(predicted_latent) vs decode(true_latent).
     This is the regressor's *added* error in physical units (deg per L/R wing
     angle channel), conditional on a perfect decoder.

  3. Per-latent-dim R^2: how much of each latent dimension's variance the
     regressor explains. Tells you which dims are predictable from body
     kinematics and which are noise.

  4. Duration metrics: predicted-vs-true scatter, MAE in samples.

  5. Error-floor decomposition: AE floor (decode(true_latent) vs true wingbeat
     at length L), regressor's added error (decode(pred) vs decode(true)), and
     end-to-end (decode(pred) vs true wingbeat). Computed by aligning the
     regressor dataset to wingbeats_L<L>.npz via (trajectory_id, position).

  6. Reconstruction example plot: a random val trajectory's wingbeats with
     ground-truth, decode(true_latent), and decode(pred_latent) overlaid so
     the visual quality is legible at a glance.

Outputs land in <regressor_dir>/eval/, mirroring the autoencoder's convention.

Run from project root:
    python code/evaluate_body_to_wingbeat.py
    python code/evaluate_body_to_wingbeat.py \\
        --regressor_dir   data/models/body_latent_regressor/run_xxx \\
        --autoencoder_dir data/models/autoencoder/run_yyy
"""

import argparse
import json
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from body_to_wingbeat import (
    BodyToWingbeat,
    _resolve_run_dir,
    load_body_to_wingbeat,
)
from body_latent_regressor import _load_split
from data_handling.body_features import BODY_CHANNEL_NAMES
from data_handling.bucket_eval import (
    WING_ANGLE_LABELS,
    WING_ANGLE_SCALE,
    channel_rmse_to_degrees,
    sa_to_lr_norm,
)


# ---------------------------------------------------------------------------
# Metric functions (pure — caller provides numpy arrays, gets numpy/dict back)
# ---------------------------------------------------------------------------


def compute_retrieval_metrics(
    pred_latents: np.ndarray,
    true_latents: np.ndarray,
    top_ks: tuple[int, ...] = (1, 5, 10, 50, 100),
) -> dict:
    """
    For each row i, rank true_latents[i] among all candidates in true_latents
    by L2 distance from pred_latents[i]. rank=0 means the true target is the
    predicted nearest neighbor (perfect retrieval).

    The candidate pool is the entire val set (true_latents itself), so the
    smallest possible rank is 0 and the largest is N-1.

    Returns a dict with per-example ranks/percentiles and aggregate metrics.
    """
    if pred_latents.shape != true_latents.shape:
        raise ValueError(
            f"shape mismatch: pred={pred_latents.shape} true={true_latents.shape}"
        )
    N, D = pred_latents.shape

    # Pairwise squared distances: (N, N). For N up to ~20k this is fine in float32.
    diff   = pred_latents[:, None, :] - true_latents[None, :, :]              # (N, N, D)
    sq_dst = np.einsum("ijk,ijk->ij", diff, diff)                              # (N, N)
    # Distance from each prediction to its OWN true target.
    self_dst = sq_dst[np.arange(N), np.arange(N)]
    # rank = number of candidates strictly closer than the true target.
    # (Using strict < so ties don't artificially inflate rank.)
    ranks = (sq_dst < self_dst[:, None]).sum(axis=1).astype(np.int64)         # (N,)

    percentiles = ranks / max(N - 1, 1)                                       # in [0, 1]
    metrics = {
        "n_val":             int(N),
        "latent_dim":        int(D),
        "median_rank":       int(np.median(ranks)),
        "mean_rank":         float(np.mean(ranks)),
        "median_percentile": float(np.median(percentiles)),
        "mean_percentile":   float(np.mean(percentiles)),
    }
    for k in top_ks:
        if k <= N:
            metrics[f"top{k}_accuracy"] = float((ranks < k).mean())
    return {**metrics, "ranks": ranks, "percentiles": percentiles}


def compute_per_dim_r2(pred_latents: np.ndarray, true_latents: np.ndarray) -> np.ndarray:
    """
    Per-dim coefficient of determination:
        R^2_k = 1 - SS_res_k / SS_tot_k
    where SS_tot uses the val-set mean of true_latents[:, k]. R^2 = 1 → perfect,
    R^2 = 0 → no better than predicting the mean, R^2 < 0 → worse than mean.

    Returns shape (D,).
    """
    true_mean = true_latents.mean(axis=0, keepdims=True)
    ss_res = ((pred_latents - true_latents) ** 2).sum(axis=0)
    ss_tot = ((true_latents - true_mean) ** 2).sum(axis=0)
    # Guard against zero-variance dims (degenerate; shouldn't happen but be safe).
    ss_tot = np.where(ss_tot > 1e-12, ss_tot, 1.0)
    return 1.0 - ss_res / ss_tot


def decoded_rmse_deg(
    decoded_pred: torch.Tensor,
    decoded_ref:  torch.Tensor,
) -> np.ndarray:
    """
    Per-channel RMSE in degrees between two (B, 6, L) tensors ALREADY in the
    per-angle-normalized L/R-residual space [L_phi..R_psi] (channels match
    WING_ANGLE_LABELS). Takes the per-channel MSE, scales to degrees via
    WING_ANGLE_SCALE. Both representations are decoded into this space upstream
    (see _decode_lr), so the metric is identical and directly comparable.
    """
    if decoded_pred.shape != decoded_ref.shape:
        raise ValueError(
            f"shape mismatch: pred={tuple(decoded_pred.shape)} ref={tuple(decoded_ref.shape)}"
        )
    sq = (decoded_pred - decoded_ref).double() ** 2
    mse_per_channel = sq.mean(dim=(0, 2)).cpu().numpy()                       # (6,)
    return channel_rmse_to_degrees(mse_per_channel)


def duration_metrics(pred_dur: np.ndarray, true_dur: np.ndarray) -> dict:
    """MAE in samples + MAPE (%) + scatter-friendly arrays."""
    pred_dur = pred_dur.astype(np.float64)
    true_dur = true_dur.astype(np.float64)
    abs_err = np.abs(pred_dur - true_dur)
    mae     = float(abs_err.mean())
    mape    = float((abs_err / np.maximum(true_dur, 1.0)).mean() * 100.0)
    return {
        "mae_samples":  mae,
        "mape_percent": mape,
        "min_true":     float(true_dur.min()),
        "max_true":     float(true_dur.max()),
    }


# ---------------------------------------------------------------------------
# Ground-truth alignment: regressor dataset ↔ wingbeats_L<L>.npz
# ---------------------------------------------------------------------------


def _align_val_sa_to_regressor(
    npz_path: str,
    splits:   dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each regressor-val row, find the matching SA wingbeat in wingbeats_L<L>.npz.

    Both files store wingbeats in trajectory order; the regressor dataset just
    skips the *last* wingbeat per trajectory (no "next"). So per trajectory_id,
    the j-th regressor row maps to the j-th L-npz row with that same id.

    Returns (sa_val_aligned, mask_into_npz) where sa_val_aligned has shape
    (n_val, 6, L) and mask_into_npz are the indices used (for downstream
    bookkeeping).
    """
    val_idx        = splits["val_idx"]
    reg_traj_ids   = splits["trajectory_ids"][val_idx]                        # (n_val,)

    with np.load(npz_path) as d:
        sa_all   = d["sa_wingbeats"]                                          # (N, 6, L)
        npz_tids = d["trajectory_ids"]                                        # (N,)

    # Position-within-trajectory for each regressor val row, computed against
    # the FULL regressor dataset order (cumcount of each traj id, then sliced
    # to val rows).
    all_traj_ids = splits["trajectory_ids"]
    # cumcount: for each row i, how many earlier rows share its trajectory_id.
    order = np.argsort(all_traj_ids, kind="stable")
    cumcount = np.empty_like(all_traj_ids, dtype=np.int64)
    # For each contiguous run of equal traj_ids (in sort order), assign 0..len-1.
    sorted_ids = all_traj_ids[order]
    run_start = np.concatenate(([True], sorted_ids[1:] != sorted_ids[:-1]))
    run_idx   = np.cumsum(run_start) - 1
    # Build per-row index-within-trajectory: for each sorted row, position is
    # (its position in `order`) − (the position of its run's first element).
    first_in_run = np.zeros(run_idx[-1] + 1, dtype=np.int64)
    first_pos    = np.flatnonzero(run_start)
    first_in_run[:] = first_pos
    cumcount[order] = np.arange(len(all_traj_ids), dtype=np.int64) - first_in_run[run_idx]

    val_within_traj = cumcount[val_idx]                                       # (n_val,)

    # Same cumcount on the L-npz, then look up by (traj_id, pos_in_traj).
    sa_idx = np.empty(len(val_idx), dtype=np.int64)
    # Build a per-trajectory list of row indices in the L-npz, in order.
    npz_traj_to_rows: dict[int, np.ndarray] = {}
    npz_order = np.argsort(npz_tids, kind="stable")
    npz_sorted = npz_tids[npz_order]
    npz_run_start = np.concatenate(([True], npz_sorted[1:] != npz_sorted[:-1]))
    npz_run_ends  = np.concatenate((np.flatnonzero(npz_run_start)[1:], [len(npz_tids)]))
    npz_run_starts = np.flatnonzero(npz_run_start)
    for rs, re in zip(npz_run_starts, npz_run_ends):
        tid = int(npz_sorted[rs])
        npz_traj_to_rows[tid] = npz_order[rs:re]

    missing = 0
    for k, (tid, pos) in enumerate(zip(reg_traj_ids, val_within_traj)):
        rows = npz_traj_to_rows.get(int(tid))
        if rows is None or pos >= len(rows):
            sa_idx[k] = -1
            missing += 1
        else:
            sa_idx[k] = rows[pos]

    if missing:
        raise RuntimeError(
            f"Could not align {missing}/{len(val_idx)} regressor val rows to "
            f"{npz_path}. The L-npz may not match the regressor dataset's source."
        )

    return sa_all[sa_idx], sa_idx


# ---------------------------------------------------------------------------
# Forward pass: predicted latents + durations for all val rows
# ---------------------------------------------------------------------------


def _predict_val(
    bw:     BodyToWingbeat,
    splits: dict,
    device: str,
    batch_size: int = 1024,
) -> dict:
    """Run the regressor on every val row; return latents + durations (numpy)."""
    val_idx = splits["val_idx"]
    body      = splits["body_means"][val_idx]
    next_body = splits["next_body_means"][val_idx]
    true_lat  = splits["target_latents"][val_idx]
    true_dur  = splits["durations"][val_idx]

    pred_lat_chunks: list[np.ndarray] = []
    pred_dur_chunks: list[np.ndarray] = []
    bw.eval()
    with torch.no_grad():
        for s in range(0, len(val_idx), batch_size):
            xb_curr = torch.from_numpy(body     [s:s + batch_size]).to(device)
            xb_next = torch.from_numpy(next_body[s:s + batch_size]).to(device)
            pred_l, pred_d = bw.predict_latent_and_duration(xb_curr, xb_next)
            pred_lat_chunks.append(pred_l.cpu().numpy())
            pred_dur_chunks.append(pred_d.cpu().numpy().astype(np.int64))
    return {
        "pred_latents": np.concatenate(pred_lat_chunks, axis=0),
        "true_latents": true_lat,
        "pred_durations": np.concatenate(pred_dur_chunks, axis=0),
        "true_durations": true_dur.astype(np.int64),
    }


def _decode_lr(bw: BodyToWingbeat, latents: np.ndarray, device: str,
               batch_size: int = 512) -> torch.Tensor:
    """Decode latents → (N, 6, L) in per-angle-normalized L/R-residual space, kept on
    `device`. Works for both representations: 'sa' latents are (N, D) and 'single_wing'
    latents are (N, 2, D); bw.decode_lr_norm dispatches on bw.representation."""
    out: list[torch.Tensor] = []
    bw.decoder.eval()
    with torch.no_grad():
        for s in range(0, latents.shape[0], batch_size):
            z = torch.from_numpy(latents[s:s + batch_size]).to(device)
            out.append(bw.decode_lr_norm(z))
    return torch.cat(out, dim=0)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_retrieval(
    retrieval: dict,
    save_path: str,
) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
    N = retrieval["n_val"]

    ax[0].hist(retrieval["ranks"], bins=min(50, max(N // 50, 10)),
               color="tab:blue", alpha=0.85, edgecolor="white")
    ax[0].set_xlabel("Rank of true latent (0 = perfect retrieval)")
    ax[0].set_ylabel("Count")
    ax[0].set_title(f"Retrieval rank histogram (N={N})")
    ax[0].axvline(retrieval["median_rank"], color="black", lw=1.6, ls="--",
                  label=f"median={retrieval['median_rank']}")
    ax[0].grid(True, alpha=0.4); ax[0].legend()

    ax[1].hist(retrieval["percentiles"], bins=40, range=(0.0, 1.0),
               color="tab:purple", alpha=0.85, edgecolor="white")
    ax[1].set_xlabel("Percentile rank   (0 = best, 1 = worst)")
    ax[1].set_ylabel("Count")
    ax[1].set_title("Retrieval percentile distribution")
    ax[1].axvline(retrieval["median_percentile"], color="black", lw=1.6, ls="--",
                  label=f"median={retrieval['median_percentile']:.3f}")
    ax[1].grid(True, alpha=0.4); ax[1].legend()

    # Annotate top-K accuracies in the corner of the rank panel.
    topk_lines = [f"top{k.split('_')[0][3:]}_acc = {retrieval[k]:.3f}"
                  for k in retrieval if k.startswith("top") and k.endswith("_accuracy")]
    if topk_lines:
        ax[0].text(0.98, 0.97, "\n".join(topk_lines),
                   transform=ax[0].transAxes, ha="right", va="top",
                   fontsize=9, family="monospace",
                   bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                             edgecolor="0.5", alpha=0.9))

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → wrote {save_path}", flush=True)


def plot_per_channel_rmse_bars(
    rmse_groups: dict[str, np.ndarray],
    save_path:   str,
    title:       str,
) -> None:
    """rmse_groups: { 'AE floor': rmse_deg(6,), 'Regressor add': ..., 'End-to-end': ... }
    Plots grouped bars per channel; missing groups are skipped silently."""
    labels  = list(WING_ANGLE_LABELS)
    groups  = list(rmse_groups.keys())
    n_groups = len(groups)
    x = np.arange(len(labels))
    width = 0.8 / max(n_groups, 1)
    colors = ["#5c9ad6", "#d6a05c", "#a85a5a", "#7eaf73"]

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    for i, name in enumerate(groups):
        vals = rmse_groups[name]
        bars = ax.bar(x + (i - (n_groups - 1) / 2) * width, vals, width,
                      label=name, color=colors[i % len(colors)], alpha=0.9)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.2f}", xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("RMSE [deg]")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.4)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → wrote {save_path}", flush=True)


def plot_error_floor_summary(
    floor_mean:   float | None,
    added_mean:   float,
    e2e_mean:     float | None,
    save_path:    str,
) -> None:
    """One bar per error component, in degrees (averaged across all 6 channels)."""
    rows = [("Regressor added", added_mean, "#d6a05c")]
    if floor_mean is not None:
        rows.insert(0, ("AE floor", floor_mean, "#5c9ad6"))
    if e2e_mean is not None:
        rows.append(("End-to-end", e2e_mean, "#a85a5a"))

    labels = [r[0] for r in rows]
    vals   = [r[1] for r in rows]
    colors = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    bars = ax.bar(labels, vals, color=colors, alpha=0.92)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.2f}°",
                    xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=10)
    ax.set_ylabel("RMSE [deg]   (avg over 6 wing-angle channels)")
    ax.set_title("Error decomposition: AE floor / regressor add / end-to-end")
    ax.grid(True, axis="y", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → wrote {save_path}", flush=True)


def plot_per_dim_r2(r2: np.ndarray, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(max(6.0, 0.4 * len(r2) + 1.5), 4.0))
    colors = ["#5c9ad6" if v >= 0 else "#a85a5a" for v in r2]
    ax.bar(np.arange(len(r2)), r2, color=colors, alpha=0.9)
    ax.axhline(0.0, color="0.4", lw=1.0)
    ax.axhline(1.0, color="0.4", lw=0.7, ls=":")
    ax.set_xlabel("Latent dimension index")
    ax.set_ylabel("R²")
    ax.set_title("Per-dim regressor R²   (1 = perfect, 0 = predict-the-mean, <0 = worse than mean)")
    ax.set_xticks(np.arange(len(r2)))
    ax.grid(True, axis="y", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → wrote {save_path}", flush=True)


def plot_duration_scatter(
    pred_dur: np.ndarray,
    true_dur: np.ndarray,
    save_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.scatter(true_dur, pred_dur, s=10, alpha=0.45, color="tab:blue", edgecolor="none")
    lo = min(int(true_dur.min()), int(pred_dur.min())) - 1
    hi = max(int(true_dur.max()), int(pred_dur.max())) + 1
    ax.plot([lo, hi], [lo, hi], color="black", lw=1.0, ls="--", label="y = x")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("True duration [samples]")
    ax.set_ylabel("Predicted duration [samples]")
    ax.set_title("Wingbeat duration: predicted vs true")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → wrote {save_path}", flush=True)


def plot_reconstruction_examples(
    sa_true:        np.ndarray,                       # (B, 6, L) L/R-norm, ground truth
    sa_decode_true: np.ndarray,                       # (B, 6, L) L/R-norm decode(true_latent)
    sa_decode_pred: np.ndarray,                       # (B, 6, L) L/R-norm decode(pred_latent)
    save_path:      str,
    n_examples:     int = 6,
    seed:           int = 0,
) -> None:
    """
    Random n_examples val wingbeats, three traces per L/R-angle subplot:
      ground truth (solid), decode(true_latent) (dashed, AE-only reconstruction),
      decode(pred_latent) (dotted, end-to-end). All in degrees on the y-axis.

    Inputs are already in per-angle-normalized L/R space (uniform across
    representations); we convert to degrees via WING_ANGLE_SCALE for the y-axis.
    """
    rng = np.random.default_rng(seed)
    n_avail = sa_true.shape[0]
    pick = rng.choice(n_avail, size=min(n_examples, n_avail), replace=False)

    def _to_deg(x: np.ndarray) -> np.ndarray:
        return x[pick] * WING_ANGLE_SCALE[None, :, None] * (180.0 / np.pi)     # (k, 6, L)

    truth_deg = _to_deg(sa_true)
    dec_t_deg = _to_deg(sa_decode_true)
    dec_p_deg = _to_deg(sa_decode_pred)

    k = len(pick)
    L = sa_true.shape[2]
    phase = np.linspace(0.0, 1.0, L)
    n_cols = 3   # one column per angle (φ, θ, ψ)
    fig, axes = plt.subplots(k, n_cols, figsize=(4.0 * n_cols, 2.3 * k), sharex=True)
    if k == 1:
        axes = axes[None, :]

    angle_layout = [("φ", 0, 3), ("θ", 1, 4), ("ψ", 2, 5)]
    for row in range(k):
        for col, (label, lc, rc) in enumerate(angle_layout):
            ax = axes[row, col]
            ax.plot(phase, truth_deg[row, lc], color="tab:blue",  lw=1.6, alpha=0.9, label="Truth L")
            ax.plot(phase, truth_deg[row, rc], color="tab:red",   lw=1.6, alpha=0.9, label="Truth R")
            ax.plot(phase, dec_t_deg[row, lc], color="tab:blue",  lw=1.3, ls="--", alpha=0.9, label="decode(true) L")
            ax.plot(phase, dec_t_deg[row, rc], color="tab:red",   lw=1.3, ls="--", alpha=0.9, label="decode(true) R")
            ax.plot(phase, dec_p_deg[row, lc], color="tab:blue",  lw=1.3, ls=":",  alpha=0.95, label="decode(pred) L")
            ax.plot(phase, dec_p_deg[row, rc], color="tab:red",   lw=1.3, ls=":",  alpha=0.95, label="decode(pred) R")
            ax.grid(True, alpha=0.4)
            if row == 0:
                ax.set_title(label, fontsize=11)
            if row == k - 1:
                ax.set_xlabel("Normalized phase")
            if col == 0:
                ax.set_ylabel(f"example {row}\n[deg]")

    handles = [
        plt.Line2D([], [], color="tab:blue", lw=1.6,            label="Truth L"),
        plt.Line2D([], [], color="tab:red",  lw=1.6,            label="Truth R"),
        plt.Line2D([], [], color="tab:blue", lw=1.3, ls="--",   label="decode(true_latent) L"),
        plt.Line2D([], [], color="tab:red",  lw=1.3, ls="--",   label="decode(true_latent) R"),
        plt.Line2D([], [], color="tab:blue", lw=1.3, ls=":",    label="decode(pred_latent) L"),
        plt.Line2D([], [], color="tab:red",  lw=1.3, ls=":",    label="decode(pred_latent) R"),
    ]
    fig.legend(handles=handles, loc="upper right",
               bbox_to_anchor=(0.998, 0.995), fontsize=8, ncol=1)
    fig.suptitle("Reconstruction examples (val wingbeats)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.86, 0.97])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → wrote {save_path}", flush=True)


# ---------------------------------------------------------------------------
# Programmatic entry point (also called from body_latent_regressor.py post-train)
# ---------------------------------------------------------------------------


def run_evaluation(
    regressor_dir:    str,
    autoencoder_dir:  str,
    dataset_path:     str,
    device:           str = "auto",
    save_dir:         str | None = None,
    n_examples:       int = 6,
    seed:             int = 0,
    npz_path:         str | None = None,
) -> dict:
    """Run the full body→wingbeat eval suite. Returns the sidecar dict and writes
    plots + evaluation.json into save_dir (defaults to <regressor_dir>/eval/).

    regressor_dir / autoencoder_dir may be either a checkpoint directory or its
    parent (the latest run_* will be picked, matching the CLI behavior).
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    r_dir = _resolve_run_dir(regressor_dir,   "best_body_latent_regressor.pt")
    a_dir = _resolve_run_dir(autoencoder_dir, "best_autoencoder.pt")
    print(f"Regressor:   {r_dir}")
    print(f"Autoencoder: {a_dir}")
    print(f"Device:      {device}")

    save_dir = save_dir or os.path.join(r_dir, "eval")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Save dir:    {save_dir}")

    bw = load_body_to_wingbeat(
        os.path.join(r_dir, "best_body_latent_regressor.pt"),
        os.path.join(a_dir, "best_autoencoder.pt"),
        device=device,
    )

    val_indices_path = os.path.join(a_dir, "val_indices.json")
    splits = _load_split(dataset_path, val_indices_path)
    n_val = len(splits["val_idx"])
    print(f"Val wingbeats: {n_val}")
    if n_val == 0:
        raise RuntimeError("Empty val split — check that the AE val_indices.json matches the dataset.")

    representation = bw.representation
    print(f"Representation: {representation}  (n_wings={bw.n_wings})")

    # --- Predictions on the full val set ---
    preds = _predict_val(bw, splits, device=device)
    pred_latents = preds["pred_latents"]            # (N, D) or (N, 2, D)
    true_latents = preds["true_latents"]
    pred_dur     = preds["pred_durations"]
    true_dur     = preds["true_durations"]
    # Retrieval and per-dim R² operate on the flat per-wingbeat latent; for
    # single_wing that is the concatenation [z_L, z_R] (2·D dims).
    pred_flat = pred_latents.reshape(len(pred_latents), -1)
    true_flat = true_latents.reshape(len(true_latents), -1)

    # --- 1. Retrieval / percentile ---
    print("\n=== Retrieval / percentile metrics ===")
    retrieval = compute_retrieval_metrics(pred_flat, true_flat)
    for key in ("n_val", "median_rank", "median_percentile",
                "top1_accuracy", "top10_accuracy", "top100_accuracy"):
        if key in retrieval:
            v = retrieval[key]
            print(f"  {key:>22s}: {v if not isinstance(v, float) else f'{v:.5f}'}")
    plot_retrieval(retrieval, os.path.join(save_dir, "retrieval.png"))

    # --- 2 / 3. Decoded RMSE in degrees (regressor's added error) + per-dim R^2 ---
    # Decode into per-angle-normalized L/R space (uniform across representations).
    decoded_pred = _decode_lr(bw, pred_latents, device=device)                 # (N_val, 6, L)
    decoded_true = _decode_lr(bw, true_latents, device=device)
    added_per_channel = decoded_rmse_deg(decoded_pred, decoded_true)
    added_mean = float(np.mean(added_per_channel))
    print(f"\n=== Decoded RMSE in degrees (decode(pred) vs decode(true)) ===")
    for label, v in zip(WING_ANGLE_LABELS, added_per_channel):
        print(f"  {label:<8s} {v:7.3f}°")
    print(f"  {'mean':<8s} {added_mean:7.3f}°")

    r2_per_dim = compute_per_dim_r2(pred_flat, true_flat)
    plot_per_dim_r2(r2_per_dim, os.path.join(save_dir, "per_dim_r2.png"))

    # --- 4. Duration ---
    dur_stats = duration_metrics(pred_dur, true_dur)
    print(f"\n=== Duration ===\n  MAE = {dur_stats['mae_samples']:.2f} samples   "
          f"MAPE = {dur_stats['mape_percent']:.1f}%   "
          f"true range = [{int(dur_stats['min_true'])}, {int(dur_stats['max_true'])}]")
    plot_duration_scatter(pred_dur, true_dur,
                          os.path.join(save_dir, "duration_scatter.png"))

    # --- 5. Error-floor decomposition ---
    # AE floor + end-to-end need the ground-truth SA wingbeat for each val latent.
    # Preferred source: the `sa_wingbeats` array stored in the regressor dataset,
    # which is the exact (6, L) normalized SA that was encoded into each latent —
    # aligned 1:1, no positional re-derivation. Fall back to aligning against
    # wingbeats_L<L>.npz only for legacy datasets without it (this fallback is
    # fragile: the npz's filtering differs and can silently misalign rows).
    floor_per_channel = None
    e2e_per_channel   = None
    gt_lr             = None   # (n_val, 6, L) ground truth in per-angle-normalized L/R space

    vi = splits["val_idx"]
    if representation == "single_wing" and "single_wing_left" in splits:
        # Stack the stored left/right single-wing residuals; they ARE the per-angle-
        # normalized L/R channels, so no S/A inversion is needed.
        gt_lr = np.concatenate(
            [splits["single_wing_left"][vi], splits["single_wing_right"][vi]], axis=1
        )                                                                       # (n_val, 6, L)
        gt_source = f"{dataset_path} (stored single_wing_left/right)"
    elif "sa_wingbeats" in splits:
        sa_norm = splits["sa_wingbeats"][vi]                                    # (n_val, 6, L) SA-norm
        gt_lr = sa_to_lr_norm(torch.from_numpy(sa_norm)).numpy()
        gt_source = f"{dataset_path} (stored sa_wingbeats)"
    else:
        ae_config_path = os.path.join(a_dir, "best_config.json")
        if npz_path is None and os.path.exists(ae_config_path):
            with open(ae_config_path) as f:
                ae_config = json.load(f)
            L = int(bw.decoder.output_len)
            data_dir = os.path.dirname(os.path.abspath(ae_config["data_path"]))
            npz_path = os.path.join(data_dir, f"wingbeats_L{L}.npz")
        if npz_path and os.path.exists(npz_path):
            print(f"\nNo stored ground truth in dataset — aligning val ground truth "
                  f"by position from {npz_path} (legacy path; rebuild the dataset to fix).")
            sa_true_aligned, _ = _align_val_sa_to_regressor(npz_path, splits)
            gt_lr = sa_to_lr_norm(torch.from_numpy(sa_true_aligned)).numpy()
            gt_source = npz_path
        else:
            gt_source = None

    if gt_lr is not None:
        gt_lr_t = torch.from_numpy(gt_lr).to(device)
        floor_per_channel = decoded_rmse_deg(decoded_true, gt_lr_t)
        e2e_per_channel   = decoded_rmse_deg(decoded_pred, gt_lr_t)
        print(f"\n=== Error-floor decomposition (RMSE in degrees, averaged over 6 channels) ===")
        print(f"  Ground truth: {gt_source}")
        print(f"  AE floor     = {float(np.mean(floor_per_channel)):.3f}°")
        print(f"  Regressor +  = {added_mean:.3f}°")
        print(f"  End-to-end   = {float(np.mean(e2e_per_channel)):.3f}°")
    else:
        print(f"\nSkipping AE-floor / end-to-end: no stored sa_wingbeats and no "
              f"wingbeats_L<L>.npz found. The regressor's added error is still computed above.")

    # --- Bar charts (per-channel + summary) ---
    rmse_groups: dict[str, np.ndarray] = {"Regressor add": added_per_channel}
    if floor_per_channel is not None:
        rmse_groups = {"AE floor": floor_per_channel, **rmse_groups}
    if e2e_per_channel is not None:
        rmse_groups["End-to-end"] = e2e_per_channel
    plot_per_channel_rmse_bars(
        rmse_groups,
        os.path.join(save_dir, "rmse_per_channel.png"),
        title="Per-channel RMSE in degrees (val set)",
    )
    plot_error_floor_summary(
        floor_mean = (float(np.mean(floor_per_channel)) if floor_per_channel is not None else None),
        added_mean = added_mean,
        e2e_mean   = (float(np.mean(e2e_per_channel))   if e2e_per_channel   is not None else None),
        save_path  = os.path.join(save_dir, "error_decomposition.png"),
    )

    # --- 6. Reconstruction example plot (only if ground truth is available) ---
    if gt_lr is not None:
        plot_reconstruction_examples(
            sa_true        = gt_lr,
            sa_decode_true = decoded_true.cpu().numpy(),
            sa_decode_pred = decoded_pred.cpu().numpy(),
            save_path      = os.path.join(save_dir, "reconstruction_examples.png"),
            n_examples     = n_examples,
            seed           = seed,
        )

    # --- JSON sidecar ---
    sidecar = {
        "evaluated_at":     datetime.now().isoformat(timespec="seconds"),
        "regressor_dir":    r_dir,
        "autoencoder_dir":  a_dir,
        "dataset_path":     dataset_path,
        "ground_truth_source": gt_source,
        "wingbeats_npz":    npz_path if npz_path and os.path.exists(npz_path) else None,
        "n_val":            int(n_val),
        "representation":   representation,
        "n_wings":          int(bw.n_wings),
        "latent_dim":       int(pred_flat.shape[1]),
        "body_feature_indices": [int(i) for i in bw.body_indices.tolist()],
        "body_feature_names":   [BODY_CHANNEL_NAMES[int(i)] for i in bw.body_indices.tolist()],
        "retrieval": {
            "median_rank":       retrieval["median_rank"],
            "mean_rank":         retrieval["mean_rank"],
            "median_percentile": retrieval["median_percentile"],
            "mean_percentile":   retrieval["mean_percentile"],
            **{k: v for k, v in retrieval.items() if k.endswith("_accuracy")},
        },
        "rmse_deg_per_channel": {
            "channel_labels":   list(WING_ANGLE_LABELS),
            "regressor_added":  [float(v) for v in added_per_channel],
            "ae_floor":         (None if floor_per_channel is None
                                 else [float(v) for v in floor_per_channel]),
            "end_to_end":       (None if e2e_per_channel is None
                                 else [float(v) for v in e2e_per_channel]),
        },
        "rmse_deg_mean": {
            "regressor_added":  added_mean,
            "ae_floor":         (None if floor_per_channel is None
                                 else float(np.mean(floor_per_channel))),
            "end_to_end":       (None if e2e_per_channel is None
                                 else float(np.mean(e2e_per_channel))),
        },
        "per_dim_r2":       [float(v) for v in r2_per_dim],
        "duration":         dur_stats,
    }
    sidecar_path = os.path.join(save_dir, "evaluation.json")
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)
    print(f"\n  → wrote {sidecar_path}")
    print(f"\nDone. Outputs in {save_dir}", flush=True)
    return sidecar


# ---------------------------------------------------------------------------
# CLI wrapper
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip(),
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--regressor_dir",    default="data/models/body_latent_regressor")
    parser.add_argument("--autoencoder_dir",  default="data/models/autoencoder")
    parser.add_argument("--dataset_path",     default="data/regressor_dataset/wingbeat_regressor_dataset.npz")
    parser.add_argument("--device",           default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--save_dir",         default=None,
                        help="Where to write the eval outputs. Default: <regressor_dir>/eval/")
    parser.add_argument("--n_examples",       type=int, default=6,
                        help="How many val wingbeats to draw in the reconstruction-examples plot.")
    parser.add_argument("--seed",             type=int, default=0,
                        help="RNG seed for the example picker.")
    parser.add_argument("--npz_path",         default=None,
                        help="Path to wingbeats_L<L>.npz. Default: derived from the AE config's "
                             "data_path (data/wingbeats_L<output_len>.npz next to trajectories.npy).")
    args = parser.parse_args()

    run_evaluation(
        regressor_dir   = args.regressor_dir,
        autoencoder_dir = args.autoencoder_dir,
        dataset_path    = args.dataset_path,
        device          = args.device,
        save_dir        = args.save_dir,
        n_examples      = args.n_examples,
        seed            = args.seed,
        npz_path        = args.npz_path,
    )


if __name__ == "__main__":
    main()
