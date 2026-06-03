"""
Per-maneuver-bucket reconstruction eval and the SA↔LR helpers it depends on.

Shared between autoencoder.py (post-training automation) and evaluate_autoencoder.py
(standalone CLI eval). This module deliberately does NOT import from autoencoder.py:
keeping the dependency arrow one-way avoids the circular import that would otherwise
follow from autoencoder.py wanting to call evaluate_by_maneuver_bucket() at the
end of training.

The SA↔LR conversion utilities also live here because both call sites need them
(autoencoder.py's per-epoch val loop reports L/R RMSE in degrees, and bucket_eval
reports the same broken down by bucket).
"""

import json
import os
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from data_handling.maneuver_scoring import BUCKETS, select_score
from transform_data import SA_PHYSICAL_SCALE, _cubic_resample


# Per-angle layout in the (L, 6) wing-angle space: [L_phi, L_theta, L_psi, R_phi, R_theta, R_psi].
_PLOT_ANGLES = [
    ("Stroke φ (deg)",    0, 3),
    ("Deviation θ (deg)", 1, 4),
    ("Rotation ψ (deg)",  2, 5),
]


# Channel labels matching (B, 6, L) tensors AFTER converting S/A → L/R residuals:
# [L_phi, L_theta, L_psi, R_phi, R_theta, R_psi].
WING_ANGLE_LABELS = ('L_phi', 'L_theta', 'L_psi', 'R_phi', 'R_theta', 'R_psi')
# Per-angle physical scale (radians): φ/ψ scale by π, θ by 0.5. Applied to both L and R.
WING_ANGLE_SCALE  = np.array([np.pi, 0.5, np.pi, np.pi, 0.5, np.pi], dtype=np.float64)


def sa_to_lr_norm(x: torch.Tensor) -> torch.Tensor:
    """
    Convert a (B, 6, L) tensor in SA_PHYSICAL_SCALE-normalized S/A coordinates
    into the corresponding L/R-residual tensor, normalized so each L/R channel
    has the same physical scale as its angle (WING_ANGLE_SCALE).

    Derivation: s_norm[c] = S[c] / scale[c], a_norm[c] = A[c] / scale[c], and
        L_residual = S + A    →    L_norm = s_norm + a_norm
        R_residual = S - A    →    R_norm = s_norm - a_norm
    Relies on SA_PHYSICAL_SCALE being symmetric across S and A for each angle
    (π for φ/ψ, 0.5 for θ), which holds by construction in transform_data.py.
    """
    s = x[:, :3, :]
    a = x[:, 3:, :]
    return torch.cat([s + a, s - a], dim=1)


def channel_rmse_to_degrees(mse_per_channel: np.ndarray) -> np.ndarray:
    """
    Convert per-channel MSE measured in WING_ANGLE_SCALE-normalized L/R space
    into per-channel RMSE in degrees:
        rmse_rad = sqrt(mse) * WING_ANGLE_SCALE
        rmse_deg = rmse_rad * 180 / pi
    """
    rmse_normalized = np.sqrt(np.asarray(mse_per_channel, dtype=np.float64))
    rmse_rad = rmse_normalized * WING_ANGLE_SCALE
    return rmse_rad * (180.0 / np.pi)


def format_rmse_degrees(rmse_deg: np.ndarray) -> str:
    """Compact per-channel RMSE-degrees readout, L | R split for legibility."""
    l_part = " ".join(f"{d:.2f}" for d in rmse_deg[:3])
    r_part = " ".join(f"{d:.2f}" for d in rmse_deg[3:])
    return f"rmse_deg(L φ/θ/ψ | R φ/θ/ψ)=[{l_part} | {r_part}]  mean={rmse_deg.mean():.2f}"


# ---------------------------------------------------------------------------
# Per-bucket reconstruction eval
# ---------------------------------------------------------------------------


def _run_model_on_bucket(
    model: nn.Module,
    sa_wingbeats_bucket: np.ndarray,   # (n, 6, L)
    device: str,
    batch_size: int = 256,
) -> tuple[float, np.ndarray]:
    """
    Push a batch of wingbeats through the autoencoder in eval mode and accumulate:
      - normalized MSE (training-space dimensionless squared error)
      - per-channel SSE in L/R space for the degrees conversion

    Returns (norm_mse, mse_per_channel_lr) where mse_per_channel_lr is (6,).
    """
    sse_total          = 0.0
    sse_per_channel_lr = torch.zeros(6, device=device, dtype=torch.float64)
    n_elements_total   = 0
    n_elements_per_ch  = 0
    model.eval()
    with torch.no_grad():
        for s in range(0, sa_wingbeats_bucket.shape[0], batch_size):
            x = torch.as_tensor(sa_wingbeats_bucket[s:s + batch_size], dtype=torch.float32, device=device)
            recon = model(x)
            sq_err = (recon - x).double() ** 2
            sse_total        += sq_err.sum().item()
            n_elements_total += x.numel()
            recon_lr  = sa_to_lr_norm(recon)
            target_lr = sa_to_lr_norm(x)
            sq_err_lr = (recon_lr - target_lr).double() ** 2
            sse_per_channel_lr += sq_err_lr.sum(dim=(0, 2))
            n_elements_per_ch  += x.size(0) * x.size(2)
    norm_mse           = sse_total / max(n_elements_total, 1)
    mse_per_channel_lr = (sse_per_channel_lr / max(n_elements_per_ch, 1)).cpu().numpy()
    return float(norm_mse), mse_per_channel_lr


def plot_per_phase_error(
    model: nn.Module,
    npz_path: str,
    val_trajectory_ids: set[int],
    device: str,
    save_dir: str,
    file_prefix: str = "",
    batch_size: int = 256,
) -> dict:
    """
    Per-phase signed reconstruction error across the validation set.

    For each phase position t and L/R-wing-angle channel c, computes:
        mean[c, t] = mean over val wingbeats of (recon - target) in degrees
        std [c, t] = std  over val wingbeats of (recon - target) in degrees

    Emits one figure with 3 subplots (one per angle: φ, θ, ψ), each showing
    L (blue) and R (red) signed-error curves with ±1 std bands and a y=0
    reference line. Bias direction is read off the sign of the mean curve;
    band height shows how consistent that bias is across wingbeats.

    Saves PNG + a JSON sidecar with the mean/std arrays. Returns the sidecar dict.
    """
    d              = np.load(npz_path)
    sa_all         = d["sa_wingbeats"]
    trajectory_ids = d["trajectory_ids"]
    L              = sa_all.shape[2]

    val_mask = np.isin(trajectory_ids, np.array(sorted(val_trajectory_ids), dtype=trajectory_ids.dtype))
    sa_val   = sa_all[val_mask]                                                # (N_val, 6, L)
    n_val    = sa_val.shape[0]
    if n_val == 0:
        raise ValueError("plot_per_phase_error: no validation wingbeats matched the trajectory_ids set.")

    scale_deg = WING_ANGLE_SCALE * (180.0 / np.pi)                             # (6,)

    # Accumulate signed errors in degrees over the val set.
    all_errs_deg = np.empty((n_val, 6, L), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for s in range(0, n_val, batch_size):
            x       = torch.as_tensor(sa_val[s:s + batch_size], dtype=torch.float32, device=device)
            recon   = model(x)
            diff_lr = (sa_to_lr_norm(recon) - sa_to_lr_norm(x)).cpu().numpy()  # (B, 6, L)
            all_errs_deg[s:s + diff_lr.shape[0]] = (
                diff_lr.astype(np.float64) * scale_deg[None, :, None] * 1.0
            ).astype(np.float32)

    mean_deg = all_errs_deg.mean(axis=0)                                       # (6, L)
    std_deg  = all_errs_deg.std (axis=0)                                       # (6, L)

    phase = np.linspace(0.0, 1.0, L)
    fig, axes = plt.subplots(len(_PLOT_ANGLES), 1, figsize=(9.5, 8.0), sharex=True)
    fig.suptitle(
        f"Per-phase signed reconstruction error  (n_val_wingbeats={n_val})",
        fontsize=14,
    )
    for row, (angle_label, L_col, R_col) in enumerate(_PLOT_ANGLES):
        ax = axes[row]
        # y=0 reference: bias direction is read off the sign relative to this line.
        ax.axhline(0.0, color="0.4", linestyle="--", lw=1.0, alpha=0.7)

        ax.plot(phase, mean_deg[L_col], color="tab:blue", lw=1.6, label="Left wing")
        ax.fill_between(
            phase,
            mean_deg[L_col] - std_deg[L_col], mean_deg[L_col] + std_deg[L_col],
            color="tab:blue", alpha=0.20, linewidth=0,
        )
        ax.plot(phase, mean_deg[R_col], color="tab:red",  lw=1.6, label="Right wing")
        ax.fill_between(
            phase,
            mean_deg[R_col] - std_deg[R_col], mean_deg[R_col] + std_deg[R_col],
            color="tab:red", alpha=0.20, linewidth=0,
        )
        ax.set_ylabel(f"{angle_label}\nerror [deg]", fontsize=10)
        ax.grid(True, alpha=0.4)
        if row == 0:
            ax.legend(loc="upper right", fontsize=9)
    axes[-1].set_xlabel("Normalized phase")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    name_stem = f"{file_prefix}per_phase_error" if file_prefix else "per_phase_error"
    os.makedirs(save_dir, exist_ok=True)
    png_path = os.path.join(save_dir, f"{name_stem}.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    sidecar = {
        "n_val_wingbeats":   int(n_val),
        "L":                 int(L),
        "channel_labels":    list(WING_ANGLE_LABELS),
        "mean_deg":          mean_deg.tolist(),
        "std_deg":           std_deg.tolist(),
        "evaluated_at":      datetime.now().isoformat(timespec="seconds"),
    }
    json_path = os.path.join(save_dir, f"{name_stem}.json")
    with open(json_path, "w") as f:
        json.dump(sidecar, f, indent=2)

    print(f"  → wrote {png_path}", flush=True)
    print(f"  → wrote {json_path}", flush=True)
    return sidecar


def _reconstruct_wing_angles_from_normalized_sa(
    sa_norm: np.ndarray,        # (6, L) normalized SA, channels-first
    template_L: np.ndarray,     # (L, 6) golden template resampled to L, radians
) -> np.ndarray:
    """
    Inverse of the fixed-L SA build: undo SA_PHYSICAL_SCALE, split S/A back into
    L/R residuals, add the L-aligned template. Returns (L, 6) wing angles in radians.
    """
    sa = sa_norm.T.astype(np.float64) * SA_PHYSICAL_SCALE        # (L, 6) rad
    S, A = sa[:, :3], sa[:, 3:]
    hat = np.concatenate([S + A, S - A], axis=1)                 # (L, 6) residuals
    return hat + template_L


def _plot_reconstructions_by_bucket(
    model: nn.Module,
    sa_val: np.ndarray,                # (n_val, 6, L) val-set wingbeats (normalized SA)
    scalar_score: np.ndarray,          # (n_val,) per-axis scalar score
    template_path: str,
    L: int,
    score_axis: str,
    device: str,
    out_path: str,
    n_per_bucket: int = 1,
    seed: int = 0,
) -> None:
    """
    For each bucket, picks `n_per_bucket` random wingbeats and plots their
    originals + the model's reconstructions on top of the golden template.
    Layout: 5 rows (buckets) × 3 cols (angles). One PNG.
    """
    template_native = np.load(template_path)                              # (template_res, 6) rad
    template_L      = _cubic_resample(template_native, L).astype(np.float64)
    template_deg    = np.rad2deg(template_L)
    phase           = np.linspace(0.0, 1.0, L)

    rng = np.random.default_rng(seed)
    n_rows = len(BUCKETS)
    n_cols = len(_PLOT_ANGLES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 2.4 * n_rows), sharex=True)
    fig.suptitle(
        f"Per-bucket reconstruction examples  "
        f"(score_axis={score_axis!r}, n_per_bucket={n_per_bucket}, seed={seed})",
        fontsize=14,
    )

    model.eval()
    for row, (bucket_name, predicate, bucket_desc) in enumerate(BUCKETS):
        bucket_mask = predicate(scalar_score)
        idx_pool    = np.flatnonzero(bucket_mask)
        n_total     = idx_pool.size
        if n_total == 0:
            for col in range(n_cols):
                ax = axes[row, col]
                ax.text(0.5, 0.5, "(no wingbeats)", ha="center", va="center",
                        transform=ax.transAxes, fontsize=10, color="gray")
                ax.set_yticks([])
            axes[row, 0].set_ylabel(f"{bucket_name}\n{bucket_desc}\nn=0")
            continue

        k    = min(n_per_bucket, n_total)
        pick = rng.choice(idx_pool, size=k, replace=False)
        with torch.no_grad():
            x     = torch.as_tensor(sa_val[pick], dtype=torch.float32, device=device)
            recon = model(x).cpu().numpy()                                # (k, 6, L)

        orig_wings  = [_reconstruct_wing_angles_from_normalized_sa(sa_val[i],  template_L) for i in pick]
        recon_wings = [_reconstruct_wing_angles_from_normalized_sa(recon[j],   template_L) for j in range(k)]

        for col, (angle_label, L_col, R_col) in enumerate(_PLOT_ANGLES):
            ax = axes[row, col]
            # Single template line (L/R averaged — they're near-identical at the
            # template level), drawn faint so the per-wingbeat traces stay primary.
            template_curve = 0.5 * (template_deg[:, L_col] + template_deg[:, R_col])
            ax.plot(phase, template_curve, color="0.45", lw=1.4, alpha=0.7, label="Template")
            for orig, rec in zip(orig_wings, recon_wings):
                ax.plot(phase, np.rad2deg(orig[:, L_col]), color="tab:blue", lw=1.4, alpha=0.85, label="Orig L")
                ax.plot(phase, np.rad2deg(orig[:, R_col]), color="tab:red",  lw=1.4, alpha=0.85, label="Orig R")
                ax.plot(phase, np.rad2deg(rec [:, L_col]), color="tab:blue", lw=1.4, alpha=0.85, linestyle="--", label="Recon L")
                ax.plot(phase, np.rad2deg(rec [:, R_col]), color="tab:red",  lw=1.4, alpha=0.85, linestyle="--", label="Recon R")
            ax.grid(True, alpha=0.4)
            if row == 0:
                ax.set_title(angle_label, fontsize=11)
            if row == n_rows - 1:
                ax.set_xlabel("Normalized phase")
            if col == 0:
                ax.set_ylabel(f"{bucket_name}\n{bucket_desc}\nn={n_total}", fontsize=10)

    legend_handles = [
        plt.Line2D([], [], color="0.45",     lw=1.4, alpha=0.7,                       label="Template"),
        plt.Line2D([], [], color="tab:blue", lw=1.4,                                  label="Orig L"),
        plt.Line2D([], [], color="tab:red",  lw=1.4,                                  label="Orig R"),
        plt.Line2D([], [], color="tab:blue", lw=1.4, linestyle="--",                  label="Recon L"),
        plt.Line2D([], [], color="tab:red",  lw=1.4, linestyle="--",                  label="Recon R"),
    ]
    fig.legend(handles=legend_handles, loc="upper right",
               bbox_to_anchor=(0.998, 0.985), fontsize=9, ncol=1)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def evaluate_by_maneuver_bucket(
    model: nn.Module,
    npz_path: str,
    val_trajectory_ids: set[int],
    score_axis: str,
    device: str,
    save_dir: str,
    file_prefix: str = "",
    print_table: bool = True,
    template_path: str | None = None,
    recon_n_per_bucket: int = 1,
    recon_seed: int = 0,
) -> dict:
    """
    Bucket validation wingbeats by their maneuver score on the requested axis,
    push each bucket through the model, and emit a table + JSON sidecar + bar
    chart. Returns the report dict.

    file_prefix: when non-empty, prepended to the output filenames. Used when
        running per grid-search configuration so files don't collide.
    print_table: set False to suppress console output (autoencoder.py post-train
        hook leaves it True so per-config numbers show up in the training log).
    """
    d              = np.load(npz_path)
    sa_all         = d["sa_wingbeats"]            # (N, 6, L) float32
    trajectory_ids = d["trajectory_ids"]          # (N,) int32

    sidecar_path = os.path.splitext(npz_path)[0] + ".json"
    with open(sidecar_path) as f:
        sidecar = json.load(f)
    channels = list(sidecar["maneuver_channel_labels"])
    maneuver_scores = d["maneuver_scores"]        # (N, C)

    val_mask = np.isin(trajectory_ids, np.array(sorted(val_trajectory_ids), dtype=trajectory_ids.dtype))
    sa_val          = sa_all[val_mask]
    scores_val_full = maneuver_scores[val_mask]
    scalar_score    = select_score(scores_val_full, score_axis, channels)

    rows: list[dict] = []
    for bucket_name, predicate, bucket_desc in BUCKETS:
        bucket_mask = predicate(scalar_score)
        n_in_bucket = int(bucket_mask.sum())
        if n_in_bucket == 0:
            rows.append({
                "bucket": bucket_name, "description": bucket_desc, "n_wingbeats": 0,
                "norm_mse": None, "rmse_deg_per_channel": None, "mean_rmse_deg": None,
            })
            continue
        norm_mse, mse_per_channel_lr = _run_model_on_bucket(model, sa_val[bucket_mask], device=device)
        rmse_deg = channel_rmse_to_degrees(mse_per_channel_lr)
        rows.append({
            "bucket":               bucket_name,
            "description":          bucket_desc,
            "n_wingbeats":          n_in_bucket,
            "norm_mse":             float(norm_mse),
            "rmse_deg_per_channel": [float(v) for v in rmse_deg],
            "mean_rmse_deg":        float(rmse_deg.mean()),
        })

    if print_table:
        title = f"score_axis={score_axis!r}, val_wingbeats={int(val_mask.sum())}"
        if file_prefix:
            title = f"{file_prefix}  |  " + title
        print(f"\n=== Per-maneuver-bucket eval  ({title}) ===", flush=True)
        header = (f"  {'bucket':<6} {'description':<18} {'n':>6}  {'norm_mse':>10}  "
                  f"{'L φ/θ/ψ (deg)':<18}  {'R φ/θ/ψ (deg)':<18}  {'mean_deg':>8}")
        print(header)
        print(f"  {'-'*6} {'-'*18} {'-'*6}  {'-'*10}  {'-'*18}  {'-'*18}  {'-'*8}")
        for r in rows:
            if r["n_wingbeats"] == 0:
                print(f"  {r['bucket']:<6} {r['description']:<18} {0:>6}  {'--':>10}  {'--':<18}  {'--':<18}  {'--':>8}")
                continue
            rmse = r["rmse_deg_per_channel"]
            l_part = " ".join(f"{v:5.2f}" for v in rmse[:3])
            r_part = " ".join(f"{v:5.2f}" for v in rmse[3:])
            print(f"  {r['bucket']:<6} {r['description']:<18} {r['n_wingbeats']:>6}  "
                  f"{r['norm_mse']:>10.6f}  {l_part:<18}  {r_part:<18}  {r['mean_rmse_deg']:>8.2f}")

    report = {
        "score_axis":         score_axis,
        "channel_labels":     list(WING_ANGLE_LABELS),
        "n_val_wingbeats":    int(val_mask.sum()),
        "n_val_trajectories": int(len(val_trajectory_ids)),
        "evaluated_at":       datetime.now().isoformat(timespec="seconds"),
        "buckets":            rows,
    }

    name_stem = f"{file_prefix}bucket_eval_{score_axis}" if file_prefix else f"bucket_eval_{score_axis}"
    os.makedirs(save_dir, exist_ok=True)
    json_path = os.path.join(save_dir, f"{name_stem}.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    labels = [r["bucket"]                                            for r in rows]
    means  = [r["mean_rmse_deg"] if r["mean_rmse_deg"] is not None else 0.0 for r in rows]
    counts = [r["n_wingbeats"]                                       for r in rows]
    bars   = ax.bar(labels, means, color=["#5c9ad6", "#7eaf73", "#d6a05c", "#d67d5c", "#a85a5a"])
    for bar, count in zip(bars, counts):
        ax.annotate(f"n={count}" if count > 0 else "n=0",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    ax.set_ylabel("Mean reconstruction RMSE [deg]  (avg over 6 wing-angle channels)")
    ax.set_xlabel("Maneuver bucket")
    title = f"Per-bucket reconstruction RMSE  (score_axis={score_axis!r})"
    if file_prefix:
        title = f"{file_prefix.rstrip('_')}  |  " + title
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.4)
    fig.tight_layout()
    chart_path = os.path.join(save_dir, f"{name_stem}.png")
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Per-bucket reconstruction plot (only when template path is supplied) ---
    recon_path: str | None = None
    if template_path is not None and os.path.exists(template_path):
        recon_path = os.path.join(save_dir, f"{name_stem}_reconstructions.png")
        L = int(sidecar["L"])
        _plot_reconstructions_by_bucket(
            model         = model,
            sa_val        = sa_val,
            scalar_score  = scalar_score,
            template_path = template_path,
            L             = L,
            score_axis    = score_axis,
            device        = device,
            out_path      = recon_path,
            n_per_bucket  = recon_n_per_bucket,
            seed          = recon_seed,
        )

    if print_table:
        print(f"  → wrote {json_path}", flush=True)
        print(f"  → wrote {chart_path}", flush=True)
        if recon_path:
            print(f"  → wrote {recon_path}", flush=True)
    return report
