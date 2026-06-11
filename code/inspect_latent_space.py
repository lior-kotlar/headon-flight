"""
Latent-space inspection toolkit for a trained autoencoder.

One script, several independent analyses — each one a "probe" of what the
latent space looks like and what each dim controls. Pick one or more via
--analysis. None of these are auto-triggered during training / evaluation;
all are manual-only exploratory tools.

Available analyses:

  variance              Bar chart of per-dim std(z) across the val set. Big
                        bars = dims the model uses; ~0 bars = dead dims.
                        Also writes the {z_mean, z_std, sort order} JSON.

  traversal             For each latent dim k, sweep z[k] from -range_std to
                        +range_std (linspace), hold all other dims at z_mean,
                        decode each step. One PNG + HTML per dim showing how
                        the reconstruction morphs along that axis.

  distribution_sampling For each latent dim k, sample `--n_samples` values
                        from the empirical distribution of z[:, k] across
                        the val set (sampling with replacement), hold all
                        other dims at z_mean, decode each. Plots the cloud
                        of decoded wingbeats with median + p10/p90 bands so
                        you see what the dim controls in its "natural" range
                        (rather than a uniform linspace). PNG per dim, no HTML.

  histograms            Single PNG with a grid of mini-histograms: one per
                        latent dim, showing the distribution of z[:, k] over
                        the val set, with z_mean marked. Useful for spotting
                        bimodal or skewed dims.

  pca                   Raw-covariance PCA over the encoded wingbeats. Finds the
                        orthogonal directions of greatest variation in latent
                        space (the raw dims usually aren't those directions).
                        Writes a scree / explained-variance plot, a JSON of the
                        components, and — like traversal but along each principal
                        component — one PNG per PC sweeping z_mean ± range_std·σ
                        along that PC and decoding, so you see what coordinated
                        change in wingbeat shape each component controls.

  pca_3d                Interactive 3D scatter (Plotly HTML) of every encoded
                        wingbeat projected onto the top-3 principal components.
                        Writes one scatter per body angular-accel axis (yaw,
                        pitch, roll), each point colored on a continuous gradient
                        by the magnitude of that axis' acceleration
                        (|body_means[:, 9/10/11]|). Shows whether high-maneuver
                        wingbeats occupy a distinct region of latent space. The
                        color scale is clipped at --color_clip_pct (default p99)
                        because the angular accel is heavy-tailed.

  template_vs_mean      Probes encoder nonlinearity by comparing three
                        reference latents against the empirical distribution
                        of encoded val wingbeats:
                          z_mean      — centroid of encoded val set
                          z_template  — encoder(zeros) (SA(template,template)=0)
                          z_zero      — origin of latent space
                        Reports per-dim z-score, full Mahalanobis distance,
                        and percentile rank for each query vs z_mean. Saves
                        a z-score bar chart, a decoded-overlay PNG+HTML, and
                        a JSON sidecar.

  all                   Run every analysis.

Run from the project root:

    python code/inspect_latent_space.py --model_dir <run_dir>
    python code/inspect_latent_space.py --model_dir <run_dir>/latent_dim_8 --analysis variance histograms
    python code/inspect_latent_space.py --model_dir <run_dir> --analysis distribution_sampling --n_samples 200
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Make sibling imports work whether invoked as `python code/inspect_latent_space.py`
# or with code/ on sys.path.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from autoencoder import WingbeatAutoencoder
from transform_data import (
    _cubic_resample,
    SINGLE_WING_PHYSICAL_SCALE,
    fixed_len_dataset_path,
    single_wing_template_path,
)
from data_handling.bucket_eval import _reconstruct_wing_angles_from_normalized_sa


# The three wing angles, in row order. Which channel column holds each angle for a
# given wing differs per representation (see _build_repr_plot).
_ANGLE_ROW_LABELS = ["Stroke φ (deg)", "Deviation θ (deg)", "Rotation ψ (deg)"]


def _reconstruct_single_wing(res_norm: np.ndarray, template_L: np.ndarray) -> np.ndarray:
    """
    Inverse of the fixed-L single-wing build: undo SINGLE_WING_PHYSICAL_SCALE on the
    (3, L) channels-first residual and add the single-wing template. Returns (L, 3)
    wing angles in radians. Mirrors bucket_eval._reconstruct_wing_angles_from_normalized_sa
    for the 6-ch S/A representation.
    """
    res = res_norm.T.astype(np.float64) * SINGLE_WING_PHYSICAL_SCALE      # (L, 3) rad
    return res + template_L


def _build_repr_plot(representation: str, template_L: np.ndarray) -> dict:
    """
    Bundle everything the plotting code needs to turn decoder channel output into a
    wing-angle figure, abstracted over the representation:

      n_channels    — decoder channels (6 for 'sa', 3 for 'single_wing')
      wings         — list of (column_title, [phi_col, theta_col, psi_col]); one entry
                      per subplot column. 'sa' has Left/Right; 'single_wing' has one wing.
      reconstruct   — fn(recon_chan (C, L), template_L) → (L, C) wing angles in radians
      template_L    — (L, C) template resampled to the model's output length, radians
    """
    if representation == "single_wing":
        return {
            "representation": "single_wing",
            "n_channels":     3,
            "wings":          [("Wing", [0, 1, 2])],
            "reconstruct":    _reconstruct_single_wing,
            "template_L":     template_L,
        }
    return {
        "representation": "sa",
        "n_channels":     6,
        "wings":          [("Left wing", [0, 1, 2]), ("Right wing", [3, 4, 5])],
        "reconstruct":    _reconstruct_wing_angles_from_normalized_sa,
        "template_L":     template_L,
    }

_ANALYSIS_CHOICES = (
    "variance",
    "traversal",
    "distribution_sampling",
    "histograms",
    "template_vs_mean",
    "pca",
    "pca_3d",
    "all",
)


# ---------------------------------------------------------------------------
# Shared loading
# ---------------------------------------------------------------------------


def _load_model(model_dir: str, device: str):
    ckpt_path = os.path.join(model_dir, "best_autoencoder.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No best_autoencoder.pt in {model_dir}.")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = WingbeatAutoencoder(
        latent_dim          = ckpt["latent_dim"],
        in_channels         = ckpt.get("in_channels", 6),
        activation          = ckpt.get("activation", "gelu"),
        dropout             = ckpt.get("dropout", 0.0),
        base_channels       = ckpt.get("base_channels", 128),
        bottleneck_len      = ckpt.get("bottleneck_len", 12),
        decoder_kernel_size = ckpt.get("decoder_kernel_size", 5),
        output_len          = ckpt["output_len"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt


def _load_inputs(model_dir: str, device: str, npz_path_override: str | None):
    """
    Common setup shared by every analysis: load model, template, fixed-L wingbeats,
    encode the val set, return a bundle of derived tensors.
    """
    model, ckpt = _load_model(model_dir, device)
    latent_dim     = int(ckpt["latent_dim"])
    output_len     = int(ckpt["output_len"])
    representation = ckpt.get("representation", "sa")
    n_channels     = int(ckpt.get("in_channels", 6))
    lat_prefix     = f"lat_{latent_dim:02d}_"

    with open(os.path.join(model_dir, "best_config.json")) as f:
        config = json.load(f)
    if npz_path_override is not None:
        npz_path = npz_path_override
    else:
        data_dir = os.path.dirname(os.path.abspath(config["data_path"]))
        npz_path = fixed_len_dataset_path(data_dir, output_len, representation)

    # The single-wing representation keys its npz array differently and uses the
    # 3-angle template sibling of the 6-ch golden template.
    if representation == "single_wing":
        array_key     = "single_wing_wingbeats"
        template_path = single_wing_template_path(config["template_path"])
    else:
        array_key     = "sa_wingbeats"
        template_path = config["template_path"]

    template_native = np.load(template_path)
    template_L      = _cubic_resample(template_native, output_len).astype(np.float64)
    repr_plot       = _build_repr_plot(representation, template_L)

    d          = np.load(npz_path)
    sa_all_np  = d[array_key]                                                  # (N, C, L)
    sa_all     = torch.from_numpy(sa_all_np).float().to(device)
    n_val      = sa_all.shape[0]
    print(
        f"Loaded model: representation={representation}, latent_dim={latent_dim}, "
        f"in_channels={n_channels}, output_len={output_len}; encoding {n_val} wingbeats.",
        flush=True,
    )

    with torch.no_grad():
        z_all  = model.encode(sa_all)                                          # (N, latent_dim)
        z_mean = z_all.mean(dim=0)
        z_std  = z_all.std (dim=0)

    return {
        "model":          model,
        "ckpt":           ckpt,
        "latent_dim":     latent_dim,
        "output_len":     output_len,
        "representation": representation,
        "n_channels":     n_channels,
        "repr_plot":      repr_plot,
        "lat_prefix":     lat_prefix,
        "config":         config,
        "npz_path":       npz_path,
        "template_L":     template_L,
        "sa_all":         sa_all,
        "z_all":          z_all,
        "z_mean":         z_mean,
        "z_std":          z_std,
        "n_val":          n_val,
    }


# ---------------------------------------------------------------------------
# Public metric functions (importable from a notebook for à la carte use)
# ---------------------------------------------------------------------------


def per_dim_zscore(
    z_query: torch.Tensor,    # (latent_dim,)
    z_ref:   torch.Tensor,    # (latent_dim,)
    z_std:   torch.Tensor,    # (latent_dim,)
) -> np.ndarray:
    """
    Per-dim signed offset in std-units: (q[k] - ref[k]) / std[k].

    L2 norm of this vector is the *diagonal* Mahalanobis distance (Σ^-1 = diag(1/σ²)).
    """
    safe_std = torch.clamp(z_std, min=1e-12)
    return ((z_query - z_ref) / safe_std).detach().cpu().numpy()


def full_mahalanobis(
    z_query:  torch.Tensor,    # (latent_dim,)
    z_ref:    torch.Tensor,    # (latent_dim,)
    cov_inv:  np.ndarray,      # (latent_dim, latent_dim)
) -> float:
    """Mahalanobis distance using the full covariance — accounts for cross-dim correlations."""
    diff = (z_query - z_ref).detach().cpu().numpy().astype(np.float64)
    return float(np.sqrt(diff @ cov_inv @ diff))


def distance_percentile(
    z_query:  torch.Tensor,    # (latent_dim,)
    z_ref:    torch.Tensor,    # (latent_dim,)
    z_all:    torch.Tensor,    # (N, latent_dim)
) -> tuple[float, float, np.ndarray]:
    """
    Percentile rank of ‖z_query − z_ref‖ in the distribution of ‖z_i − z_ref‖
    over the val set. Returns (percentile, query_distance, all_distances).
    """
    diff = (z_query - z_ref).detach().cpu().numpy()
    d_query = float(np.linalg.norm(diff))
    all_diffs = (z_all - z_ref).detach().cpu().numpy()
    d_all = np.linalg.norm(all_diffs, axis=1)
    pct = float(100.0 * (d_all < d_query).mean())
    return pct, d_query, d_all


# ---------------------------------------------------------------------------
# Analysis: variance
# ---------------------------------------------------------------------------


def _plot_variance_bar(z_std: torch.Tensor, latent_dim: int, out_path: str) -> np.ndarray:
    order = torch.argsort(z_std, descending=True).cpu().numpy()
    sorted_std = z_std[order].cpu().numpy()
    fig, ax = plt.subplots(figsize=(max(6.0, latent_dim * 0.6), 4.0))
    ax.bar(range(latent_dim), sorted_std, color="tab:purple")
    ax.set_xticks(range(latent_dim))
    ax.set_xticklabels([f"d{int(i)}" for i in order], rotation=0, fontsize=9)
    ax.set_ylabel("std(z) across val set")
    ax.set_xlabel("Latent dim (sorted by std)")
    ax.set_title(
        f"Latent dim activity — larger std = used by the model, ~0 = unused  "
        f"(latent_dim={latent_dim})"
    )
    ax.grid(True, axis="y", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return order


def run_variance(out_dir: str, lat_prefix: str, z_mean: torch.Tensor, z_std: torch.Tensor, latent_dim: int) -> None:
    os.makedirs(out_dir, exist_ok=True)
    var_png  = os.path.join(out_dir, f"{lat_prefix}latent_variance.png")
    order    = _plot_variance_bar(z_std, latent_dim, var_png)
    print(f"  → wrote {var_png}", flush=True)
    with open(os.path.join(out_dir, f"{lat_prefix}latent_variance.json"), "w") as f:
        json.dump({
            "z_mean":                  [float(v) for v in z_mean.cpu()],
            "z_std":                   [float(v) for v in z_std.cpu()],
            "sorted_dims_by_std_desc": [int(i)   for i in order],
        }, f, indent=2)


# ---------------------------------------------------------------------------
# Analysis: traversal (deterministic linspace sweep)
# ---------------------------------------------------------------------------


def _plot_traversal_for_dim(
    k:           int,
    z_mean:      torch.Tensor,
    z_std:       torch.Tensor,
    model:       WingbeatAutoencoder,
    repr_plot:   dict,
    n_steps:     int,
    range_std:   float,
    device:      str,
    out_path:    str,
) -> None:
    deltas = torch.linspace(-range_std, range_std, n_steps).to(device)
    z_sweep = z_mean.unsqueeze(0).repeat(n_steps, 1)
    z_sweep[:, k] = z_mean[k] + deltas * z_std[k]
    with torch.no_grad():
        recon_sa = model.decode(z_sweep).cpu().numpy()                          # (n_steps, C, L)

    template_L   = repr_plot["template_L"]
    reconstruct  = repr_plot["reconstruct"]
    wings        = repr_plot["wings"]
    L            = template_L.shape[0]
    phase        = np.linspace(0.0, 1.0, L)
    template_deg = np.rad2deg(template_L)
    colors       = plt.get_cmap("coolwarm")(np.linspace(0.0, 1.0, n_steps))
    mid_idx      = n_steps // 2

    fig, axes = plt.subplots(3, len(wings), figsize=(5.75 * len(wings), 8.5),
                             sharex=True, squeeze=False)
    fig.suptitle(
        f"Latent traversal — dim {k}   (z_mean={float(z_mean[k]):+.3f}, "
        f"z_std={float(z_std[k]):.3f})   ±{range_std:.1f}σ, {n_steps} steps",
        fontsize=13,
    )
    for col_idx, (wing_title, wing_cols) in enumerate(wings):
        for row, angle_label in enumerate(_ANGLE_ROW_LABELS):
            ax       = axes[row, col_idx]
            use_col  = wing_cols[row]
            ax.plot(phase, template_deg[:, use_col],
                    color="0.5", linestyle="--", lw=1.2, alpha=0.7, label="Template")
            for step_idx in range(n_steps):
                wing_angles = reconstruct(recon_sa[step_idx], template_L)
                lw = 2.2 if step_idx == mid_idx else 1.3
                ax.plot(phase, np.rad2deg(wing_angles[:, use_col]),
                        color=colors[step_idx], lw=lw)
            ax.grid(True, alpha=0.4)
            if row == 0:
                ax.set_title(wing_title, fontsize=11)
            if col_idx == 0:
                ax.set_ylabel(angle_label, fontsize=10)
            if row == 2:
                ax.set_xlabel("Normalized phase")

    sm = plt.cm.ScalarMappable(
        cmap="coolwarm", norm=plt.Normalize(vmin=-range_std, vmax=range_std)
    )
    sm.set_array([])
    fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.04, label=f"z[{k}] offset (σ)")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_traversal(
    out_dir:    str,
    lat_prefix: str,
    model:      WingbeatAutoencoder,
    repr_plot:  dict,
    z_mean:     torch.Tensor,
    z_std:      torch.Tensor,
    latent_dim: int,
    n_steps:    int,
    range_std:  float,
    device:     str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for k in range(latent_dim):
        out_path = os.path.join(out_dir, f"{lat_prefix}dim_{k:03d}.png")
        _plot_traversal_for_dim(
            k=k, z_mean=z_mean, z_std=z_std, model=model, repr_plot=repr_plot,
            n_steps=n_steps, range_std=range_std, device=device, out_path=out_path,
        )
        print(f"  → wrote {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Analysis: distribution sampling (bootstrap from empirical z[:,k])
# ---------------------------------------------------------------------------


def _plot_distribution_sampling_for_dim(
    k:            int,
    z_all_k:      np.ndarray,        # (N,) — empirical values of z[:, k]
    z_mean:       torch.Tensor,      # (latent_dim,)
    model:        WingbeatAutoencoder,
    repr_plot:    dict,
    n_samples:    int,
    device:       str,
    rng:          np.random.Generator,
    out_path:     str,
) -> None:
    """
    Sample n_samples scalar values from z_all[:, k] with replacement (bootstrap),
    build full latent vectors where every other entry is z_mean, decode all,
    and plot the cloud of reconstructions with a median + p10/p90 band overlay.
    """
    # 1. Sample n_samples scalars from the empirical 1-D distribution.
    sampled_vals = rng.choice(z_all_k, size=n_samples, replace=True)            # (n_samples,)
    # 2. Build (n_samples, latent_dim) latent matrix.
    z_sweep = z_mean.unsqueeze(0).repeat(n_samples, 1).clone()
    z_sweep[:, k] = torch.from_numpy(sampled_vals).to(device=device, dtype=z_sweep.dtype)
    # 3. Decode in one batched pass.
    with torch.no_grad():
        recon_sa = model.decode(z_sweep).cpu().numpy()                          # (n_samples, C, L)

    template_L   = repr_plot["template_L"]
    reconstruct  = repr_plot["reconstruct"]
    wings        = repr_plot["wings"]
    n_channels   = repr_plot["n_channels"]
    L            = template_L.shape[0]
    phase        = np.linspace(0.0, 1.0, L)
    template_deg = np.rad2deg(template_L)

    # Pre-convert every reconstruction to wing-angle space once, then plot
    # per-cell from the cached (n_samples, L, C) tensor.
    wing_deg = np.empty((n_samples, L, n_channels), dtype=np.float64)
    for i in range(n_samples):
        wing_deg[i] = np.rad2deg(reconstruct(recon_sa[i], template_L))

    fig, axes = plt.subplots(3, len(wings), figsize=(5.75 * len(wings), 8.5),
                             sharex=True, squeeze=False)
    fig.suptitle(
        f"Distribution sampling — dim {k}   "
        f"(z_mean={float(z_mean[k]):+.3f},  "
        f"sampled empirical z[:, {k}]: min={sampled_vals.min():+.3f}, "
        f"max={sampled_vals.max():+.3f}, n={n_samples})",
        fontsize=12,
    )
    last_col = len(wings) - 1
    for col_idx, (wing_title, wing_cols) in enumerate(wings):
        for row, angle_label in enumerate(_ANGLE_ROW_LABELS):
            ax       = axes[row, col_idx]
            use_col  = wing_cols[row]
            ys = wing_deg[:, :, use_col]                                        # (n_samples, L)
            # Spaghetti — all sampled curves, low alpha.
            for i in range(n_samples):
                ax.plot(phase, ys[i], color="tab:blue", lw=0.6, alpha=0.12)
            # Median + p10/p90 band.
            p10 = np.percentile(ys, 10, axis=0)
            p50 = np.percentile(ys, 50, axis=0)
            p90 = np.percentile(ys, 90, axis=0)
            ax.fill_between(phase, p10, p90, color="tab:blue", alpha=0.25, label="p10–p90")
            ax.plot(phase, p50, color="tab:blue", lw=2.0, label="median")
            # Template reference (gray dashed).
            ax.plot(phase, template_deg[:, use_col], color="0.5", linestyle="--", lw=1.2,
                    alpha=0.7, label="Template")
            ax.grid(True, alpha=0.4)
            if row == 0:
                ax.set_title(wing_title, fontsize=11)
            if col_idx == 0:
                ax.set_ylabel(angle_label, fontsize=10)
            if row == 2:
                ax.set_xlabel("Normalized phase")
            if row == 0 and col_idx == last_col:
                ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_distribution_sampling(
    out_dir:    str,
    lat_prefix: str,
    model:      WingbeatAutoencoder,
    repr_plot:  dict,
    z_all:      torch.Tensor,
    z_mean:     torch.Tensor,
    latent_dim: int,
    n_samples:  int,
    device:     str,
    seed:       int,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    z_all_np = z_all.detach().cpu().numpy()                                     # (N, latent_dim)
    rng = np.random.default_rng(seed)
    for k in range(latent_dim):
        out_path = os.path.join(out_dir, f"{lat_prefix}dim_{k:03d}.png")
        _plot_distribution_sampling_for_dim(
            k=k, z_all_k=z_all_np[:, k], z_mean=z_mean, model=model,
            repr_plot=repr_plot, n_samples=n_samples, device=device, rng=rng,
            out_path=out_path,
        )
        print(f"  → wrote {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Analysis: per-dim histograms
# ---------------------------------------------------------------------------


def _plot_histograms_grid(
    z_all:      torch.Tensor,        # (N, latent_dim)
    z_mean:     torch.Tensor,        # (latent_dim,)
    z_std:      torch.Tensor,        # (latent_dim,)
    latent_dim: int,
    out_path:   str,
    n_bins:     int = 40,
) -> None:
    """
    Grid of mini-histograms — one per latent dim — showing the distribution of
    z[:, k] over the val set. Vertical line marks z_mean[k]. Subtitle shows the
    per-dim std so dead dims are obvious.
    """
    z_all_np = z_all.detach().cpu().numpy()                                     # (N, latent_dim)
    z_mean_np = z_mean.detach().cpu().numpy()
    z_std_np  = z_std.detach().cpu().numpy()

    # Roughly square grid.
    n_cols = int(np.ceil(np.sqrt(latent_dim)))
    n_rows = int(np.ceil(latent_dim / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.1 * n_rows),
                             sharex=False, sharey=False)
    axes = np.atleast_2d(axes)
    for k in range(latent_dim):
        r, c = divmod(k, n_cols)
        ax = axes[r, c]
        ax.hist(z_all_np[:, k], bins=n_bins, color="tab:blue", alpha=0.75)
        ax.axvline(z_mean_np[k], color="black", lw=1.2,
                   label=f"z_mean={z_mean_np[k]:+.2f}")
        ax.set_title(f"d{k}  std={z_std_np[k]:.3f}", fontsize=9)
        ax.tick_params(axis="both", labelsize=7)
        ax.grid(True, alpha=0.3)
    # Hide unused cells in the last row.
    for k in range(latent_dim, n_rows * n_cols):
        r, c = divmod(k, n_cols)
        axes[r, c].axis("off")

    fig.suptitle(
        f"Per-dim distribution of z[:, k] over val set  (latent_dim={latent_dim}, "
        f"N={z_all_np.shape[0]})",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_histograms(out_dir: str, lat_prefix: str, z_all: torch.Tensor,
                   z_mean: torch.Tensor, z_std: torch.Tensor, latent_dim: int) -> None:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{lat_prefix}latent_histograms.png")
    _plot_histograms_grid(z_all, z_mean, z_std, latent_dim, out_path)
    print(f"  → wrote {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Analysis: PCA (raw-covariance principal components of the latent space)
# ---------------------------------------------------------------------------


def _compute_pca(z_all: torch.Tensor, z_mean: torch.Tensor) -> dict:
    """
    Raw-covariance PCA over the encoded wingbeats: center on z_mean (no per-dim
    scaling) and SVD the centered matrix. Returns unit components, the std of the
    data along each component (sqrt of the eigenvalue), and explained-variance.
    """
    z_all_np  = z_all.detach().cpu().numpy().astype(np.float64)                  # (N, D)
    z_mean_np = z_mean.detach().cpu().numpy().astype(np.float64)                 # (D,)
    n         = z_all_np.shape[0]
    zc        = z_all_np - z_mean_np
    # full_matrices=False → vt is (min(N,D), D); for N >> D this is (D, D).
    _, s, vt = np.linalg.svd(zc, full_matrices=False)
    eigvals      = (s ** 2) / max(n - 1, 1)                                       # variance along each PC
    sigma_pc     = np.sqrt(eigvals)                                              # std along each PC
    total        = float(eigvals.sum()) if eigvals.sum() > 0 else 1.0
    expl_ratio   = eigvals / total
    return {
        "components":      vt,                          # (n_pc, D), each row a unit PC
        "sigma_pc":        sigma_pc,                    # (n_pc,)
        "explained_var":   eigvals,                     # (n_pc,)
        "explained_ratio": expl_ratio,                  # (n_pc,)
        "cumulative":      np.cumsum(expl_ratio),       # (n_pc,)
        "n_wingbeats":     int(n),
    }


def _plot_pca_scree(pca: dict, out_path: str) -> None:
    ratio = pca["explained_ratio"]
    cum   = pca["cumulative"]
    n_pc  = len(ratio)
    fig, ax = plt.subplots(figsize=(max(6.0, n_pc * 0.6), 4.2))
    ax.bar(range(n_pc), ratio, color="tab:green", alpha=0.8, label="per-PC")
    ax.set_xticks(range(n_pc))
    ax.set_xticklabels([f"PC{k}" for k in range(n_pc)], fontsize=9)
    ax.set_ylabel("explained variance ratio")
    ax.set_xlabel("Principal component")
    ax.grid(True, axis="y", alpha=0.4)
    ax2 = ax.twinx()
    ax2.plot(range(n_pc), cum, color="tab:red", marker="o", lw=1.6, label="cumulative")
    ax2.set_ylabel("cumulative explained variance", color="tab:red")
    ax2.set_ylim(0.0, 1.02)
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax.set_title(
        f"Latent PCA scree — raw covariance over {pca['n_wingbeats']} wingbeats "
        f"({n_pc} components)"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_pca_traversal_for_pc(
    k:           int,
    pca:         dict,
    z_mean:      torch.Tensor,
    model:       WingbeatAutoencoder,
    repr_plot:   dict,
    n_steps:     int,
    range_std:   float,
    device:      str,
    out_path:    str,
) -> None:
    """
    Sweep z_mean ± range_std·σ along principal component k (every other direction
    held at z_mean), decode each step, and plot the wing-angle morph. Mirrors
    _plot_traversal_for_dim but moves along a PC direction instead of a raw dim.
    """
    component = torch.from_numpy(pca["components"][k]).to(device=device, dtype=z_mean.dtype)
    sigma_k   = float(pca["sigma_pc"][k])
    deltas    = torch.linspace(-range_std, range_std, n_steps).to(device)
    z_sweep   = z_mean.unsqueeze(0).repeat(n_steps, 1).clone()
    z_sweep  += (deltas * sigma_k).unsqueeze(1) * component.unsqueeze(0)
    with torch.no_grad():
        recon_sa = model.decode(z_sweep).cpu().numpy()                          # (n_steps, C, L)

    template_L   = repr_plot["template_L"]
    reconstruct  = repr_plot["reconstruct"]
    wings        = repr_plot["wings"]
    L            = template_L.shape[0]
    phase        = np.linspace(0.0, 1.0, L)
    template_deg = np.rad2deg(template_L)
    colors       = plt.get_cmap("coolwarm")(np.linspace(0.0, 1.0, n_steps))
    mid_idx      = n_steps // 2

    fig, axes = plt.subplots(3, len(wings), figsize=(5.75 * len(wings), 8.5),
                             sharex=True, squeeze=False)
    fig.suptitle(
        f"Latent PCA traversal — PC {k}   "
        f"(σ={sigma_k:.3f}, explained var={100 * pca['explained_ratio'][k]:.1f}%)   "
        f"±{range_std:.1f}σ, {n_steps} steps",
        fontsize=13,
    )
    for col_idx, (wing_title, wing_cols) in enumerate(wings):
        for row, angle_label in enumerate(_ANGLE_ROW_LABELS):
            ax       = axes[row, col_idx]
            use_col  = wing_cols[row]
            ax.plot(phase, template_deg[:, use_col],
                    color="0.5", linestyle="--", lw=1.2, alpha=0.7, label="Template")
            for step_idx in range(n_steps):
                wing_angles = reconstruct(recon_sa[step_idx], template_L)
                lw = 2.2 if step_idx == mid_idx else 1.3
                ax.plot(phase, np.rad2deg(wing_angles[:, use_col]),
                        color=colors[step_idx], lw=lw)
            ax.grid(True, alpha=0.4)
            if row == 0:
                ax.set_title(wing_title, fontsize=11)
            if col_idx == 0:
                ax.set_ylabel(angle_label, fontsize=10)
            if row == 2:
                ax.set_xlabel("Normalized phase")

    sm = plt.cm.ScalarMappable(
        cmap="coolwarm", norm=plt.Normalize(vmin=-range_std, vmax=range_std)
    )
    sm.set_array([])
    fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.04, label=f"PC[{k}] offset (σ)")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_pca(
    out_dir:    str,
    lat_prefix: str,
    model:      WingbeatAutoencoder,
    repr_plot:  dict,
    z_all:      torch.Tensor,
    z_mean:     torch.Tensor,
    latent_dim: int,
    n_steps:    int,
    range_std:  float,
    device:     str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    pca = _compute_pca(z_all, z_mean)
    n_pc = len(pca["sigma_pc"])

    scree_path = os.path.join(out_dir, f"{lat_prefix}pca_scree.png")
    _plot_pca_scree(pca, scree_path)
    print(f"  → wrote {scree_path}", flush=True)

    json_path = os.path.join(out_dir, f"{lat_prefix}pca.json")
    with open(json_path, "w") as f:
        json.dump({
            "n_wingbeats":                   pca["n_wingbeats"],
            "latent_dim":                    latent_dim,
            "z_mean":                        [float(v) for v in z_mean.cpu()],
            "components":                    [[float(v) for v in row] for row in pca["components"]],
            "sigma_pc":                      [float(v) for v in pca["sigma_pc"]],
            "explained_variance":            [float(v) for v in pca["explained_var"]],
            "explained_variance_ratio":      [float(v) for v in pca["explained_ratio"]],
            "cumulative_explained_variance": [float(v) for v in pca["cumulative"]],
        }, f, indent=2)
    print(f"  → wrote {json_path}", flush=True)

    for k in range(n_pc):
        out_path = os.path.join(out_dir, f"{lat_prefix}pc_{k:03d}.png")
        _plot_pca_traversal_for_pc(
            k=k, pca=pca, z_mean=z_mean, model=model, repr_plot=repr_plot,
            n_steps=n_steps, range_std=range_std, device=device, out_path=out_path,
        )
        print(f"  → wrote {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Analysis: pca_3d (3D scatter on top-3 PCs, colored by pitch-accel magnitude)
# ---------------------------------------------------------------------------

# Body-kinematics columns for angular acceleration. Body means are laid out
# [v(3), a(3), ω(3), α(3)]; the angular-accel block is cols 9-11 ordered
# (yaw, pitch, roll) — see transform_data._BODY_ALPHA_COLS and
# maneuver_scoring.CHANNEL_LABELS. One 3D scatter is drawn per axis.
_ANGULAR_ACCEL_AXES = [
    ("yaw",   9),
    ("pitch", 10),
    ("roll",  11),
]


def _plot_pca_scatter_3d_plotly(
    coords:         np.ndarray,    # (N, 3) projection onto top-3 PCs
    color_scalar:   np.ndarray,    # (N,) |angular accel| for the chosen axis
    axis_name:      str,           # "yaw" | "pitch" | "roll"
    expl_ratio:     np.ndarray,    # (>=3,) explained-variance ratio per PC
    color_clip_pct: float,
    n_wingbeats:    int,
    out_path:       str,
) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly not installed — skipping interactive 3D PCA scatter.", flush=True)
        return

    cmax = float(np.percentile(color_scalar, color_clip_pct))
    fig = go.Figure(
        data=go.Scatter3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
            mode="markers",
            marker=dict(
                size=3,
                color=color_scalar,
                colorscale="Viridis",
                cmin=0.0,
                cmax=cmax,
                opacity=0.6,
                showscale=True,
                colorbar=dict(title=f"‖{axis_name} angular accel‖"),
            ),
            hovertemplate=(
                "PC0=%{x:.3f}<br>PC1=%{y:.3f}<br>PC2=%{z:.3f}"
                f"<br>|{axis_name} α|=%{{marker.color:.3g}}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=(
            f"Latent PCA 3D scatter — {n_wingbeats} wingbeats, "
            f"colored by |{axis_name} angular accel| (color clipped at p{color_clip_pct:g})"
        ),
        scene=dict(
            xaxis_title=f"PC0 ({100 * expl_ratio[0]:.1f}% var)",
            yaxis_title=f"PC1 ({100 * expl_ratio[1]:.1f}% var)",
            zaxis_title=f"PC2 ({100 * expl_ratio[2]:.1f}% var)",
        ),
        width=1100, height=900,
        template="plotly_white",
    )
    fig.write_html(out_path, include_plotlyjs="cdn")


def run_pca_3d(
    out_dir:        str,
    lat_prefix:     str,
    z_all:          torch.Tensor,
    z_mean:         torch.Tensor,
    latent_dim:     int,
    npz_path:       str,
    color_clip_pct: float,
) -> None:
    if latent_dim < 3:
        print(f"  latent_dim={latent_dim} < 3 — cannot draw a 3D PCA scatter; skipping.", flush=True)
        return

    d = np.load(npz_path)
    if "body_means" not in d.files:
        print(f"  {npz_path} has no 'body_means' array — cannot color by angular accel; skipping.",
              flush=True)
        return
    body_means = d["body_means"]                                                # (N, 12)
    if body_means.shape[0] != z_all.shape[0]:
        print(f"  body_means rows ({body_means.shape[0]}) ≠ encoded wingbeats "
              f"({z_all.shape[0]}) — refusing to plot a misaligned scatter; skipping.", flush=True)
        return

    os.makedirs(out_dir, exist_ok=True)
    pca = _compute_pca(z_all, z_mean)
    z_all_np  = z_all.detach().cpu().numpy().astype(np.float64)
    z_mean_np = z_mean.detach().cpu().numpy().astype(np.float64)
    coords = (z_all_np - z_mean_np) @ pca["components"][:3].T                    # (N, 3)

    # One scatter per angular-accel axis (yaw, pitch, roll). The PCA projection
    # (coords) is shared; only the per-point color scalar changes.
    for axis_name, col in _ANGULAR_ACCEL_AXES:
        color_scalar = np.abs(body_means[:, col]).astype(np.float64)            # (N,)
        out_path = os.path.join(out_dir, f"{lat_prefix}pca_scatter_3d_{axis_name}.html")
        _plot_pca_scatter_3d_plotly(
            coords=coords, color_scalar=color_scalar, axis_name=axis_name,
            expl_ratio=pca["explained_ratio"], color_clip_pct=color_clip_pct,
            n_wingbeats=pca["n_wingbeats"], out_path=out_path,
        )
        print(f"  → wrote {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Analysis: template vs mean
# ---------------------------------------------------------------------------


def _summarize_comparison(
    name:      str,
    z_query:   torch.Tensor,
    z_ref:     torch.Tensor,
    z_std:     torch.Tensor,
    cov_inv:   np.ndarray,
    z_all:     torch.Tensor,
) -> dict:
    zscore = per_dim_zscore(z_query, z_ref, z_std)
    mahal  = full_mahalanobis(z_query, z_ref, cov_inv)
    pct, d_q, d_all = distance_percentile(z_query, z_ref, z_all)
    return {
        "name":                          name,
        "raw_l2_distance":               float(d_q),
        "per_dim_zscore":                [float(v) for v in zscore],
        "diagonal_mahalanobis":          float(np.linalg.norm(zscore)),
        "full_mahalanobis":              float(mahal),
        "percentile_rank_vs_centroid":   float(pct),
        "centroid_distance_distribution": {
            "n_samples": int(d_all.size),
            "mean":      float(d_all.mean()),
            "std":       float(d_all.std()),
            "median":    float(np.median(d_all)),
            "p05":       float(np.percentile(d_all,  5)),
            "p95":       float(np.percentile(d_all, 95)),
        },
    }


def _plot_zscores(comparisons: list[dict], latent_dim: int, out_path: str) -> None:
    n = len(comparisons)
    fig, axes = plt.subplots(n, 1, figsize=(max(7.0, latent_dim * 0.6), 2.6 * n + 0.6), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, comp in zip(axes, comparisons):
        zscores = np.asarray(comp["per_dim_zscore"], dtype=np.float64)
        ax.axhline(0.0, color="black",   lw=1.0, alpha=0.7)
        for thr in (-2, -1, 1, 2):
            ax.axhline(thr, color="0.6", linestyle="--", lw=0.8, alpha=0.5)
        colors = ["tab:blue" if v >= 0 else "tab:red" for v in zscores]
        ax.bar(range(latent_dim), zscores, color=colors)
        ax.set_xticks(range(latent_dim))
        ax.set_xticklabels([f"d{i}" for i in range(latent_dim)], fontsize=8)
        ax.set_ylabel("z-score [std]")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_title(
            f"{comp['name']}   "
            f"L2={comp['raw_l2_distance']:.3f}   "
            f"diag-Mahal={comp['diagonal_mahalanobis']:.3f}   "
            f"full-Mahal={comp['full_mahalanobis']:.3f}   "
            f"pct={comp['percentile_rank_vs_centroid']:.1f}%",
            fontsize=10,
        )
    axes[-1].set_xlabel("Latent dim")
    fig.suptitle(f"Per-dim z-score: query − z_mean (latent_dim={latent_dim})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_decoded_overlay(
    decoded:       dict,
    repr_plot:     dict,
    out_path:      str,
    latent_dim:    int,
) -> None:
    template_L = repr_plot["template_L"]
    wings      = repr_plot["wings"]
    L = template_L.shape[0]
    phase = np.linspace(0.0, 1.0, L)
    template_deg = np.rad2deg(template_L)

    line_specs = [
        ("z_zero",     "tab:red",    "dotted", 1.2, 0.7),
        ("z_mean",     "tab:orange", "dashed", 1.6, 0.95),
        ("z_template", "tab:blue",   "solid",  1.6, 0.95),
    ]

    fig, axes = plt.subplots(3, len(wings), figsize=(5.75 * len(wings), 8.5),
                             sharex=True, squeeze=False)
    fig.suptitle(
        f"Decoded reference latents vs the golden template  (latent_dim={latent_dim})",
        fontsize=13,
    )
    last_col = len(wings) - 1
    for col_idx, (wing_title, wing_cols) in enumerate(wings):
        for row, angle_label in enumerate(_ANGLE_ROW_LABELS):
            ax = axes[row, col_idx]
            use_col = wing_cols[row]
            ax.plot(phase, template_deg[:, use_col],
                    color="0.25", lw=2.4, label="Template (input)")
            for name, color, style, lw, alpha in line_specs:
                wing = decoded[name]
                ax.plot(phase, np.rad2deg(wing[:, use_col]),
                        color=color, linestyle=style, lw=lw, alpha=alpha,
                        label=f"decoder({name})")
            ax.grid(True, alpha=0.4)
            if row == 0:
                ax.set_title(wing_title, fontsize=11)
            if col_idx == 0:
                ax.set_ylabel(angle_label, fontsize=10)
            if row == 2:
                ax.set_xlabel("Normalized phase")
            if row == 0 and col_idx == last_col:
                ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_decoded_overlay_plotly(
    decoded:    dict,
    repr_plot:  dict,
    out_path:   str,
    latent_dim: int,
) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("plotly not installed — skipping interactive HTML decoded plot.", flush=True)
        return

    template_L   = repr_plot["template_L"]
    wings        = repr_plot["wings"]
    n_cols       = len(wings)
    L            = template_L.shape[0]
    phase        = np.linspace(0.0, 1.0, L)
    template_deg = np.rad2deg(template_L)

    template_spec = ("Template (input)", "rgba(64,64,64,1.0)", "solid", 2.6)
    line_specs = [
        ("decoder(z_zero)",     "rgba(214,39,40,0.85)",  "dot",   1.2),
        ("decoder(z_mean)",     "rgba(255,127,14,0.95)", "dash",  1.8),
        ("decoder(z_template)", "rgba(31,119,180,0.95)", "solid", 1.8),
    ]
    decoded_keys = ["z_zero", "z_mean", "z_template"]

    angle_short = ["Stroke φ", "Deviation θ", "Rotation ψ"]
    # Subplot titles laid out row-major: for each row (angle) iterate the wing columns.
    subplot_titles = [
        f"{wing_title} — {angle_short[row]}"
        for row in range(3) for wing_title, _ in wings
    ]
    fig = make_subplots(
        rows=3, cols=n_cols,
        shared_xaxes=True,
        subplot_titles=subplot_titles,
        vertical_spacing=0.06, horizontal_spacing=0.06,
    )
    y_titles = ["Stroke φ [deg]", "Deviation θ [deg]", "Rotation ψ [deg]"]

    for col_idx, (wing_title, wing_cols) in enumerate(wings):
        plotly_col = col_idx + 1
        for row, use_col in enumerate(wing_cols, start=1):
            t_name, t_color, t_dash, t_width = template_spec
            fig.add_trace(
                go.Scatter(
                    x=phase, y=template_deg[:, use_col],
                    mode="lines", name=t_name,
                    line=dict(color=t_color, width=t_width, dash=t_dash),
                    legendgroup=t_name,
                    showlegend=(row == 1 and col_idx == 0),
                ),
                row=row, col=plotly_col,
            )
            for (trace_name, color, dash, width), key in zip(line_specs, decoded_keys):
                wing = decoded[key]
                fig.add_trace(
                    go.Scatter(
                        x=phase, y=np.rad2deg(wing[:, use_col]),
                        mode="lines", name=trace_name,
                        line=dict(color=color, width=width, dash=dash),
                        legendgroup=trace_name,
                        showlegend=(row == 1 and col_idx == 0),
                    ),
                    row=row, col=plotly_col,
                )
            fig.update_yaxes(title_text=y_titles[row - 1], row=row, col=plotly_col)
    for plotly_col in range(1, n_cols + 1):
        fig.update_xaxes(title_text="Normalized phase", row=3, col=plotly_col)

    fig.update_layout(
        title=f"Decoded reference latents vs the golden template  (latent_dim={latent_dim})",
        height=900, width=max(700, 650 * n_cols),
        hovermode="x unified",
        template="plotly_white",
    )
    fig.write_html(out_path, include_plotlyjs="cdn")


def run_template_vs_mean(
    out_dir:    str,
    lat_prefix: str,
    model:      WingbeatAutoencoder,
    repr_plot:  dict,
    z_all:      torch.Tensor,
    z_mean:     torch.Tensor,
    z_std:      torch.Tensor,
    latent_dim: int,
    output_len: int,
    npz_path:   str,
    n_val:      int,
    model_dir:  str,
    device:     str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    template_L  = repr_plot["template_L"]
    reconstruct = repr_plot["reconstruct"]
    n_channels  = repr_plot["n_channels"]

    with torch.no_grad():
        zero_input = torch.zeros((1, n_channels, output_len), dtype=torch.float32, device=device)
        z_template = model.encode(zero_input).squeeze(0)
        z_zero     = torch.zeros(latent_dim, dtype=torch.float32, device=device)

    z_all_np = z_all.detach().cpu().numpy().astype(np.float64)
    cov = np.cov(z_all_np, rowvar=False)
    if cov.ndim == 0:
        cov = cov.reshape(1, 1)
    cov_inv = np.linalg.pinv(cov)

    comparisons = [
        _summarize_comparison(
            name="z_template vs z_mean",
            z_query=z_template, z_ref=z_mean, z_std=z_std, cov_inv=cov_inv, z_all=z_all,
        ),
        _summarize_comparison(
            name="z_zero vs z_mean",
            z_query=z_zero, z_ref=z_mean, z_std=z_std, cov_inv=cov_inv, z_all=z_all,
        ),
    ]

    print()
    for c in comparisons:
        print(
            f"  {c['name']:<32}  "
            f"L2={c['raw_l2_distance']:.4f}  "
            f"diag-Mahal={c['diagonal_mahalanobis']:.3f}  "
            f"full-Mahal={c['full_mahalanobis']:.3f}  "
            f"pct={c['percentile_rank_vs_centroid']:5.1f}%"
        )
    d_summary = comparisons[0]["centroid_distance_distribution"]
    print(
        f"\n  reference distribution ‖z_i − z_mean‖:  "
        f"median={d_summary['median']:.3f}  "
        f"p05={d_summary['p05']:.3f}  p95={d_summary['p95']:.3f}  "
        f"(n={d_summary['n_samples']})",
        flush=True,
    )

    decoded_wings: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for name, z_vec in [("z_template", z_template), ("z_mean", z_mean), ("z_zero", z_zero)]:
            recon_sa = model.decode(z_vec.unsqueeze(0)).cpu().numpy()
            decoded_wings[name] = reconstruct(recon_sa[0], template_L)

    json_payload = {
        "latent_dim":  latent_dim,
        "output_len":  output_len,
        "model_dir":   os.path.abspath(model_dir),
        "npz_path":    npz_path,
        "n_val_wingbeats": int(n_val),
        "reference_vectors": {
            "z_mean":     [float(v) for v in z_mean.cpu()],
            "z_template": [float(v) for v in z_template.cpu()],
            "z_zero":     [float(v) for v in z_zero.cpu()],
            "z_std":      [float(v) for v in z_std.cpu()],
        },
        "comparisons": comparisons,
    }
    json_path         = os.path.join(out_dir, f"{lat_prefix}template_vs_mean.json")
    zscore_path       = os.path.join(out_dir, f"{lat_prefix}template_vs_mean_zscore.png")
    decoded_png_path  = os.path.join(out_dir, f"{lat_prefix}template_vs_mean_decoded.png")
    decoded_html_path = os.path.join(out_dir, f"{lat_prefix}template_vs_mean_decoded.html")
    with open(json_path, "w") as f:
        json.dump(json_payload, f, indent=2)
    _plot_zscores(comparisons, latent_dim, zscore_path)
    _plot_decoded_overlay(decoded_wings, repr_plot, decoded_png_path, latent_dim)
    _save_decoded_overlay_plotly(decoded_wings, repr_plot, decoded_html_path, latent_dim)
    print(f"  → wrote {json_path}",         flush=True)
    print(f"  → wrote {zscore_path}",       flush=True)
    print(f"  → wrote {decoded_png_path}",  flush=True)
    print(f"  → wrote {decoded_html_path}", flush=True)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.strip(), formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model_dir", required=True,
                        help="Directory holding best_autoencoder.pt and best_config.json.")
    parser.add_argument("--analysis", nargs="+", default=["all"], choices=list(_ANALYSIS_CHOICES),
                        help="Which analyses to run. 'all' = every one (default).")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--out_dir", default=None,
                        help="Default: <model_dir>/latent_space_inspection/")
    parser.add_argument("--npz_path", default=None,
                        help="Override the fixed-L npz path. Default: derived from best_config.json.")
    # traversal-specific
    parser.add_argument("--n_steps", type=int, default=7,
                        help="traversal: number of sweep steps from -range_std to +range_std (default: 7).")
    parser.add_argument("--range_std", type=float, default=3.0,
                        help="traversal: half-width of the sweep in std-of-z units (default: 3.0).")
    # distribution_sampling-specific
    parser.add_argument("--n_samples", type=int, default=100,
                        help="distribution_sampling: how many empirical-z[:,k] samples per dim (default: 100).")
    parser.add_argument("--sampling_seed", type=int, default=0,
                        help="distribution_sampling: RNG seed for reproducible bootstrap (default: 0).")
    # pca_3d-specific
    parser.add_argument("--color_clip_pct", type=float, default=99.0,
                        help="pca_3d: clip the pitch-accel color scale at this percentile "
                             "so the heavy tail doesn't wash out the gradient (default: 99).")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    out_root = args.out_dir or os.path.join(args.model_dir, "latent_space_inspection")
    os.makedirs(out_root, exist_ok=True)

    requested = set(args.analysis)
    if "all" in requested:
        requested = {c for c in _ANALYSIS_CHOICES if c != "all"}

    ctx = _load_inputs(args.model_dir, device, args.npz_path)

    if "variance" in requested:
        print("\n[variance]", flush=True)
        run_variance(out_root, ctx["lat_prefix"], ctx["z_mean"], ctx["z_std"], ctx["latent_dim"])

    if "histograms" in requested:
        print("\n[histograms]", flush=True)
        run_histograms(out_root, ctx["lat_prefix"], ctx["z_all"], ctx["z_mean"],
                       ctx["z_std"], ctx["latent_dim"])

    if "template_vs_mean" in requested:
        print("\n[template_vs_mean]", flush=True)
        run_template_vs_mean(
            out_dir=os.path.join(out_root, "template_vs_mean"),
            lat_prefix=ctx["lat_prefix"],
            model=ctx["model"], repr_plot=ctx["repr_plot"],
            z_all=ctx["z_all"], z_mean=ctx["z_mean"], z_std=ctx["z_std"],
            latent_dim=ctx["latent_dim"], output_len=ctx["output_len"],
            npz_path=ctx["npz_path"], n_val=ctx["n_val"],
            model_dir=args.model_dir, device=device,
        )

    if "pca" in requested:
        print("\n[pca]", flush=True)
        run_pca(
            out_dir=os.path.join(out_root, "pca"),
            lat_prefix=ctx["lat_prefix"],
            model=ctx["model"], repr_plot=ctx["repr_plot"],
            z_all=ctx["z_all"], z_mean=ctx["z_mean"],
            latent_dim=ctx["latent_dim"],
            n_steps=args.n_steps, range_std=args.range_std, device=device,
        )

    if "pca_3d" in requested:
        print("\n[pca_3d]", flush=True)
        run_pca_3d(
            out_dir=os.path.join(out_root, "pca_3d"),
            lat_prefix=ctx["lat_prefix"],
            z_all=ctx["z_all"], z_mean=ctx["z_mean"],
            latent_dim=ctx["latent_dim"],
            npz_path=ctx["npz_path"], color_clip_pct=args.color_clip_pct,
        )

    if "traversal" in requested:
        print("\n[traversal]", flush=True)
        run_traversal(
            out_dir=os.path.join(out_root, "traversal"),
            lat_prefix=ctx["lat_prefix"],
            model=ctx["model"], repr_plot=ctx["repr_plot"],
            z_mean=ctx["z_mean"], z_std=ctx["z_std"],
            latent_dim=ctx["latent_dim"],
            n_steps=args.n_steps, range_std=args.range_std, device=device,
        )

    if "distribution_sampling" in requested:
        print("\n[distribution_sampling]", flush=True)
        run_distribution_sampling(
            out_dir=os.path.join(out_root, "distribution_sampling"),
            lat_prefix=ctx["lat_prefix"],
            model=ctx["model"], repr_plot=ctx["repr_plot"],
            z_all=ctx["z_all"], z_mean=ctx["z_mean"],
            latent_dim=ctx["latent_dim"],
            n_samples=args.n_samples, device=device, seed=args.sampling_seed,
        )

    print(f"\nDone. Outputs in: {out_root}", flush=True)


if __name__ == "__main__":
    main()
