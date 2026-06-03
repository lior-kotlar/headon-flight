"""
Latent-space placement of the "average wingbeat" — a probe of encoder nonlinearity.

Compares three reference latent vectors against the empirical distribution of
encoded validation wingbeats:

    z_mean             — mean of encoded val wingbeats   (latent-space centroid)
    z_template         — encoder(zeros), the encoding of an SA-normalized input
                         that matches the golden template exactly (SA(template,
                         template) = 0, so the input is the zero tensor)
    z_zero             — origin of latent space (reference baseline / "no info")

For each of {z_template, z_zero} versus z_mean, three comparison metrics
are computed:

    (A) per-dim z-score: (q - z_mean) / z_std  per latent entry
    (B) full Mahalanobis distance using the covariance of z over the val set
    (C) percentile rank of ‖q − z_mean‖ in the distribution of ‖z_i − z_mean‖
        across the val set — answers "is q closer to the centroid than a typical
        wingbeat is, or farther?"

Two figures are emitted:

    * <prefix>_zscore.png   — per-dim z-score bar charts (3 stacked subplots)
    * <prefix>_decoded.png  — overlay of the wing-angle reconstructions decoded
                              from each latent vector, plus the actual template

A JSON sidecar carries the numeric report so the analysis can be re-plotted or
diffed across models.

Standalone — NOT triggered by training or evaluate_autoencoder. Invoke manually:

    python code/inspect_latent_template_vs_mean.py --model_dir <run_dir>
    python code/inspect_latent_template_vs_mean.py --model_dir <run_dir>/latent_dim_8
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

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from autoencoder import WingbeatAutoencoder
from transform_data import _cubic_resample
from data_handling.bucket_eval import _reconstruct_wing_angles_from_normalized_sa


# Per-angle layout in the (L, 6) wing-angle space, matching bucket_eval._PLOT_ANGLES.
_ANGLES = [
    ("Stroke φ (deg)",    0, 3),
    ("Deviation θ (deg)", 1, 4),
    ("Rotation ψ (deg)",  2, 5),
]


# ---------------------------------------------------------------------------
# Comparison metrics — three independent functions, one per option.
# ---------------------------------------------------------------------------


def per_dim_zscore(
    z_query: torch.Tensor,    # (latent_dim,)
    z_ref:   torch.Tensor,    # (latent_dim,) reference centroid
    z_std:   torch.Tensor,    # (latent_dim,) per-dim std across the val set
) -> np.ndarray:
    """
    (A) Per-dim signed offset in std-units.

    Returns (q[k] - ref[k]) / std[k] as a length-latent_dim numpy array. The
    L2 norm of this vector is the *diagonal* Mahalanobis distance — what you'd
    use if you ignore correlations between latent dims. Useful as a per-dim
    diagnostic: a single entry far from 0 says "the gap concentrates in dim k."
    """
    safe_std = torch.clamp(z_std, min=1e-12)
    return ((z_query - z_ref) / safe_std).detach().cpu().numpy()


def full_mahalanobis(
    z_query:  torch.Tensor,    # (latent_dim,)
    z_ref:    torch.Tensor,    # (latent_dim,)
    cov_inv:  np.ndarray,      # (latent_dim, latent_dim) inverse covariance of z
) -> float:
    """
    (B) Mahalanobis distance with the full covariance.

    sqrt((q - ref)^T Σ^-1 (q - ref)) — the natural notion of distance "in
    natural units of the data distribution," accounting for both per-dim std
    AND correlations between dims. For uncorrelated dims this reduces to (A)'s
    L2 norm. Returns a single scalar.
    """
    diff = (z_query - z_ref).detach().cpu().numpy().astype(np.float64)
    return float(np.sqrt(diff @ cov_inv @ diff))


def distance_percentile(
    z_query:  torch.Tensor,    # (latent_dim,)
    z_ref:    torch.Tensor,    # (latent_dim,)
    z_all:    torch.Tensor,    # (N, latent_dim) — empirical population
) -> tuple[float, float, np.ndarray]:
    """
    (C) Percentile rank of ‖z_query − z_ref‖ in the distribution of
    ‖z_i − z_ref‖ over the full val set.

    Returns (percentile_rank, query_distance, distribution_distances). A
    percentile_rank of 5.0 means: 5% of real wingbeats lie closer to z_ref
    than z_query does — i.e. z_query is closer to the centroid than 95% of
    real wingbeats. A rank of 95.0 means z_query is unusually far from z_ref.
    """
    diff = (z_query - z_ref).detach().cpu().numpy()
    d_query = float(np.linalg.norm(diff))
    all_diffs = (z_all - z_ref).detach().cpu().numpy()
    d_all = np.linalg.norm(all_diffs, axis=1)
    pct = float(100.0 * (d_all < d_query).mean())
    return pct, d_query, d_all


def _summarize_comparison(
    name:      str,
    z_query:   torch.Tensor,
    z_ref:     torch.Tensor,
    z_std:     torch.Tensor,
    cov_inv:   np.ndarray,
    z_all:     torch.Tensor,
) -> dict:
    """Run all three metrics on (z_query vs z_ref) and bundle them in a JSON-friendly dict."""
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


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_zscores(
    comparisons:  list[dict],   # output of _summarize_comparison, one per query
    latent_dim:   int,
    out_path:     str,
) -> None:
    """One stacked subplot per comparison, signed z-score per latent dim."""
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
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_decoded_overlay_plotly(
    decoded:    dict,           # name → (L, 6) wing-angle ndarray in radians
    template_L: np.ndarray,
    out_path:   str,
    latent_dim: int,
) -> None:
    """
    Interactive HTML companion to _plot_decoded_overlay. Same 3 rows × 2 cols
    layout. Each line is a separate Plotly trace; toggling a legend entry
    hides/shows it across all six subplots via legendgroup.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("plotly not installed — skipping interactive HTML decoded plot.", flush=True)
        return

    L            = template_L.shape[0]
    phase        = np.linspace(0.0, 1.0, L)
    template_deg = np.rad2deg(template_L)

    # Line styles mirror the matplotlib version so the two outputs look related.
    template_spec = ("Template (input)", "rgba(64,64,64,1.0)",  "solid", 2.6)
    line_specs = [
        ("decoder(z_zero)",           "rgba(214,39,40,0.85)",   "dot",   1.2),
        ("decoder(z_mean)",           "rgba(255,127,14,0.95)",  "dash",  1.8),
        ("decoder(z_template)",       "rgba(31,119,180,0.95)",  "solid", 1.8),
    ]
    # Name → (decoded array key, ordering preserved). The legend ordering follows
    # this list, mirroring the matplotlib z-order.
    decoded_keys = ["z_zero", "z_mean", "z_template"]

    fig = make_subplots(
        rows=3, cols=2,
        shared_xaxes=True,
        subplot_titles=("Left wing — Stroke φ",     "Right wing — Stroke φ",
                        "Left wing — Deviation θ", "Right wing — Deviation θ",
                        "Left wing — Rotation ψ",  "Right wing — Rotation ψ"),
        vertical_spacing=0.06, horizontal_spacing=0.06,
    )
    y_titles = ["Stroke φ [deg]", "Deviation θ [deg]", "Rotation ψ [deg]"]

    for col_wing in (0, 1):
        plotly_col = col_wing + 1
        for row, (_, L_col, R_col) in enumerate(_ANGLES, start=1):
            use_col = L_col if col_wing == 0 else R_col
            t_name, t_color, t_dash, t_width = template_spec
            fig.add_trace(
                go.Scatter(
                    x=phase, y=template_deg[:, use_col],
                    mode="lines", name=t_name,
                    line=dict(color=t_color, width=t_width, dash=t_dash),
                    legendgroup=t_name,
                    showlegend=(row == 1 and col_wing == 0),
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
                        showlegend=(row == 1 and col_wing == 0),
                    ),
                    row=row, col=plotly_col,
                )
            fig.update_yaxes(title_text=y_titles[row - 1], row=row, col=plotly_col)
    fig.update_xaxes(title_text="Normalized phase", row=3, col=1)
    fig.update_xaxes(title_text="Normalized phase", row=3, col=2)

    fig.update_layout(
        title=f"Decoded reference latents vs the golden template  (latent_dim={latent_dim})",
        height=900, width=1300,
        hovermode="x unified",
        template="plotly_white",
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn")


def _plot_decoded_overlay(
    decoded:       dict,           # name → (L, 6) wing-angle ndarray in radians
    template_L:    np.ndarray,     # (L, 6) actual template in radians
    out_path:      str,
    latent_dim:    int,
) -> None:
    """3 rows × 2 cols (angle × wing). Five lines per cell: template + 4 decoded latents."""
    L = template_L.shape[0]
    phase = np.linspace(0.0, 1.0, L)
    template_deg = np.rad2deg(template_L)

    # Style table: order matters — drawn back-to-front, last on top.
    line_specs = [
        ("z_zero",           "tab:red",    "dotted",  1.2, 0.7),
        ("z_mean",           "tab:orange", "dashed",  1.6, 0.95),
        ("z_template",       "tab:blue",   "solid",   1.6, 0.95),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(11.5, 8.5), sharex=True)
    fig.suptitle(
        f"Decoded reference latents vs the golden template  (latent_dim={latent_dim})",
        fontsize=13,
    )
    wing_titles = ["Left wing", "Right wing"]

    for col_wing in (0, 1):
        for row, (angle_label, L_col, R_col) in enumerate(_ANGLES):
            ax = axes[row, col_wing]
            use_col = L_col if col_wing == 0 else R_col
            # Actual template — bold gray reference (the "ground truth").
            ax.plot(phase, template_deg[:, use_col],
                    color="0.25", lw=2.4, label="Template (input)")
            for name, color, style, lw, alpha in line_specs:
                wing = decoded[name]                                       # (L, 6) radians
                ax.plot(phase, np.rad2deg(wing[:, use_col]),
                        color=color, linestyle=style, lw=lw, alpha=alpha,
                        label=f"decoder({name})")
            ax.grid(True, alpha=0.4)
            if row == 0:
                ax.set_title(wing_titles[col_wing], fontsize=11)
            if col_wing == 0:
                ax.set_ylabel(angle_label, fontsize=10)
            if row == 2:
                ax.set_xlabel("Normalized phase")
            if row == 0 and col_wing == 1:
                ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _load_model(model_dir: str, device: str):
    ckpt_path = os.path.join(model_dir, "best_autoencoder.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No best_autoencoder.pt in {model_dir}.")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = WingbeatAutoencoder(
        latent_dim          = ckpt["latent_dim"],
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip(),
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_dir", required=True,
                        help="Directory holding best_autoencoder.pt and best_config.json.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--out_dir", default=None,
                        help="Default: <model_dir>/template_vs_mean/")
    parser.add_argument("--npz_path", default=None,
                        help="Override the fixed-L npz path. Default: derived from best_config.json.")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = args.out_dir or os.path.join(args.model_dir, "template_vs_mean")
    os.makedirs(out_dir, exist_ok=True)

    model, ckpt = _load_model(args.model_dir, device)
    latent_dim  = int(ckpt["latent_dim"])
    output_len  = int(ckpt["output_len"])
    lat_prefix  = f"lat_{latent_dim:02d}_"

    with open(os.path.join(args.model_dir, "best_config.json")) as f:
        config = json.load(f)
    if args.npz_path is not None:
        npz_path = args.npz_path
    else:
        data_dir = os.path.dirname(os.path.abspath(config["data_path"]))
        npz_path = os.path.join(data_dir, f"wingbeats_L{output_len}.npz")
    template_native = np.load(config["template_path"])
    template_L      = _cubic_resample(template_native, output_len).astype(np.float64)

    d         = np.load(npz_path)
    sa_all_np = d["sa_wingbeats"]                                          # (N, 6, L) float32
    sa_all    = torch.from_numpy(sa_all_np).float().to(device)
    n_val     = sa_all.shape[0]
    print(f"Loaded model: latent_dim={latent_dim}, output_len={output_len}; "
          f"encoding {n_val} wingbeats.", flush=True)

    # --- Compute the four reference latent vectors ---
    with torch.no_grad():
        z_all = model.encode(sa_all)                                       # (N, latent_dim)
        z_mean = z_all.mean(dim=0)
        z_std  = z_all.std (dim=0)

        # z_template: SA(template, template) = 0, so encode the all-zeros input.
        zero_input  = torch.zeros((1, 6, output_len), dtype=torch.float32, device=device)
        z_template  = model.encode(zero_input).squeeze(0)

        # z_zero: origin of latent space.
        z_zero = torch.zeros(latent_dim, dtype=torch.float32, device=device)

    # --- Covariance of z over the val set (for full Mahalanobis) ---
    z_all_np = z_all.detach().cpu().numpy().astype(np.float64)             # (N, latent_dim)
    # rowvar=False → variables are columns (latent dims), samples are rows.
    cov = np.cov(z_all_np, rowvar=False)
    if cov.ndim == 0:
        cov = cov.reshape(1, 1)
    # Pseudo-inverse for safety: if the encoder collapsed some dim to zero,
    # the covariance is singular and np.linalg.inv would crash.
    cov_inv = np.linalg.pinv(cov)

    # --- Run all three metrics for each query vector vs z_mean ---
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

    # --- Console summary ---
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

    # --- Decode each reference latent to wing-angle space for visual overlay ---
    decoded_wings: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for name, z_vec in [
            ("z_template", z_template),
            ("z_mean",     z_mean),
            ("z_zero",     z_zero),
        ]:
            recon_sa = model.decode(z_vec.unsqueeze(0)).cpu().numpy()      # (1, 6, L)
            decoded_wings[name] = _reconstruct_wing_angles_from_normalized_sa(
                recon_sa[0], template_L)                                   # (L, 6) radians

    # --- Save outputs ---
    json_payload = {
        "latent_dim":  latent_dim,
        "output_len":  output_len,
        "model_dir":   os.path.abspath(args.model_dir),
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
    json_path = os.path.join(out_dir, f"{lat_prefix}template_vs_mean.json")
    with open(json_path, "w") as f:
        json.dump(json_payload, f, indent=2)

    zscore_path       = os.path.join(out_dir, f"{lat_prefix}template_vs_mean_zscore.png")
    decoded_png_path  = os.path.join(out_dir, f"{lat_prefix}template_vs_mean_decoded.png")
    decoded_html_path = os.path.join(out_dir, f"{lat_prefix}template_vs_mean_decoded.html")
    _plot_zscores(comparisons, latent_dim, zscore_path)
    _plot_decoded_overlay(decoded_wings, template_L, decoded_png_path, latent_dim)
    _save_decoded_overlay_plotly(decoded_wings, template_L, decoded_html_path, latent_dim)

    print(f"\n  → wrote {json_path}",         flush=True)
    print(f"  → wrote {zscore_path}",       flush=True)
    print(f"  → wrote {decoded_png_path}",  flush=True)
    print(f"  → wrote {decoded_html_path}", flush=True)


if __name__ == "__main__":
    main()
