"""
Latent-axis traversal — visualize what each latent dim controls.

Loads a trained autoencoder checkpoint, encodes the validation wingbeats to
get the natural per-dim mean and std, then for each latent entry k sweeps it
from -range_std to +range_std around the mean (all other entries held at the
mean) and decodes each step. One PNG per latent dim shows how the
reconstructed wing angles change as that one dim is swept.

Also emits a variance bar chart so you can see which dims carry most of the
val-set variation (large std = "active") vs which are unused (~0 std).

Run from the project root:
    python code/inspect_latent_dims.py --model_dir <run_dir>
    python code/inspect_latent_dims.py --model_dir <run_dir>/latent_dim_8
    python code/inspect_latent_dims.py --model_dir <run_dir> --n_steps 9 --range_std 4
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

# Make sibling imports work whether invoked as `python code/inspect_latent_dims.py`
# or with code/ on sys.path.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from autoencoder import WingbeatAutoencoder
from transform_data import _cubic_resample
from data_handling.bucket_eval import _reconstruct_wing_angles_from_normalized_sa


# Same per-angle layout used by the bucket-eval reconstruction plots.
_ANGLES = [
    ("Stroke φ (deg)",    0, 3),
    ("Deviation θ (deg)", 1, 4),
    ("Rotation ψ (deg)",  2, 5),
]


def _load_model(model_dir: str, device: str):
    ckpt_path = os.path.join(model_dir, "best_autoencoder.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No best_autoencoder.pt in {model_dir}.")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = WingbeatAutoencoder(
        latent_dim          = ckpt['latent_dim'],
        activation          = ckpt.get('activation', 'gelu'),
        dropout             = ckpt.get('dropout', 0.0),
        base_channels       = ckpt.get('base_channels', 128),
        bottleneck_len      = ckpt.get('bottleneck_len', 12),
        decoder_kernel_size = ckpt.get('decoder_kernel_size', 5),
        output_len          = ckpt['output_len'],
    )
    model.load_state_dict(ckpt['state_dict'])
    model.to(device).eval()
    return model, ckpt


def _save_variance_plotly(
    z_std:      torch.Tensor,
    order:      np.ndarray,
    latent_dim: int,
    html_path:  str,
) -> None:
    """Interactive HTML companion to _plot_variance_bar."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly not installed — skipping interactive HTML variance plot.", flush=True)
        return
    sorted_std = z_std[order].cpu().numpy()
    labels     = [f"d{int(i)}" for i in order]
    fig = go.Figure(go.Bar(x=labels, y=sorted_std, marker_color="rebeccapurple"))
    fig.update_layout(
        title=(
            f"Latent dim activity — larger std = used by the model, ~0 = unused  "
            f"(latent_dim={latent_dim})"
        ),
        xaxis_title="Latent dim (sorted by std)",
        yaxis_title="std(z) across val set",
        template="plotly_white",
        height=420, width=max(640, latent_dim * 60),
    )
    os.makedirs(os.path.dirname(os.path.abspath(html_path)), exist_ok=True)
    fig.write_html(html_path, include_plotlyjs="cdn")


def _save_traversal_plotly(
    k:           int,
    z_mean:      torch.Tensor,
    z_std:       torch.Tensor,
    model:       WingbeatAutoencoder,
    template_L:  np.ndarray,
    n_steps:     int,
    range_std:   float,
    device:      str,
    html_path:   str,
) -> None:
    """
    Interactive HTML companion to _plot_traversal_for_dim. Same 3 rows × 2 cols
    layout (angle × wing); each trace can be toggled from the legend so users
    can isolate one sweep step at a time.
    """
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.subplots import make_subplots
    except ImportError:
        print("plotly not installed — skipping interactive HTML traversal plot.", flush=True)
        return

    deltas = torch.linspace(-range_std, range_std, n_steps).to(device)
    z_sweep = z_mean.unsqueeze(0).repeat(n_steps, 1)
    z_sweep[:, k] = z_mean[k] + deltas * z_std[k]
    with torch.no_grad():
        recon_sa = model.decode(z_sweep).cpu().numpy()

    L            = template_L.shape[0]
    phase        = np.linspace(0.0, 1.0, L)
    template_deg = np.rad2deg(template_L)
    # Hex strings from a coolwarm-equivalent diverging palette, sampled at n_steps points.
    colors = px.colors.sample_colorscale("RdBu_r", list(np.linspace(0.0, 1.0, n_steps)))
    mid_idx = n_steps // 2

    fig = make_subplots(
        rows=3, cols=2,
        shared_xaxes=True,
        subplot_titles=("Left wing — Stroke φ",  "Right wing — Stroke φ",
                        "Left wing — Deviation θ", "Right wing — Deviation θ",
                        "Left wing — Rotation ψ",  "Right wing — Rotation ψ"),
        vertical_spacing=0.06, horizontal_spacing=0.06,
    )

    delta_vals = deltas.cpu().numpy()

    for col_wing in (0, 1):
        plotly_col = col_wing + 1
        for row, (_, L_col, R_col) in enumerate(_ANGLES, start=1):
            use_col = L_col if col_wing == 0 else R_col
            # Template overlay
            fig.add_trace(
                go.Scatter(
                    x=phase, y=template_deg[:, use_col],
                    mode="lines", name="Template",
                    line=dict(color="gray", width=1.4, dash="dash"),
                    legendgroup="Template",
                    showlegend=(row == 1 and col_wing == 0),
                ),
                row=row, col=plotly_col,
            )
            # n_steps traversal traces
            for step_idx in range(n_steps):
                wing_angles = _reconstruct_wing_angles_from_normalized_sa(
                    recon_sa[step_idx], template_L
                )
                width = 3.2 if step_idx == mid_idx else 1.8
                trace_name = f"δ={delta_vals[step_idx]:+.2f}σ"
                fig.add_trace(
                    go.Scatter(
                        x=phase, y=np.rad2deg(wing_angles[:, use_col]),
                        mode="lines", name=trace_name,
                        line=dict(color=colors[step_idx], width=width),
                        legendgroup=trace_name,
                        showlegend=(row == 1 and col_wing == 0),
                    ),
                    row=row, col=plotly_col,
                )
            fig.update_yaxes(title_text=["Stroke φ [deg]", "Deviation θ [deg]", "Rotation ψ [deg]"][row - 1],
                             row=row, col=plotly_col)
    fig.update_xaxes(title_text="Normalized phase", row=3, col=1)
    fig.update_xaxes(title_text="Normalized phase", row=3, col=2)

    fig.update_layout(
        title=(
            f"Latent traversal — dim {k}   (z_mean={float(z_mean[k]):+.3f}, "
            f"z_std={float(z_std[k]):.3f})   ±{range_std:.1f}σ, {n_steps} steps"
        ),
        height=900, width=1300,
        hovermode="x unified",
        template="plotly_white",
        legend=dict(title=f"z[{k}] offset"),
    )
    os.makedirs(os.path.dirname(os.path.abspath(html_path)), exist_ok=True)
    fig.write_html(html_path, include_plotlyjs="cdn")


def _plot_variance_bar(z_std: torch.Tensor, latent_dim: int, out_path: str) -> np.ndarray:
    """Bar chart of z_std per dim, sorted descending. Returns the sort order."""
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


def _plot_traversal_for_dim(
    k:           int,
    z_mean:      torch.Tensor,      # (latent_dim,) on device
    z_std:       torch.Tensor,      # (latent_dim,) on device
    model:       WingbeatAutoencoder,
    template_L:  np.ndarray,        # (L, 6) radians, golden template resampled to L
    n_steps:     int,
    range_std:   float,
    device:      str,
    out_path:    str,
) -> None:
    """
    Build the sweep z_sweep with shape (n_steps, latent_dim), decode, and plot.
    Layout: 3 rows (angle φ/θ/ψ) × 2 cols (left/right wing). Per cell: template
    overlay (gray dashed) + `n_steps` curves color-graded by sweep direction.
    The middle step (δ=0) is drawn thicker so the mean wingbeat is easy to spot.
    """
    deltas = torch.linspace(-range_std, range_std, n_steps).to(device)
    z_sweep = z_mean.unsqueeze(0).repeat(n_steps, 1)
    z_sweep[:, k] = z_mean[k] + deltas * z_std[k]
    with torch.no_grad():
        recon_sa = model.decode(z_sweep).cpu().numpy()                 # (n_steps, 6, L)

    L            = template_L.shape[0]
    phase        = np.linspace(0.0, 1.0, L)
    template_deg = np.rad2deg(template_L)
    colors       = plt.get_cmap("coolwarm")(np.linspace(0.0, 1.0, n_steps))
    mid_idx      = n_steps // 2  # step closest to δ=0

    fig, axes = plt.subplots(3, 2, figsize=(11.5, 8.5), sharex=True)
    fig.suptitle(
        f"Latent traversal — dim {k}   (z_mean={float(z_mean[k]):+.3f}, "
        f"z_std={float(z_std[k]):.3f})   ±{range_std:.1f}σ, {n_steps} steps",
        fontsize=13,
    )
    wing_labels = ["Left wing", "Right wing"]
    for col_wing in (0, 1):
        for row, (angle_label, L_col, R_col) in enumerate(_ANGLES):
            ax       = axes[row, col_wing]
            use_col  = L_col if col_wing == 0 else R_col
            ax.plot(phase, template_deg[:, use_col],
                    color="0.5", linestyle="--", lw=1.2, alpha=0.7, label="Template")
            for step_idx in range(n_steps):
                wing_angles = _reconstruct_wing_angles_from_normalized_sa(
                    recon_sa[step_idx], template_L)                    # (L, 6) radians
                lw = 2.2 if step_idx == mid_idx else 1.3
                ax.plot(phase, np.rad2deg(wing_angles[:, use_col]),
                        color=colors[step_idx], lw=lw)
            ax.grid(True, alpha=0.4)
            if row == 0:
                ax.set_title(wing_labels[col_wing], fontsize=11)
            if col_wing == 0:
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.strip(), formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model_dir", required=True,
                        help="Directory holding best_autoencoder.pt and best_config.json.")
    parser.add_argument("--n_steps",   type=int,   default=7,
                        help="Number of sweep steps from -range_std to +range_std (default: 7).")
    parser.add_argument("--range_std", type=float, default=3.0,
                        help="Half-width of the sweep in std-of-z units (default: 3.0).")
    parser.add_argument("--device",    default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--out_dir",   default=None,
                        help="Default: <model_dir>/latent_traversal/")
    parser.add_argument("--npz_path",  default=None,
                        help="Override the fixed-L npz path. Default: derived from best_config.json.")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = args.out_dir or os.path.join(args.model_dir, "latent_traversal")
    os.makedirs(out_dir, exist_ok=True)

    model, ckpt = _load_model(args.model_dir, device)
    latent_dim  = int(ckpt['latent_dim'])
    output_len  = int(ckpt['output_len'])

    config_path = os.path.join(args.model_dir, "best_config.json")
    with open(config_path) as f:
        config = json.load(f)
    if args.npz_path is not None:
        npz_path = args.npz_path
    else:
        data_dir = os.path.dirname(os.path.abspath(config['data_path']))
        npz_path = os.path.join(data_dir, f"wingbeats_L{output_len}.npz")
    template      = np.load(config['template_path'])
    template_L    = _cubic_resample(template, output_len).astype(np.float64)

    d            = np.load(npz_path)
    sa_all_np    = d['sa_wingbeats']                                # (N, 6, L)
    sa_all       = torch.from_numpy(sa_all_np).float().to(device)

    print(
        f"Loaded model: latent_dim={latent_dim}, output_len={output_len}; "
        f"encoding {sa_all.shape[0]} wingbeats to estimate the latent distribution.",
        flush=True,
    )

    with torch.no_grad():
        z_all = model.encode(sa_all)                                 # (N, latent_dim)
    z_mean = z_all.mean(dim=0)
    z_std  = z_all.std (dim=0)

    # All outputs carry a lat_<NN>_ prefix so files from different latent_dim
    # checkpoints can coexist (and be diffed) in the same directory if the user
    # copies them around for comparison.
    lat_prefix = f"lat_{latent_dim:02d}_"

    var_png  = os.path.join(out_dir, f"{lat_prefix}latent_variance.png")
    order    = _plot_variance_bar(z_std, latent_dim, var_png)
    print(f"  → wrote {var_png}", flush=True)
    var_html = os.path.join(out_dir, f"{lat_prefix}latent_variance.html")
    _save_variance_plotly(z_std, order, latent_dim, var_html)
    print(f"  → wrote {var_html}", flush=True)
    with open(os.path.join(out_dir, f"{lat_prefix}latent_variance.json"), "w") as f:
        json.dump({
            "z_mean":                  [float(v) for v in z_mean.cpu()],
            "z_std":                   [float(v) for v in z_std.cpu()],
            "sorted_dims_by_std_desc": [int(i)   for i in order],
        }, f, indent=2)

    for k in range(latent_dim):
        out_path = os.path.join(out_dir, f"{lat_prefix}dim_{k:03d}.png")
        _plot_traversal_for_dim(
            k          = k,
            z_mean     = z_mean,
            z_std      = z_std,
            model      = model,
            template_L = template_L,
            n_steps    = args.n_steps,
            range_std  = args.range_std,
            device     = device,
            out_path   = out_path,
        )
        html_path = os.path.join(out_dir, f"{lat_prefix}dim_{k:03d}.html")
        _save_traversal_plotly(
            k          = k,
            z_mean     = z_mean,
            z_std      = z_std,
            model      = model,
            template_L = template_L,
            n_steps    = args.n_steps,
            range_std  = args.range_std,
            device     = device,
            html_path  = html_path,
        )
        print(f"  → wrote {out_path}  +  {html_path}", flush=True)

    print(f"\nDone. {latent_dim} traversal PNGs + variance chart saved in: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
