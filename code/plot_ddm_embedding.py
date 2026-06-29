"""
Visualise a directed-diffusion-map embedding produced by directed_diffusion_map.py.

The embedding itself is built from wing angles only (directed_diffusion_map.py no
longer folds body kinematics into the metric); the body angular accelerations
below are post-hoc colour overlays, never inputs. So colouring by them answers the
non-circular question "does the wing-only embedding organise by body response?".

Answers the question "what do the diffusion axes physically mean?" by:
  1. eigenvalue scree   — how many diffusion coordinates actually carry structure;
  2. interactive 3-D    — DC1/DC2/DC3 scatter of every wingbeat as a standalone Plotly
     HTML you can rotate. Coloured (with a colour-feature dropdown) only when --color
     is set, else a plain monochrome cloud;
  3. coloured scatters  — the leading 2-D plane (DC1 vs DC2), one panel per physical
     quantity (body yaw/pitch/roll accel), so you can see which physical variable
     each axis organises by;
  4. multi-pair view    — the single most embedding-aligned quantity shown across the
     three leading coordinate pairs (DC1-DC2, DC1-DC3, DC2-DC3);
  5. correlation map    — |corr(coordinate, physical quantity)| as a heatmap + a
     printed table, the numeric version of (3).

--color is off by default: only the scree and the (uncoloured) 3-D HTML are produced.
Pass --color to colour the 3-D plot and emit the colour-by-quantity views (3-5).
Outputs are PNG (matplotlib Agg) except the interactive 3-D Plotly HTML. The physical
quantities are read from the `color_*` arrays the DDM run stored in the .npz (see
default_color_features).

Run from project root:
    python code/plot_ddm_embedding.py            # uncoloured: scree + 3-D HTML
    python code/plot_ddm_embedding.py --color    # + colour-by-quantity views
    python code/plot_ddm_embedding.py --embedding data/analysis/ddm_embedding.npz --save_dir data/analysis/ddm
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_embedding(path: str) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Load coordinates, eigenvalues, and the {name: (N,)} colour-feature dict.

    Supports both the .npz (color_<name> arrays + color_feature_names) and the .pt
    (nested color_features dict) layouts written by directed_diffusion_map.save_result.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        z = np.load(path, allow_pickle=True)
        coords = z["coordinates"]
        eigvals = z["eigenvalues"]
        names = [str(n) for n in z["color_feature_names"]] if "color_feature_names" in z.files else []
        features = {n: z[f"color_{n}"] for n in names if f"color_{n}" in z.files}
    elif ext == ".pt":
        import torch
        d = torch.load(path, map_location="cpu", weights_only=False)
        coords = np.asarray(d["coordinates"])
        eigvals = np.asarray(d["eigenvalues"])
        features = {k: np.asarray(v) for k, v in d.get("color_features", {}).items()}
    else:
        raise ValueError(f"Unsupported embedding extension {ext!r}; use .npz or .pt")
    return np.asarray(coords), np.asarray(eigvals), features


def _subsample(n: int, max_points: int, seed: int = 0) -> np.ndarray:
    """Indices to plot: all of them if n <= max_points, else a random subset (scatter
    plots of 50k+ points are slow and over-plotted)."""
    if n <= max_points:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_points, replace=False))


def plot_scree(eigvals: np.ndarray, save_path: str) -> None:
    """Eigenvalue magnitude vs coordinate index — where the spectrum flattens is
    roughly how many diffusion coordinates carry real structure."""
    fig, ax = plt.subplots(figsize=(7, 4))
    idx = np.arange(1, len(eigvals) + 1)
    ax.plot(idx, eigvals, "o-", lw=1.5)
    ax.set_xlabel("diffusion coordinate (non-trivial, 1 = leading)")
    ax.set_ylabel(r"eigenvalue $\lambda_i$")
    ax.set_title("Diffusion-map spectrum (scree)")
    ax.grid(True, alpha=0.4)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  scree → {save_path}", flush=True)


def plot_colored_scatter_grid(
    coords: np.ndarray,
    features: dict[str, np.ndarray],
    save_path: str,
    dc: tuple[int, int] = (0, 1),
    max_points: int = 20000,
) -> None:
    """Leading plane (coords[:,dc[0]] vs coords[:,dc[1]]), one panel per feature,
    each point coloured by that feature's value."""
    if not features:
        print("  (no colour features stored — skipping coloured scatter grid)", flush=True)
        return
    names = list(features)
    ncol = min(3, len(names))
    nrow = int(np.ceil(len(names) / ncol))
    sel = _subsample(coords.shape[0], max_points)
    x, y = coords[sel, dc[0]], coords[sel, dc[1]]

    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 4.2 * nrow), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for i, name in enumerate(names):
        ax = axes.flat[i]
        ax.set_visible(True)
        c = features[name][sel]
        # Robust colour range (2–98th pct) so a few outliers don't wash out the scale.
        lo, hi = np.percentile(c, [2, 98])
        sc = ax.scatter(x, y, c=c, s=4, alpha=0.6, cmap="Spectral_r",
                        vmin=lo, vmax=hi, linewidths=0, rasterized=True)
        ax.set_title(name)
        ax.set_xlabel(f"DC{dc[0] + 1}")
        ax.set_ylabel(f"DC{dc[1] + 1}")
        ax.margins(0.05)               # pad data limits so edge points aren't clipped on the spine
        fig.colorbar(sc, ax=ax, shrink=0.85)
    fig.suptitle(f"Diffusion embedding — DC{dc[0] + 1} vs DC{dc[1] + 1}, coloured by physical quantity")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  coloured scatter grid → {save_path}", flush=True)


def plot_leading_pairs(
    coords: np.ndarray,
    feature: np.ndarray,
    feature_name: str,
    save_path: str,
    max_points: int = 20000,
) -> None:
    """The three leading coordinate pairs, all coloured by one quantity — shows the
    embedding's 3-D shape and how that quantity threads through it."""
    if coords.shape[1] < 3:
        return
    sel = _subsample(coords.shape[0], max_points)
    c = feature[sel]
    lo, hi = np.percentile(c, [2, 98])
    pairs = [(0, 1), (0, 2), (1, 2)]
    # constrained_layout auto-spaces the three panels, their axis labels, the suptitle
    # and the shared colorbar so none of them overlap.
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for ax, (a, b) in zip(axes, pairs):
        sc = ax.scatter(coords[sel, a], coords[sel, b], c=c, s=4, alpha=0.6,
                        cmap="Spectral_r", vmin=lo, vmax=hi, linewidths=0, rasterized=True)
        ax.set_xlabel(f"DC{a + 1}")
        ax.set_ylabel(f"DC{b + 1}")
        ax.margins(0.05)               # pad data limits so edge points aren't clipped on the spine
    fig.colorbar(sc, ax=axes, shrink=0.85, label=feature_name)
    fig.suptitle(f"Leading coordinate pairs, coloured by {feature_name}")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  leading-pairs view → {save_path}", flush=True)


def plot_3d_dcs_plotly(
    coords: np.ndarray,
    features: dict[str, np.ndarray],
    save_path: str,
    colored: bool = False,
    max_points: int = 20000,
) -> None:
    """Interactive 3-D scatter of the leading three diffusion coordinates (DC1, DC2,
    DC3), one point per wingbeat, written as a standalone Plotly HTML.

    colored=False (default): a single monochrome cloud — just the manifold shape.
    colored=True: points coloured by physical quantity, with a dropdown to switch
    which colour feature (body yaw/pitch/roll accel) drives the colour scale.
    """
    import plotly.graph_objects as go

    if coords.shape[1] < 3:
        print("  (need >=3 diffusion coordinates for the 3-D plot — skipping)", flush=True)
        return

    sel = _subsample(coords.shape[0], max_points)
    x, y, z = coords[sel, 0], coords[sel, 1], coords[sel, 2]
    layout_kwargs: dict = {}

    if not colored or not features:
        if colored and not features:
            print("  (--color set but no colour features stored — drawing uncoloured 3-D plot)", flush=True)
        fig = go.Figure(go.Scatter3d(
            x=x, y=y, z=z, mode="markers",
            marker=dict(size=2, color="#1f77b4", opacity=0.6), hoverinfo="skip",
        ))
    else:
        # One trace per colour feature; a dropdown toggles which one is visible.
        names = list(features)
        traces = []
        for i, name in enumerate(names):
            c = features[name][sel]
            lo, hi = np.percentile(c, [2, 98])
            traces.append(go.Scatter3d(
                x=x, y=y, z=z, mode="markers", name=name, visible=(i == 0),
                marker=dict(size=2, color=c, cmin=lo, cmax=hi, colorscale="Spectral_r",
                            opacity=0.6, colorbar=dict(title=name)),
                hoverinfo="skip",
            ))
        buttons = [
            dict(label=name, method="update",
                 args=[{"visible": [j == i for j in range(len(names))]}])
            for i, name in enumerate(names)
        ]
        fig = go.Figure(traces)
        layout_kwargs["updatemenus"] = [dict(
            buttons=buttons, direction="down", showactive=True,
            x=0.0, xanchor="left", y=1.08, yanchor="top",
        )]

    fig.update_layout(
        title="Diffusion embedding — DC1 / DC2 / DC3",
        scene=dict(xaxis_title="DC1", yaxis_title="DC2", zaxis_title="DC3"),
        width=950, height=820, margin=dict(l=0, r=0, t=50, b=0),
        **layout_kwargs,
    )
    fig.write_html(save_path, include_plotlyjs="cdn")
    print(f"  3-D DC1/DC2/DC3 scatter → {save_path}", flush=True)


def coordinate_feature_correlation(
    coords: np.ndarray,
    features: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[str]]:
    """|Pearson corr| between every diffusion coordinate and every physical quantity.
    Returns (corr (n_coords, n_features), feature_names)."""
    names = list(features)
    n_coords = coords.shape[1]
    corr = np.zeros((n_coords, len(names)), dtype=np.float64)
    for j, name in enumerate(names):
        f = features[name].astype(np.float64)
        f = f - f.mean()
        fstd = f.std()
        for i in range(n_coords):
            c = coords[:, i].astype(np.float64)
            c = c - c.mean()
            denom = c.std() * fstd
            corr[i, j] = 0.0 if denom < 1e-12 else float(np.mean(c * f) / denom)
    return corr, names


def plot_correlation_heatmap(corr: np.ndarray, names: list[str], save_path: str) -> None:
    """Heatmap of |corr(coordinate, quantity)| — the numeric 'what does each axis mean'."""
    if not names:
        return
    absc = np.abs(corr)
    n_coords = absc.shape[0]
    fig, ax = plt.subplots(figsize=(1.2 * len(names) + 3, 0.4 * n_coords + 2))
    im = ax.imshow(absc, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(n_coords))
    ax.set_yticklabels([f"DC{i + 1}" for i in range(n_coords)])
    for i in range(n_coords):
        for j in range(len(names)):
            ax.text(j, i, f"{absc[i, j]:.2f}", ha="center", va="center",
                    color="white" if absc[i, j] < 0.6 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.85, label="|correlation|")
    ax.set_title("Diffusion coordinate ↔ physical quantity |correlation|")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  correlation heatmap → {save_path}", flush=True)


def print_correlation_table(corr: np.ndarray, names: list[str]) -> None:
    """Console version of the heatmap: for each coordinate, its best-aligned quantity."""
    if not names:
        return
    print("\nDiffusion coordinate ↔ physical quantity (signed corr; * = strongest |corr| in row):")
    header = "  " + " " * 6 + "".join(f"{n[:12]:>14s}" for n in names)
    print(header)
    for i in range(corr.shape[0]):
        best = int(np.argmax(np.abs(corr[i])))
        cells = "".join(
            (f"{corr[i, j]:>13.2f}" + ("*" if j == best else " ")) for j in range(len(names))
        )
        print(f"  DC{i + 1:<4d}{cells}")
    print()


def visualize(embedding_path: str, save_dir: str, colored: bool = False, max_points: int = 20000) -> None:
    coords, eigvals, features = load_embedding(embedding_path)
    os.makedirs(save_dir, exist_ok=True)
    print(f"[viz] embedding: {coords.shape[0]} points × {coords.shape[1]} coords | "
          f"colour={'on' if colored else 'off'} | "
          f"{len(features)} colour features: {list(features)}", flush=True)

    plot_scree(eigvals, os.path.join(save_dir, "scree.png"))
    plot_3d_dcs_plotly(coords, features, os.path.join(save_dir, "ddm_3d.html"),
                       colored=colored, max_points=max_points)

    if not colored:
        print("  (--color not set: skipping colour-by-quantity plots; pass --color to enable)", flush=True)
    elif features:
        plot_colored_scatter_grid(coords, features, os.path.join(save_dir, "scatter_DC1_DC2.png"), dc=(0, 1))
        corr, names = coordinate_feature_correlation(coords, features)
        plot_correlation_heatmap(corr, names, os.path.join(save_dir, "coord_feature_corr.png"))
        print_correlation_table(corr, names)
        # Colour the leading-pairs view by whichever quantity aligns best with the
        # leading plane (max |corr| over DC1/DC2).
        lead = np.abs(corr[:2]).max(axis=0)
        top = names[int(np.argmax(lead))]
        plot_leading_pairs(coords, features[top], top,
                           os.path.join(save_dir, "leading_pairs.png"))
    else:
        print("  (--color set but no colour features stored — only scree + 3-D plot produced)", flush=True)

    print(f"[viz] done → {save_dir}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Visualise a DDM embedding (PNG).")
    p.add_argument("--embedding", default="data/analysis/ddm_embedding.npz",
                   help="path to the .npz/.pt written by directed_diffusion_map.py")
    p.add_argument("--save_dir", default=None,
                   help="output dir for PNGs (default: <embedding_dir>/ddm_viz)")
    p.add_argument("--max_points", type=int, default=20000,
                   help="max points drawn per scatter (random subsample above this)")
    p.add_argument("--color", dest="colored", action="store_true",
                   help="colour the plots by physical quantity (body yaw/pitch/roll accel) and emit the "
                        "colour-by-quantity views; default is uncoloured (plain manifold shape only)")
    p.set_defaults(colored=False)
    args = p.parse_args()

    save_dir = args.save_dir or os.path.join(os.path.dirname(os.path.abspath(args.embedding)), "ddm_viz")
    visualize(args.embedding, save_dir, colored=args.colored, max_points=args.max_points)


if __name__ == "__main__":
    main()
