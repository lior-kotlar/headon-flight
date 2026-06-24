"""
wingbeat_angle_space.py — 3D wing-angle-space visualization (model-independent).

A wingbeat for one wing is a sequence of three angles — stroke φ, deviation θ,
rotation ψ — sampled along the beat. Plotted with one angle per axis, a single
wingbeat is a closed *loop* through (φ, θ, ψ) space (it is periodic, so it
returns near its start), and a whole set of wingbeats fills a *point cloud* —
the region of angle space the wing actually visits. This is a state-space /
phase-portrait view of wing kinematics: the time/phase axis is dropped and the
trajectory is shown directly in angle space.

This module knows nothing about the autoencoder. It only turns wing-angle arrays
into interactive Plotly HTML. Two kinds of consumers use it:

  * transform_data.py — saves an angle-space view of the golden template next to
    the saved template .npy (the template is itself one representative wingbeat).
  * inspect_latent_space.py — draws the val-set point cloud and overlays an
    original wingbeat against its encode → latent/PC traversal → decode loops.

Core data unit
--------------
A "wingbeat" here is an (L, 3) array of [φ, θ, ψ] for ONE wing, in radians or
degrees (the `units` argument says which; everything is plotted in degrees).
A "trajectory" is a raw (T, 6) array [Lφ, Lθ, Lψ, Rφ, Rθ, Rψ]; helpers below
segment it into per-wing wingbeats.

Public API
----------
  make_angle_space_figure(cloud=..., loops=[...])  -> plotly Figure (the builder)
  plot_wingbeats_cloud(wingbeats, out_path)        -> point cloud HTML
  plot_single_wingbeat(wingbeat, out_path)         -> one phase-colored loop HTML
  plot_trajectory_cloud(traj, out_path)            -> segment a (T,6) flight, cloud
  wingbeats_from_trajectory(traj, wing=...)        -> list of (n, 3) wingbeats
  write_html(fig, out_path)                        -> save a Figure to HTML

CLI (quick look)
----------------
  # Golden template (one wing or both wings overlaid):
  python code/wingbeat_angle_space.py --template data/autoencoder_dataset/golden_template.npy

  # One flight trajectory as a point cloud (index into trajectories.npy):
  python code/wingbeat_angle_space.py --trajectories data/autoencoder_dataset/trajectories.npy --traj_index 0

  # The whole dataset of trajectories as one cloud:
  python code/wingbeat_angle_space.py --trajectories data/autoencoder_dataset/trajectories.npy --all
"""

import os

import numpy as np


# One axis per wing angle. Order matches the channel order [phi, theta, psi].
ANGLE_AXIS_TITLES = ("Stroke φ [deg]", "Deviation θ [deg]", "Rotation ψ [deg]")
# Short symbols for the same three angles (subplot titles / 2D hover labels).
ANGLE_SHORT = ("φ", "θ", "ψ")

# Left/right colors mirror the convention used by the template PNGs in transform_data.
WING_COLORS = {"left": "royalblue", "right": "crimson"}


def _import_plotly():
    try:
        import plotly.graph_objects as go
    except ImportError as e:  # surfaced to the caller; transform_data wraps this in try/except
        raise ImportError(
            "plotly is required for wingbeat angle-space plots (pip install plotly)."
        ) from e
    return go


def _to_deg(arr: np.ndarray, units: str) -> np.ndarray:
    """Return an (L, 3) angle array in degrees. `units` is 'rad' or 'deg'."""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"expected an (L, 3) wing-angle array, got shape {arr.shape}.")
    if units == "rad":
        return np.rad2deg(arr)
    if units == "deg":
        return arr
    raise ValueError(f"units must be 'rad' or 'deg', got {units!r}.")


def _concat_points_phase(wingbeats, units: str):
    """
    Stack a list of (L_i, 3) wingbeats into a single (M, 3) point array (degrees)
    plus an (M,) normalized-phase array in [0, 1] per wingbeat — so the cloud can
    be colored by where in the beat each point falls.
    """
    pts, phase = [], []
    for wb in wingbeats:
        wb_deg = _to_deg(wb, units)
        L = wb_deg.shape[0]
        pts.append(wb_deg)
        phase.append(np.linspace(0.0, 1.0, L))
    if not pts:
        return np.empty((0, 3)), np.empty((0,))
    return np.concatenate(pts, axis=0), np.concatenate(phase, axis=0)


def _loop_trace(go, spec: dict, default_units: str):
    """
    Build one Scatter3d loop from a spec dict:
      name           legend label
      angles         (L, 3) wing angles
      units          'rad' | 'deg'  (default: builder's units)
      close          append the first point to close the loop (default True)
      color          solid line/marker color (used when color_by_phase is False)
      color_by_phase color the loop along the beat with `colorscale` (default False)
      colorscale     Plotly colorscale name (default 'Viridis')
      markers        draw markers as well as the line (default False)
      width          line width (default 4)
      marker_size    marker size (default 3)
      show_colorbar  show the phase colorbar for this loop (default False)
    """
    angles = _to_deg(spec["angles"], spec.get("units", default_units))
    if spec.get("close", True):
        angles = np.vstack([angles, angles[:1]])
    n = angles.shape[0]
    phase = np.linspace(0.0, 1.0, n)

    width   = spec.get("width", 4)
    markers = spec.get("markers", False)
    mode    = "lines+markers" if markers else "lines"

    if spec.get("color_by_phase", False):
        colorscale = spec.get("colorscale", "Viridis")
        line   = dict(color=phase, colorscale=colorscale, width=width)
        marker = dict(
            size=spec.get("marker_size", 3), color=phase, colorscale=colorscale,
            cmin=0.0, cmax=1.0,
            showscale=spec.get("show_colorbar", False),
            colorbar=dict(title="beat phase", x=1.0, xanchor="left", len=0.75, y=0.5),
        )
    else:
        color  = spec.get("color", "crimson")
        line   = dict(color=color, width=width)
        marker = dict(size=spec.get("marker_size", 3), color=color)

    kwargs = dict(
        x=angles[:, 0], y=angles[:, 1], z=angles[:, 2],
        mode=mode, name=spec.get("name", "wingbeat"), line=line,
        hovertemplate=(
            "φ=%{x:.1f}<br>θ=%{y:.1f}<br>ψ=%{z:.1f}"
            "<br>phase=%{text:.2f}<extra>" + spec.get("name", "wingbeat") + "</extra>"
        ),
        text=phase,
    )
    if markers:
        kwargs["marker"] = marker
    return go.Scatter3d(**kwargs)


def make_angle_space_figure(
    cloud=None,
    loops=None,
    *,
    units: str = "deg",
    title: str = "Wingbeat angle space (one wing)",
    cloud_color_by_phase: bool = True,
    cloud_show_colorbar: bool = True,
    cloud_max_points: int = 20000,
    cloud_opacity: float = 0.25,
    cloud_marker_size: int = 2,
    aspectmode: str = "data",
    width: int = 1000,
    height: int = 850,
    seed: int = 0,
):
    """
    Assemble a 3D angle-space figure: an optional background point `cloud` (a list
    of (L_i, 3) wingbeats) plus any number of foreground `loops` (a list of spec
    dicts, see _loop_trace). Returns a plotly Figure; the caller saves it.

    The cloud is subsampled to `cloud_max_points` for render speed. `aspectmode`
    defaults to 'data' (equal scale on all axes, so loop geometry is undistorted);
    note the deviation θ range is much smaller than φ/ψ, so the manifold renders as
    a thin slab. Pass aspectmode='cube' to fill the box instead.
    """
    go = _import_plotly()
    fig = go.Figure()

    if cloud is not None and len(cloud) > 0:
        pts, phase = _concat_points_phase(cloud, units)
        if pts.shape[0] > cloud_max_points:
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.choice(pts.shape[0], size=cloud_max_points, replace=False))
            pts, phase = pts[idx], phase[idx]
        marker = dict(size=cloud_marker_size, opacity=cloud_opacity)
        if cloud_color_by_phase:
            marker.update(
                color=phase, colorscale="Viridis", cmin=0.0, cmax=1.0,
                showscale=cloud_show_colorbar,
                colorbar=dict(title="beat phase", x=1.0, xanchor="left", len=0.75, y=0.5),
            )
        else:
            marker.update(color="lightgray")
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers", marker=marker, name=f"wingbeat cloud (n={pts.shape[0]})",
            hovertemplate="φ=%{x:.1f}<br>θ=%{y:.1f}<br>ψ=%{z:.1f}<extra>cloud</extra>",
        ))

    for spec in (loops or []):
        fig.add_trace(_loop_trace(go, spec, units))
        if spec.get("mark_start", False):
            a0 = _to_deg(spec["angles"], spec.get("units", units))[0]
            fig.add_trace(go.Scatter3d(
                x=[a0[0]], y=[a0[1]], z=[a0[2]], mode="markers",
                marker=dict(size=6, color=spec.get("color", "black"), symbol="diamond"),
                name=f"{spec.get('name', 'wingbeat')} start", showlegend=False,
            ))

    fig.update_layout(
        title=title, width=width, height=height, template="plotly_white",
        scene=dict(
            xaxis_title=ANGLE_AXIS_TITLES[0],
            yaxis_title=ANGLE_AXIS_TITLES[1],
            zaxis_title=ANGLE_AXIS_TITLES[2],
            aspectmode=aspectmode,
        ),
        # The phase colorbar lives at the right edge (x≈1.02 by default), so keep
        # the legend at the top-left — otherwise the two stack and obscure each other.
        legend=dict(
            x=0.0, xanchor="left", y=1.0, yanchor="top",
            bgcolor="rgba(255,255,255,0.6)", bordercolor="rgba(0,0,0,0.2)", borderwidth=1,
        ),
    )
    return fig


def write_html(fig, out_path: str) -> str:
    """Save a plotly Figure to HTML (plotly.js via CDN), creating parent dirs."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn")
    return out_path


# ---------------------------------------------------------------------------
# 2D projections (same cloud + loops as the 3D view, two angle-pair panels)
# ---------------------------------------------------------------------------


def _add_cloud_2d(go, fig, row, col, pts, phase, xi, yi, *,
                  color_by_phase, show_colorbar, opacity, size, showlegend):
    """Scatter the (subsampled) cloud points projected onto axes (xi, yi)."""
    marker = dict(size=size, opacity=opacity)
    if color_by_phase:
        marker.update(
            color=phase, colorscale="Viridis", cmin=0.0, cmax=1.0,
            showscale=show_colorbar,
            colorbar=dict(title="beat phase", x=1.0, xanchor="left", len=0.75, y=0.5),
        )
    else:
        marker.update(color="lightgray")
    fig.add_trace(
        go.Scatter(
            x=pts[:, xi], y=pts[:, yi], mode="markers", marker=marker,
            name="wingbeat cloud", legendgroup="wingbeat cloud", showlegend=showlegend,
            hovertemplate=(
                f"{ANGLE_SHORT[xi]}=%{{x:.1f}}<br>{ANGLE_SHORT[yi]}=%{{y:.1f}}<extra>cloud</extra>"
            ),
        ),
        row=row, col=col,
    )


def _add_loop_2d(go, fig, row, col, spec, default_units, xi, yi, *, showlegend):
    """Draw one loop (and optional start marker) projected onto axes (xi, yi)."""
    angles = _to_deg(spec["angles"], spec.get("units", default_units))
    if spec.get("close", True):
        angles = np.vstack([angles, angles[:1]])
    n = angles.shape[0]
    phase = np.linspace(0.0, 1.0, n)

    width   = spec.get("width", 4)
    markers = spec.get("markers", False)
    mode    = "lines+markers" if markers else "lines"
    name    = spec.get("name", "wingbeat")

    if spec.get("color_by_phase", False):
        # 2D lines can't carry a per-point gradient; gradient goes on the markers,
        # the line stays a neutral gray.
        line   = dict(color="rgba(120,120,120,0.6)", width=width)
        marker = dict(
            size=spec.get("marker_size", 4), color=phase,
            colorscale=spec.get("colorscale", "Viridis"), cmin=0.0, cmax=1.0,
            showscale=spec.get("show_colorbar", False) and showlegend,
            colorbar=dict(title="beat phase", x=1.0, xanchor="left", len=0.75, y=0.5),
        )
    else:
        color  = spec.get("color", "crimson")
        line   = dict(color=color, width=width)
        marker = dict(size=spec.get("marker_size", 4), color=color)

    kwargs = dict(
        x=angles[:, xi], y=angles[:, yi], mode=mode, name=name,
        line=line, legendgroup=name, showlegend=showlegend,
        hovertemplate=(
            f"{ANGLE_SHORT[xi]}=%{{x:.1f}}<br>{ANGLE_SHORT[yi]}=%{{y:.1f}}<extra>{name}</extra>"
        ),
    )
    if markers:
        kwargs["marker"] = marker
    fig.add_trace(go.Scatter(**kwargs), row=row, col=col)

    if spec.get("mark_start", False):
        a0 = _to_deg(spec["angles"], spec.get("units", default_units))[0]
        fig.add_trace(
            go.Scatter(
                x=[a0[xi]], y=[a0[yi]], mode="markers",
                marker=dict(size=9, color=spec.get("color", "black"), symbol="diamond"),
                name=f"{name} start", legendgroup=name, showlegend=False,
            ),
            row=row, col=col,
        )


def make_angle_space_2d_figure(
    cloud=None,
    loops=None,
    *,
    units: str = "deg",
    title: str = "Wingbeat angle space — 2D projections",
    panels=((0, 2), (0, 1)),
    cloud_color_by_phase: bool = True,
    cloud_show_colorbar: bool = True,
    cloud_max_points: int = 8000,
    cloud_opacity: float = 0.15,
    cloud_marker_size: int = 3,
    width: int = 1350,
    height: int = 650,
    seed: int = 0,
):
    """
    Two (or more) 2D angle-pair panels in one figure, carrying the *same* data as
    make_angle_space_figure — background point `cloud` plus foreground `loops`.
    `panels` is a list of (x_index, y_index) angle-column pairs; the default
    ((0, 2), (0, 1)) draws ψ-vs-φ and θ-vs-φ. The cloud is subsampled once and
    shared across panels; the legend (one entry per logical trace) and the phase
    colorbar are shown only on the first panel so they don't duplicate.
    """
    go = _import_plotly()
    try:
        from plotly.subplots import make_subplots
    except ImportError as e:
        raise ImportError(
            "plotly is required for wingbeat angle-space plots (pip install plotly)."
        ) from e

    n_panels = len(panels)
    subplot_titles = [f"{ANGLE_SHORT[yi]} vs {ANGLE_SHORT[xi]}" for xi, yi in panels]
    fig = make_subplots(rows=1, cols=n_panels, subplot_titles=subplot_titles,
                        horizontal_spacing=0.10)

    pts = phase = None
    if cloud is not None and len(cloud) > 0:
        pts, phase = _concat_points_phase(cloud, units)
        if pts.shape[0] > cloud_max_points:
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.choice(pts.shape[0], size=cloud_max_points, replace=False))
            pts, phase = pts[idx], phase[idx]

    for ci, (xi, yi) in enumerate(panels, start=1):
        first = ci == 1
        if pts is not None:
            _add_cloud_2d(
                go, fig, 1, ci, pts, phase, xi, yi,
                color_by_phase=cloud_color_by_phase,
                show_colorbar=cloud_show_colorbar and first,
                opacity=cloud_opacity, size=cloud_marker_size, showlegend=first,
            )
        for spec in (loops or []):
            _add_loop_2d(go, fig, 1, ci, spec, units, xi, yi, showlegend=first)
        fig.update_xaxes(title_text=ANGLE_AXIS_TITLES[xi], row=1, col=ci)
        # Equal axes: 1 unit on y == 1 unit on x for this panel, so loop geometry
        # is undistorted. scaleanchor must name the panel's own x axis (x, x2, …).
        xref = "x" if ci == 1 else f"x{ci}"
        fig.update_yaxes(title_text=ANGLE_AXIS_TITLES[yi], scaleanchor=xref,
                         scaleratio=1, row=1, col=ci)

    fig.update_layout(
        title=title, width=width, height=height, template="plotly_white",
        # Legend horizontal below the panels, colorbar at the right edge — neither
        # overlaps the other or the plot area.
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.14, yanchor="top"),
        margin=dict(b=120, r=110, t=80),
    )
    return fig


# ---------------------------------------------------------------------------
# Convenience entry points
# ---------------------------------------------------------------------------


def plot_wingbeats_cloud(
    wingbeats,
    out_path: str,
    *,
    title: str = "Wingbeat angle space — point cloud",
    units: str = "deg",
    **kwargs,
) -> str:
    """Scatter a set of (L_i, 3) wingbeats as one phase-colored point cloud → HTML."""
    fig = make_angle_space_figure(cloud=wingbeats, units=units, title=title, **kwargs)
    return write_html(fig, out_path)


def plot_single_wingbeat(
    wingbeat,
    out_path: str,
    *,
    name: str = "wingbeat",
    title: str = "Wingbeat angle space — single wingbeat",
    units: str = "deg",
    color: str = "crimson",
    color_by_phase: bool = True,
    **kwargs,
) -> str:
    """Draw one (L, 3) wingbeat as a closed, phase-colored loop with a start marker → HTML."""
    loop = dict(
        name=name, angles=wingbeat, units=units, color=color,
        color_by_phase=color_by_phase, show_colorbar=color_by_phase,
        markers=True, close=True, mark_start=True, width=5,
    )
    fig = make_angle_space_figure(cloud=None, loops=[loop], units=units, title=title, **kwargs)
    return write_html(fig, out_path)


def wingbeats_from_trajectory(traj, wing: str = "both", *, resample_L: int | None = None):
    """
    Segment a raw (T, 6) flight trajectory into per-wing wingbeats using the same
    stroke-peak boundaries as the dataset builder, and return a list of (n_i, 3)
    [φ, θ, ψ] arrays. `wing` is 'left', 'right', or 'both' (both wings appended as
    independent wingbeats). If `resample_L` is given, each wingbeat is
    CubicSpline-resampled to that length. Units follow the input (raw npy = radians).
    """
    # Lazy import keeps this module free of a transform_data dependency at import
    # time (transform_data imports *this* module for the template plot).
    from transform_data import _wingbeat_peaks, _cubic_resample

    traj = np.asarray(traj)
    if traj.ndim != 2 or traj.shape[1] != 6:
        raise ValueError(f"expected a (T, 6) trajectory, got shape {traj.shape}.")
    col_sets = {
        "left":  [(0, 1, 2)],
        "right": [(3, 4, 5)],
        "both":  [(0, 1, 2), (3, 4, 5)],
    }.get(wing)
    if col_sets is None:
        raise ValueError(f"wing must be 'left', 'right', or 'both', got {wing!r}.")

    peaks = _wingbeat_peaks(traj)
    out = []
    for i in range(len(peaks) - 1):
        seg = traj[int(peaks[i]):int(peaks[i + 1])]
        if seg.shape[0] < 2:
            continue
        for cols in col_sets:
            wb = seg[:, list(cols)]
            if resample_L:
                wb = _cubic_resample(wb, resample_L)
            out.append(np.asarray(wb, dtype=np.float64))
    return out


def plot_trajectory_cloud(
    traj,
    out_path: str,
    *,
    wing: str = "both",
    units: str = "rad",
    title: str | None = None,
    **kwargs,
) -> str:
    """Segment a (T, 6) flight into wingbeats and plot them as one point cloud → HTML."""
    wbs = wingbeats_from_trajectory(traj, wing=wing)
    if title is None:
        title = f"Wingbeat angle space — trajectory point cloud ({wing} wing, {len(wbs)} beats)"
    return plot_wingbeats_cloud(wbs, out_path, title=title, units=units, **kwargs)


# ---------------------------------------------------------------------------
# CLI (quick look at a template, a single trajectory, or the whole dataset)
# ---------------------------------------------------------------------------


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Render a 3D wing-angle-space view (point cloud or single loop).",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--template", help="A golden-template .npy ((L,6) or (L,3)) to draw as loops.")
    src.add_argument("--trajectories", help="A trajectories.npy (object array of (T,6) flights).")
    parser.add_argument("--traj_index", type=int, default=0,
                        help="With --trajectories: which flight to plot (default 0). Ignored with --all.")
    parser.add_argument("--all", action="store_true",
                        help="With --trajectories: pool every flight's wingbeats into one cloud.")
    parser.add_argument("--wing", choices=["left", "right", "both"], default="both")
    parser.add_argument("--out", default=None, help="Output .html path (default: next to the source).")
    parser.add_argument("--cloud_max_points", type=int, default=20000)
    args = parser.parse_args()

    if args.template:
        template = np.load(args.template)
        out = args.out or (os.path.splitext(args.template)[0] + "_angle_space.html")
        if template.shape[1] == 6:
            loops = [
                dict(name="Left wing",  angles=template[:, 0:3], units="rad",
                     color=WING_COLORS["left"],  markers=True, close=True, mark_start=True, width=5),
                dict(name="Right wing", angles=template[:, 3:6], units="rad",
                     color=WING_COLORS["right"], markers=True, close=True, mark_start=True, width=5),
            ]
            fig = make_angle_space_figure(loops=loops, title=f"Golden template — wing angle space ({args.template})")
            write_html(fig, out)
        else:
            plot_single_wingbeat(template, out, units="rad", name="template",
                                 title=f"Golden template — wing angle space ({args.template})")
        print(f"→ wrote {out}")
        return

    trajectories = np.load(args.trajectories, allow_pickle=True)
    if args.all:
        wbs = []
        for traj in trajectories:
            wbs.extend(wingbeats_from_trajectory(traj, wing=args.wing))
        out = args.out or (os.path.splitext(args.trajectories)[0] + f"_angle_space_cloud_{args.wing}.html")
        plot_wingbeats_cloud(wbs, out, units="rad", cloud_max_points=args.cloud_max_points,
                             title=f"Wingbeat angle space — all flights ({args.wing} wing, {len(wbs)} beats)")
    else:
        traj = trajectories[args.traj_index]
        out = args.out or (os.path.splitext(args.trajectories)[0] + f"_angle_space_traj{args.traj_index}_{args.wing}.html")
        plot_trajectory_cloud(traj, out, wing=args.wing, units="rad",
                              cloud_max_points=args.cloud_max_points)
    print(f"→ wrote {out}")


if __name__ == "__main__":
    _main()
