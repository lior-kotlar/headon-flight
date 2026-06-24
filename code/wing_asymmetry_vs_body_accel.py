"""
wing_asymmetry_vs_body_accel.py — does a single-wing latent direction track body
angular acceleration?

For the single-wing autoencoder, every physical wingbeat is encoded as two latent
vectors — its LEFT wing and its RIGHT wing. This tool projects both onto a chosen
latent direction (a PCA component by default, or a raw latent dim), forms the two
per-wingbeat combinations

    diff = proj_left − proj_right   (antisymmetric — survives for roll/yaw maneuvers)
    sum  = proj_left + proj_right   (symmetric    — survives for pitch maneuvers)

and scatters each against the three body-frame angular accelerations
(α_yaw, α_pitch, α_roll), one point per wingbeat. Because left and right wing angles
are stored in the SAME sign convention (transform_data.generate_single_wing_template),
a symmetric maneuver (pitch) lands in `sum` and cancels in `diff`; an antisymmetric
one (roll/yaw) lands in `diff`. The 2×3 panel grid lets the data say which body axis
each combination tracks.

The PCA basis and the encode are reused verbatim from inspect_latent_space, so
"PC3" here is the exact component drawn in that tool's angle_space HTMLs.

The x-axis quantity is a selectable body-kinematics proxy (--target): the current
beat's angular accel (default), the NEXT beat's accel, the mean of this+next, or the
across-beat change in angular velocity ω_next−ω_this. Columns are the body axes
(--axes; default yaw/pitch/roll).

Run from the project root:

    python code/wing_asymmetry_vs_body_accel.py \
        --model_dir data/models/autoencoder_single_wing/ae_sw_latsweep_20260617_174025/latent_dim_4
    # several directions / proxies / a subset of axes at once (one figure per combo):
    python code/wing_asymmetry_vs_body_accel.py --model_dir <dir> \
        --component pc:0 pc:1 --target accel accel_next dvel_next --axes pitch roll
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Sibling imports work whether invoked as `python code/...` or with code/ on path.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from inspect_latent_space import _load_inputs, _compute_pca

# body_means layout is [v(3), a(3), ω(3), α(3)], each triple ordered (yaw, pitch, roll).
# So angular VELOCITY ω is cols 6-8 and angular ACCELERATION α is cols 9-11 (the latter
# matching inspect_latent_space / transform_data._BODY_ALPHA_COLS).
_AXIS_COLS = {
    "yaw":   {"omega": 6, "alpha":  9},
    "pitch": {"omega": 7, "alpha": 10},
    "roll":  {"omega": 8, "alpha": 11},
}
_ALL_AXES = ["yaw", "pitch", "roll"]

# The two per-wingbeat wing combinations, row order in the panel grid.
_COMBOS = ["diff (L−R)", "sum (L+R)"]


def _parse_component(spec: str) -> tuple[str, int]:
    """'pc:3' → ('pc', 3); 'dim:1' → ('dim', 1). Bare 'N' is treated as a PC."""
    spec = spec.strip().lower()
    if ":" in spec:
        kind, idx = spec.split(":", 1)
    else:
        kind, idx = "pc", spec
    kind = kind.strip()
    if kind not in ("pc", "dim"):
        raise ValueError(f"component kind must be 'pc' or 'dim'; got {kind!r} (from {spec!r}).")
    return kind, int(idx)


def _direction_vector(kind: str, idx: int, pca: dict, latent_dim: int) -> tuple[np.ndarray, str, float | None]:
    """
    Return (unit direction in latent space, human label, explained_ratio|None).
    'pc' → the idx-th PCA component; 'dim' → the idx-th raw-latent one-hot axis.
    """
    if kind == "pc":
        n_pc = pca["components"].shape[0]
        if not 0 <= idx < n_pc:
            raise ValueError(f"PC index {idx} out of range [0, {n_pc}).")
        comp = pca["components"][idx].astype(np.float64)
        return comp, f"PC{idx}", float(pca["explained_ratio"][idx])
    if not 0 <= idx < latent_dim:
        raise ValueError(f"latent dim index {idx} out of range [0, {latent_dim}).")
    comp = np.zeros(latent_dim, dtype=np.float64)
    comp[idx] = 1.0
    return comp, f"dim{idx}", None


# ---------------------------------------------------------------------------
# Body-kinematics target proxies (the x-axis quantity), selectable via --target
# ---------------------------------------------------------------------------
#
# Each builder maps the per-wingbeat body kinematics to ONE scalar per wingbeat for a
# given axis, plus a validity mask (False where the proxy is undefined — e.g. the last
# wingbeat of a trajectory has no "next"). `nxt[i]` is the row of the next wingbeat in
# the same trajectory, or -1 if none. `cols` is _AXIS_COLS[axis] (its ω and α columns).
#
# All proxies inherit the dataset's L/R-mirror symmetry: the augmentation mirrors whole
# trajectories, so next-beat yaw/roll flip sign and pitch is kept, exactly like this-beat
# values. Hence the structural zeros (diff·pitch, sum·yaw, sum·roll ≡ 0) hold for every
# proxy — a built-in correctness check on the next-beat indexing.


def _next_index(traj_ids_N: np.ndarray) -> np.ndarray:
    """Row of the next wingbeat within the same trajectory, or -1 for the last beat of a
    trajectory. The builder stores wingbeats sequentially within each trajectory, so the
    next row is i+1 whenever it shares the trajectory id."""
    n = traj_ids_N.shape[0]
    nxt = np.full(n, -1, dtype=np.int64)
    same = traj_ids_N[1:] == traj_ids_N[:-1]
    i = np.nonzero(same)[0]
    nxt[i] = i + 1
    return nxt


def _target_accel(bm, nxt, cols):
    """Angular acceleration of THIS wingbeat (the original proxy)."""
    return bm[:, cols["alpha"]].astype(np.float64), np.ones(bm.shape[0], dtype=bool)


def _target_accel_next(bm, nxt, cols):
    """Angular acceleration of the NEXT wingbeat — does the current wing state lead it?"""
    valid = nxt >= 0
    y = np.full(bm.shape[0], np.nan)
    y[valid] = bm[nxt[valid], cols["alpha"]]
    return y, valid


def _target_accel_mean(bm, nxt, cols):
    """Mean angular acceleration of THIS and the next wingbeat."""
    valid = nxt >= 0
    a = bm[:, cols["alpha"]].astype(np.float64)
    y = np.full(bm.shape[0], np.nan)
    y[valid] = 0.5 * (a[valid] + a[nxt[valid]])
    return y, valid


def _target_dvel_next(bm, nxt, cols):
    """Change in angular VELOCITY across to the next wingbeat: ω_next − ω_this — a
    finite-difference 'how much rotation did this beat actually impart to the body'."""
    valid = nxt >= 0
    w = bm[:, cols["omega"]].astype(np.float64)
    y = np.full(bm.shape[0], np.nan)
    y[valid] = w[nxt[valid]] - w[valid]
    return y, valid


# name → (human label for plot/JSON, builder). Pick with --target (multi-valued).
_TARGETS = {
    "accel":      ("α (this beat)",         _target_accel),
    "accel_next": ("α (next beat)",         _target_accel_next),
    "accel_mean": ("mean α (this+next)",    _target_accel_mean),
    "dvel_next":  ("Δω (ω_next − ω_this)",  _target_dvel_next),
}


def _binned_trend(x: np.ndarray, y: np.ndarray, n_bins: int = 12):
    """
    Equal-count (quantile) x-bins → (bin_x_mean, bin_y_mean, bin_y_SE). Robust to the
    heavy-tailed α distribution: bins hold equal counts rather than equal width.
    """
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order]
    edges = np.linspace(0, len(x), n_bins + 1).astype(int)
    bx, by, bse = [], [], []
    for b in range(n_bins):
        s, e = edges[b], edges[b + 1]
        if e - s < 1:
            continue
        seg = ys[s:e]
        bx.append(float(xs[s:e].mean()))
        by.append(float(seg.mean()))
        bse.append(float(seg.std(ddof=1) / np.sqrt(len(seg))) if len(seg) > 1 else 0.0)
    return np.array(bx), np.array(by), np.array(bse)


def _panel_stats(x: np.ndarray, y: np.ndarray) -> dict:
    """Pearson r, Spearman ρ, and an OLS slope/intercept — all over every point."""
    r, _   = stats.pearsonr(x, y)
    rho, _ = stats.spearmanr(x, y)
    slope, intercept = np.polyfit(x, y, 1)
    return {
        "pearson_r":   float(r),
        "spearman_rho": float(rho),
        "slope":       float(slope),
        "intercept":   float(intercept),
        "n":           int(len(x)),
    }


def _draw_panel(ax, x, y, st, *, axis_name, combo_label, clip_pct):
    xlo, xhi = np.percentile(x, clip_pct), np.percentile(x, 100 - clip_pct)
    ax.scatter(x, y, s=4, alpha=0.12, color="0.4", linewidths=0, zorder=1)
    bx, by, bse = _binned_trend(x, y)
    ax.errorbar(bx, by, yerr=bse, color="tab:red", marker="o", ms=4, lw=1.6,
                capsize=2, zorder=5, label="binned mean ± SE")
    xfit = np.linspace(xlo, xhi, 100)
    ax.plot(xfit, st["slope"] * xfit + st["intercept"], color="tab:blue", lw=1.6,
            zorder=4, label="linear fit")
    ax.axhline(0.0, color="0.7", lw=0.8, zorder=0)
    ax.set_xlim(xlo, xhi)
    ax.grid(True, alpha=0.3)
    ax.set_title(
        f"{combo_label}  vs  {axis_name}\n"
        f"r={st['pearson_r']:+.3f}   ρ={st['spearman_rho']:+.3f}   "
        f"slope={st['slope']:+.3g}   n={st['n']}",
        fontsize=10,
    )


def _analyze_component(
    spec: str, ctx: dict, pca: dict, body_means_N: np.ndarray, nxt: np.ndarray,
    target_key: str, axes: list[str], out_dir: str, lat_prefix: str, clip_pct: float,
) -> None:
    kind, idx = _parse_component(spec)
    latent_dim = ctx["latent_dim"]
    comp, label, expl = _direction_vector(kind, idx, pca, latent_dim)
    target_label, target_fn = _TARGETS[target_key]

    z_all  = ctx["z_all"].detach().cpu().numpy().astype(np.float64)              # (2N, D)
    z_mean = ctx["z_mean"].detach().cpu().numpy().astype(np.float64)             # (D,)

    proj = (z_all - z_mean) @ comp                                              # (2N,)
    proj_pairs = proj.reshape(-1, 2)                                           # (N, 2): [left, right]
    left, right = proj_pairs[:, 0], proj_pairs[:, 1]
    combos = {
        "diff (L−R)": left - right,
        "sum (L+R)":  left + right,
    }

    n_axes = len(axes)
    fig, grid = plt.subplots(2, n_axes, figsize=(5.5 * n_axes, 9.5), squeeze=False)
    n_wb = proj_pairs.shape[0]
    expl_txt = f", {100 * expl:.1f}% of latent var" if expl is not None else ""
    fig.suptitle(
        f"Single-wing latent asymmetry vs body kinematics — {label}{expl_txt}\n"
        f"target = {target_label}   ({n_wb} wingbeats, latent_dim={latent_dim}, "
        f"projection sign is arbitrary)",
        fontsize=13,
    )

    results: dict[str, dict] = {}
    for row, combo_label in enumerate(_COMBOS):
        y_proj = combos[combo_label]
        for col, axis_name in enumerate(axes):
            tgt, valid = target_fn(body_means_N, nxt, _AXIS_COLS[axis_name])
            m = valid & np.isfinite(tgt) & np.isfinite(y_proj)
            x, y = tgt[m], y_proj[m]
            st = _panel_stats(x, y)
            results[f"{combo_label} | {axis_name}"] = st
            ax = grid[row][col]
            _draw_panel(ax, x, y, st, axis_name=axis_name, combo_label=combo_label,
                        clip_pct=clip_pct)
            if col == 0:
                ax.set_ylabel(f"{label} projection — {combo_label}", fontsize=10)
            if row == 1:
                ax.set_xlabel(f"{axis_name}:  {target_label}", fontsize=10)
    grid[0][n_axes - 1].legend(fontsize=8, loc="upper right")

    # The dataset is L/R-mirror augmented (process_data.augment_dataset): the augmentation
    # mirrors whole trajectories — wings swapped (diff→−diff, sum unchanged) and yaw/roll
    # sign-flipped, pitch kept — for THIS-beat and NEXT-beat quantities alike. So for every
    # target the symmetric combination (sum) is exactly uncorrelated with yaw & roll, and
    # the antisymmetric one (diff) exactly uncorrelated with pitch: structural zeros.
    fig.text(0.5, 0.005,
             "Mirror augmentation: sum (L+R) ≡ uncorrelated with yaw & roll; "
             "diff (L−R) ≡ uncorrelated with pitch (structural zeros, not findings).",
             ha="center", fontsize=8, style="italic", color="0.35")

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    tag      = f"{kind}{idx:03d}_{target_key}"
    png_path = os.path.join(out_dir, f"{lat_prefix}{tag}.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    json_path = os.path.join(out_dir, f"{lat_prefix}{tag}.json")
    with open(json_path, "w") as f:
        json.dump({
            "component":         label,
            "component_spec":    f"{kind}:{idx}",
            "target":            target_key,
            "target_label":      target_label,
            "axes":              list(axes),
            "latent_dim":        latent_dim,
            "explained_ratio":   expl,
            "n_wingbeats":       n_wb,
            "npz_path":          ctx["npz_path"],
            "correlations":      results,
        }, f, indent=2)

    print(f"\n  [{label} | {target_label}]" + (f"  ({100 * expl:.1f}% var)" if expl is not None else ""))
    print(f"    {'combination':<12} {'axis':<6} {'pearson_r':>10} {'spearman_ρ':>11} {'slope':>11} {'n':>7}")
    for key, st in results.items():
        combo, axis = key.split(" | ")
        print(f"    {combo:<12} {axis:<6} {st['pearson_r']:>+10.3f} "
              f"{st['spearman_rho']:>+11.3f} {st['slope']:>+11.3g} {st['n']:>7}")
    print(f"  → wrote {png_path}")
    print(f"  → wrote {json_path}", flush=True)


def _assert_pairing(wing_side: np.ndarray, body_means: np.ndarray) -> None:
    """The single-wing build appends each wingbeat's LEFT then RIGHT row, with
    body_means duplicated across the pair. Verify before relying on the (N,2) reshape."""
    if wing_side.shape[0] % 2 != 0:
        raise ValueError(f"odd number of single-wing rows ({wing_side.shape[0]}); expected 2 per wingbeat.")
    if not (np.all(wing_side[0::2] == 0) and np.all(wing_side[1::2] == 1)):
        raise ValueError("wing_side is not strictly [left, right, left, right, ...]; "
                         "the (N,2) left/right pairing assumption does not hold.")
    if not np.allclose(body_means[0::2], body_means[1::2]):
        raise ValueError("body_means differs between paired left/right rows; expected it "
                         "to be duplicated per wingbeat.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.strip(), formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model_dir", required=True,
                        help="Single-wing model dir holding best_autoencoder.pt and best_config.json.")
    parser.add_argument("--component", nargs="+", default=["pc:3"],
                        help="Latent direction(s) to project on: 'pc:K' (PCA component, default pc:3) "
                             "or 'dim:K' (raw latent dim). Pass several for one figure each.")
    parser.add_argument("--target", nargs="+", default=["accel"], choices=list(_TARGETS),
                        help="Body-kinematics proxy on the x-axis: accel=this beat's α; "
                             "accel_next=next beat's α; accel_mean=mean(this,next) α; "
                             "dvel_next=ω_next−ω_this. Pass several for one figure each (default accel).")
    parser.add_argument("--axes", nargs="+", default=_ALL_AXES, choices=_ALL_AXES,
                        help="Which body axes to plot as columns (default: all three). "
                             "E.g. --axes pitch roll to drop yaw.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--out_dir", default=None,
                        help="Default: <model_dir>/latent_space_inspection/asymmetry_vs_accel/")
    parser.add_argument("--npz_path", default=None,
                        help="Override the fixed-L npz path. Default: derived from best_config.json.")
    parser.add_argument("--clip_pct", type=float, default=1.0,
                        help="Clip the x display range to [p, 100-p] so the heavy α tail doesn't "
                             "compress the bulk (stats still use every point). Default: 1.")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ctx = _load_inputs(args.model_dir, device, args.npz_path)
    if ctx["representation"] != "single_wing":
        raise SystemExit(
            f"This tool needs the single_wing representation (left/right per wingbeat); "
            f"model is '{ctx['representation']}'."
        )

    d = np.load(ctx["npz_path"])
    for key in ("body_means", "wing_side", "trajectory_ids"):
        if key not in d.files:
            raise SystemExit(f"{ctx['npz_path']} has no '{key}' array — cannot run this analysis.")
    body_means = d["body_means"]
    wing_side  = d["wing_side"]
    if body_means.shape[0] != ctx["z_all"].shape[0]:
        raise SystemExit(
            f"body_means rows ({body_means.shape[0]}) ≠ encoded wingbeats "
            f"({ctx['z_all'].shape[0]}) — refusing to plot a misaligned correlation."
        )
    _assert_pairing(wing_side, body_means)

    # Collapse to one row per physical wingbeat (the L/R rows are duplicates), then index
    # each wingbeat's next beat within the same trajectory for the next-beat proxies.
    body_means_N   = body_means[0::2]                       # (N, 12)
    trajectory_ids = np.asarray(d["trajectory_ids"][0::2])  # (N,)
    nxt            = _next_index(trajectory_ids)

    out_dir = args.out_dir or os.path.join(args.model_dir, "latent_space_inspection", "asymmetry_vs_accel")
    os.makedirs(out_dir, exist_ok=True)

    pca = _compute_pca(ctx["z_all"], ctx["z_mean"])
    for target_key in args.target:
        for spec in args.component:
            _analyze_component(spec, ctx, pca, body_means_N, nxt, target_key, args.axes,
                               out_dir, ctx["lat_prefix"], args.clip_pct)

    print(f"\nDone. Outputs in: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
