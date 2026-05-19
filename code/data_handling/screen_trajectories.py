"""
Visual screening tool for wing-angle trajectories.

For every trajectory in `data/trajectories.npy`, writes a PNG plot of its 6 wing
angles (left/right × phi/theta/psi) with detected wingbeat peaks overlaid. Also
writes a sortable `summary.csv` and an `index.html` thumbnail grid sorted by a
suspicion score so non-periodic / broken trajectories surface first.

After running this, eyeball the high-suspicion trajectories and create
`data/excluded_trajectories.json`:

    {
      "excluded_indices": [17, 42, 103],
      "notes": "17: non-periodic; 42: tracking dropped mid-flight; 103: garbage."
    }

Then point `autoencoder_config.json` at it via the `excluded_trajectories_path`
key — training will skip those trajectories before the train/val split.

Usage (from project root):
    python code/data_handling/screen_trajectories.py
    python code/data_handling/screen_trajectories.py --data_path data/trajectories.npy
"""
import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Allow importing transform_data from sibling directory `code/`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transform_data import _wingbeat_peaks, SA_PHYSICAL_SCALE, trajectory_asymmetry_score


def _per_trajectory_stats(traj: np.ndarray) -> dict:
    """
    Computes summary statistics + detected peaks for one trajectory.

    The L-R asymmetry metric: per-angle mean(|L - R|) over all time samples,
    normalized by that angle's physical scale (π for φ/ψ, 0.5 for θ). The
    `asymmetry_score` is the max across angles — one bad angle is enough to
    flag the trajectory, even if the others are fine.
    """
    peaks = _wingbeat_peaks(traj)
    n_wb = max(0, len(peaks) - 1)
    mean_wb_len = float(np.diff(peaks).mean()) if n_wb > 0 else 0.0

    # Column order in `traj`: [L_phi, L_theta, L_psi, R_phi, R_theta, R_psi]
    phi_LR_mean   = float(np.abs(traj[:, 0] - traj[:, 3]).mean())
    theta_LR_mean = float(np.abs(traj[:, 1] - traj[:, 4]).mean())
    psi_LR_mean   = float(np.abs(traj[:, 2] - traj[:, 5]).mean())

    # asymmetry_score is the metric the autoencoder's auto-filter uses too.
    asymmetry_score = trajectory_asymmetry_score(traj)

    return {
        'n_samples':        int(len(traj)),
        'n_peaks':          int(len(peaks)),
        'n_wingbeats':      int(n_wb),
        'mean_wb_length':   mean_wb_len,
        'phi_LR_mean':      phi_LR_mean,
        'theta_LR_mean':    theta_LR_mean,
        'psi_LR_mean':      psi_LR_mean,
        'asymmetry_score':  asymmetry_score,
        'peaks':            peaks,
    }


def _suspicion_score(stats: dict, median_asymmetry: float) -> float:
    """
    Suspicion = ratio of the trajectory's asymmetry score to the dataset median.
    A value of 1.0 is typical. Values >> 1 indicate the L-R gap is much larger
    than usual, which is the signature of garbage trajectories.
    """
    if median_asymmetry <= 0:
        return 0.0
    return stats['asymmetry_score'] / median_asymmetry


def _plot_trajectory(
    traj: np.ndarray,
    peaks: np.ndarray,
    idx: int,
    suspicion: float,
    asymmetry_score: float,
    median_asymmetry: float,
    out_path: str,
) -> None:
    """3 angle panels (L/R overlay) + a 4th panel of normalized |L-R| per angle."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    angle_labels = ['Stroke φ [rad]', 'Deviation θ [rad]', 'Rotation ψ [rad]']
    left_cols   = [0, 1, 2]
    right_cols  = [3, 4, 5]
    x = np.arange(len(traj))

    for ax, label, lc, rc in zip(axes[:3], angle_labels, left_cols, right_cols):
        ax.plot(x, traj[:, lc], color='blue', lw=0.8, label='Left')
        ax.plot(x, traj[:, rc], color='red',  lw=0.8, label='Right')
        for p in peaks:
            ax.axvline(p, color='gray', lw=0.4, ls='--', alpha=0.4)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

    # Bottom panel: |L-R| normalized by each angle's physical scale, all 3 on one axis.
    diffs_normalized = [
        np.abs(traj[:, lc] - traj[:, rc]) / float(SA_PHYSICAL_SCALE[lc])
        for lc, rc in zip(left_cols, right_cols)
    ]
    diff_colors = ['tab:blue', 'tab:green', 'tab:purple']
    diff_labels = ['|L-R| φ / π', '|L-R| θ / 0.5', '|L-R| ψ / π']
    for diff, color, dl in zip(diffs_normalized, diff_colors, diff_labels):
        axes[3].plot(x, diff, color=color, lw=0.7, label=dl)
    # Dataset median asymmetry as a reference line — a typical trajectory hovers near this value.
    if median_asymmetry > 0:
        axes[3].axhline(median_asymmetry, color='gray', lw=1.0, ls='--',
                        label=f'dataset median = {median_asymmetry:.3f}')
    axes[3].set_ylabel('|L−R| (normalized)')
    axes[3].legend(loc='upper right', fontsize=8, ncol=2)
    axes[3].grid(True, alpha=0.3)

    if suspicion >= 6.0:
        title_color = 'red'
    elif suspicion >= 3.0:
        title_color = 'darkorange'
    else:
        title_color = 'black'
    fig.suptitle(
        f"Trajectory {idx:03d}  ·  n={len(traj)} samples  ·  {len(peaks)} peaks  ·  "
        f"asymmetry={asymmetry_score:.3f}  ·  suspicion={suspicion:.1f}×median",
        color=title_color, fontsize=12,
    )
    axes[0].legend(loc='upper right', fontsize=8)
    axes[-1].set_xlabel('Sample index')
    plt.tight_layout()
    fig.savefig(out_path, dpi=80, bbox_inches='tight')
    plt.close(fig)


def _write_summary_csv(rows: list, out_path: str) -> None:
    keys = ['idx', 'suspicion', 'asymmetry_score',
            'phi_LR_mean', 'theta_LR_mean', 'psi_LR_mean',
            'n_samples', 'n_peaks', 'n_wingbeats', 'mean_wb_length']
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(keys)
        for r in rows:
            w.writerow([f'{r[k]:.4f}' if isinstance(r[k], float) else r[k] for k in keys])


def _write_index_html(rows: list, out_path: str) -> None:
    """Thumbnail grid sorted high-suspicion → low. Click thumbnail for full image."""
    rows = sorted(rows, key=lambda r: -r['suspicion'])
    with open(out_path, 'w') as f:
        f.write('<!doctype html><html><head><meta charset="utf-8"><title>Trajectory screening</title>')
        f.write('<style>'
                'body{font-family:sans-serif;margin:20px;background:#fafafa}'
                'h1{margin-bottom:5px}'
                '.legend{margin:10px 0 20px;color:#555}'
                '.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:15px}'
                '.card{background:white;border:1px solid #ddd;padding:8px;border-radius:4px}'
                '.card.high{border-color:#d33;border-width:2px}'
                '.card.medium{border-color:#e80;border-width:2px}'
                '.card-title{font-weight:bold;margin-bottom:5px}'
                '.card-stats{color:#666;font-size:0.9em;margin-top:5px}'
                '.card img{width:100%;height:auto;display:block}'
                '</style></head><body>')
        f.write(f'<h1>Trajectory screening — {len(rows)} trajectories</h1>')
        f.write('<div class="legend">Sorted by suspicion (high → low). '
                'Suspicion = trajectory asymmetry score divided by the dataset median asymmetry. '
                '<span style="color:#d33">Red border</span> = ≥6×. '
                '<span style="color:#e80">Orange border</span> = ≥3×. '
                'Click a thumbnail for the full image (includes a |L-R| panel).</div>')
        f.write('<div class="grid">')
        for r in rows:
            cls = 'high' if r['suspicion'] >= 6 else ('medium' if r['suspicion'] >= 3 else '')
            fname = f'traj_{r["idx"]:03d}.png'
            f.write(f'<div class="card {cls}">')
            f.write(f'<div class="card-title">#{r["idx"]:03d} — suspicion {r["suspicion"]:.1f}×</div>')
            f.write(f'<a href="{fname}" target="_blank"><img src="{fname}" loading="lazy"></a>')
            f.write(f'<div class="card-stats">asymmetry={r["asymmetry_score"]:.3f}  '
                    f'peaks={r["n_peaks"]}  wb={r["n_wingbeats"]}  n_samples={r["n_samples"]}</div>')
            f.write('</div>')
        f.write('</div></body></html>')


def main() -> None:
    parser = argparse.ArgumentParser(description="Visual screening for wing-angle trajectories")
    parser.add_argument('--data_path', default='data/trajectories.npy', help='Path to trajectories .npy')
    parser.add_argument('--out_dir',   default='data/analysis/trajectory_screening',
                        help='Output directory (PNGs + summary + index)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    trajectories = np.load(args.data_path, allow_pickle=True)
    n = len(trajectories)
    print(f'Loaded {n} trajectories from {args.data_path}')

    # First pass: per-trajectory stats. Needed for the dataset-median asymmetry that
    # anchors the suspicion ratio.
    all_stats = [_per_trajectory_stats(t) for t in trajectories]
    median_asymmetry = float(np.median([s['asymmetry_score'] for s in all_stats])) if all_stats else 0.0
    print(f'Dataset median asymmetry score: {median_asymmetry:.4f}')

    rows = []
    for idx, (traj, stats) in enumerate(zip(trajectories, all_stats)):
        susp = _suspicion_score(stats, median_asymmetry)
        png_path = os.path.join(args.out_dir, f'traj_{idx:03d}.png')
        _plot_trajectory(traj, stats['peaks'], idx, susp, stats['asymmetry_score'], median_asymmetry, png_path)

        row = {k: v for k, v in stats.items() if k != 'peaks'}
        row['idx']       = idx
        row['suspicion'] = susp
        rows.append(row)

        if (idx + 1) % 20 == 0 or idx == n - 1:
            print(f'  plotted {idx + 1}/{n}')

    _write_summary_csv(rows, os.path.join(args.out_dir, 'summary.csv'))
    _write_index_html(rows, os.path.join(args.out_dir, 'index.html'))

    n_high = sum(1 for r in rows if r['suspicion'] >= 6)
    n_med  = sum(1 for r in rows if 3 <= r['suspicion'] < 6)
    print(f'\nFlagged: {n_high} high-suspicion (≥6×), {n_med} moderate (≥3×), {n - n_high - n_med} clean.')
    print(f'Output : {args.out_dir}')
    print(f'  Open  {os.path.join(args.out_dir, "index.html")}  in a browser to scan.')


if __name__ == '__main__':
    main()
