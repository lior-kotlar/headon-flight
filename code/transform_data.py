"""
transform_data.py — wingbeat extraction and fixed-length dataset builder.

This module has two responsibilities:

  1. Extract per-wingbeat segments from raw flight trajectories and build the
     S/A "golden template" used to convert each wingbeat to its
     (symmetric, asymmetric) residual representation.
  2. Optionally produce a fixed-length wingbeat dataset (wingbeats_L<L>.npz)
     where every wingbeat is CubicSpline-resampled to a constant length L.
     This is consumed by the fixed-L autoencoder so the network learns
     wingbeat *shape* without having to also learn duration.

------------------------------------------------------------------------------
CLI usage (run from the project root)
------------------------------------------------------------------------------

# Build the variable-length artifacts (trajectories.npy + body_kinematics.npy
# + golden_template.npy):
python code/transform_data.py

# Additionally build a fixed-length wingbeat dataset at L=80 samples
# (also computes per-wingbeat body_means and maneuver_scores):
python code/transform_data.py --fixed_len 80

# Always-emitted files (next to trajectories.npy):
#   data/trajectories.npy     — object array of (T_i, 6) wing-angle arrays per flight
#   data/body_kinematics.npy  — object array of (T_i, 12) body-kinematic arrays per
#                               flight, columns [v(3), a(3), ω(3), α(3)]. Same indexing
#                               as trajectories.npy — the asymmetry filter prunes both
#                               in lockstep.
#
# Fixed-L build adds:
#   data/wingbeats_L80.npz   — arrays: sa_wingbeats (N, 6, L), body_means (N, 12),
#                              maneuver_scores (N, 6), durations (N,),
#                              trajectory_ids (N,)
#   data/wingbeats_L80.json  — sidecar: schema_version, L, interpolation method,
#                              build timestamp, md5 hashes of trajectories.npy +
#                              body_kinematics.npy + template, plus maneuver-scoring
#                              params (W, T_coh, T_mag_percentile, per-channel T_mag).
#                              Drift detection rebuilds the npz if any of those change.

------------------------------------------------------------------------------
Programmatic usage (e.g. from autoencoder.py auto-build hook)
------------------------------------------------------------------------------

from transform_data import (
    fixed_len_dataset_path,            # -> "data/wingbeats_L80.npz"
    fixed_len_dataset_is_valid,        # -> (bool, reason)
    build_fixed_len_dataset_from_disk, # rebuilds the npz + sidecar
    _cubic_resample,                   # (arr (n, C), L) -> (L, C) via CubicSpline
)

ok, reason = fixed_len_dataset_is_valid(data_dir="data", L=80)
if not ok:
    build_fixed_len_dataset_from_disk(L=80, data_dir="data")

------------------------------------------------------------------------------
Notes
------------------------------------------------------------------------------
- Interpolation is always CubicSpline (scipy.interpolate.CubicSpline). The
  sidecar records this as _FIXED_LEN_INTERP_METHOD so a future switch in
  method invalidates old caches.
- The sidecar's md5 hashes of trajectories.npy and the template file are how
  fixed_len_dataset_is_valid() detects stale caches without re-reading the
  full arrays.
- Channel order in sa_wingbeats is (S_phi, S_theta, S_psi, A_phi, A_theta,
  A_psi), normalized by SA_PHYSICAL_SCALE. Stored channels-first as (N, 6, L)
  to match the autoencoder's Conv1d expectations.
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime

import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import interp1d, CubicSpline
import matplotlib.pyplot as plt
from loguru import logger

# Allow imports from data_handling/ when run from the project root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_handling'))
from data_handling.process_data import PROCESSED_DATA_DIR, _extract_features_and_targets, gather_condensed_h5

# Column indices in the (N, 6) wing matrix: [L_phi, L_theta, L_psi, R_phi, R_theta, R_psi]
_L_PHI = 0
_R_PHI = 3

# Per-channel divisors for normalizing the S/A residual representation to roughly [-1, 1].
# Mirrors the physical bounds used by PhysicalWingNormalizer on raw angles — stroke φ and
# rotation ψ scale by π; deviation θ by 0.5 rad. S and A are linear combinations of L and R
# in the same units, so the same divisor vector applies.
# Order: [S_phi, S_theta, S_psi, A_phi, A_theta, A_psi]
SA_PHYSICAL_SCALE = np.array([np.pi, 0.5, np.pi, np.pi, 0.5, np.pi], dtype=np.float32)

# Per-channel divisors for the single-wing residual representation [phi, theta, psi].
# Same physical bounds as one half of SA_PHYSICAL_SCALE — a single wing's residual is
# in raw-angle units (not the L±R combination), so stroke φ / rotation ψ scale by π and
# deviation θ by 0.5 rad, identical to the per-angle scale used for L/R reporting.
SINGLE_WING_PHYSICAL_SCALE = np.array([np.pi, 0.5, np.pi], dtype=np.float32)


def trajectory_asymmetry_score(traj: np.ndarray) -> float:
    """
    Per-trajectory L/R asymmetry score. For each angle (phi/theta/psi), computes
    mean(|L - R|) over all time samples and divides by that angle's physical scale.
    Returns the max across the 3 angles — one badly-broken angle is enough to flag
    the trajectory. Used by screen_trajectories.py and by the autoencoder's
    auto-filter to detect garbage trajectories with unrealistic L-R gaps.

    Column order in `traj`: [L_phi, L_theta, L_psi, R_phi, R_theta, R_psi].
    """
    normalized = [
        float(np.abs(traj[:, lc] - traj[:, rc]).mean()) / float(SA_PHYSICAL_SCALE[lc])
        for lc, rc in ((0, 3), (1, 4), (2, 5))
    ]
    return float(max(normalized))


def _wingbeat_peaks(traj: np.ndarray) -> np.ndarray:
    """Returns wingbeat boundary indices as the average of left and right phi peaks."""
    left_peaks,  _ = find_peaks(traj[:, _L_PHI], distance=50)
    right_peaks, _ = find_peaks(traj[:, _R_PHI], distance=50)
    n = min(len(left_peaks), len(right_peaks))
    return ((left_peaks[:n] + right_peaks[:n]) / 2).astype(int)


def _segment_to_sa(segment: np.ndarray, template: np.ndarray) -> np.ndarray:
    """
    Converts one wingbeat segment (n, 6) to its S/A representation (n, 6):
    [S_phi, S_theta, S_psi, A_phi, A_theta, A_psi].
    """
    n = segment.shape[0]
    phase_template = np.linspace(0, 1, template.shape[0])
    phase_segment  = np.linspace(0, 1, n)
    matched = interp1d(phase_template, template, axis=0, kind='cubic')(phase_segment)
    hat = segment - matched
    S = (hat[:, :3] + hat[:, 3:]) / 2.0
    A = (hat[:, :3] - hat[:, 3:]) / 2.0
    return np.concatenate([S, A], axis=1).astype(np.float32)


def _sa_to_segment(sa: np.ndarray, template: np.ndarray) -> np.ndarray:
    """
    Inverse of _segment_to_sa: converts S/A representation (n, 6) back to
    wing angles (n, 6) [L_phi, L_theta, L_psi, R_phi, R_theta, R_psi].

    Inverse derivation:
        S = (hat_L + hat_R) / 2  →  hat_L = S + A
        A = (hat_L - hat_R) / 2  →  hat_R = S - A
        wing_angles = [hat_L, hat_R] + matched_template
    """
    n = sa.shape[0]
    phase_template = np.linspace(0, 1, template.shape[0])
    phase_segment  = np.linspace(0, 1, n)
    matched = interp1d(phase_template, template, axis=0, kind='cubic')(phase_segment)

    S = sa[:, :3]
    A = sa[:, 3:]
    hat = np.concatenate([S + A, S - A], axis=1)
    return (hat + matched).astype(np.float32)


def _segment_to_single_wing(wing: np.ndarray, template3: np.ndarray) -> np.ndarray:
    """
    Converts one single-wing segment (n, 3) [phi, theta, psi] to its residual
    against the phase-matched single-wing template: residual = wing - matched.

    `template3` is the (template_res, 3) single-wing golden template. The same
    phase-matching (cubic interp onto the segment's normalized phase grid) used
    by _segment_to_sa is applied here.
    """
    n = wing.shape[0]
    phase_template = np.linspace(0, 1, template3.shape[0])
    phase_segment  = np.linspace(0, 1, n)
    matched = interp1d(phase_template, template3, axis=0, kind='cubic')(phase_segment)
    return (wing - matched).astype(np.float32)


def _single_wing_to_segment(residual: np.ndarray, template3: np.ndarray) -> np.ndarray:
    """
    Inverse of _segment_to_single_wing: wing angles = residual + matched template.
    `residual` is (n, 3); returns (n, 3) [phi, theta, psi].
    """
    n = residual.shape[0]
    phase_template = np.linspace(0, 1, template3.shape[0])
    phase_segment  = np.linspace(0, 1, n)
    matched = interp1d(phase_template, template3, axis=0, kind='cubic')(phase_segment)
    return (residual + matched).astype(np.float32)


def verify_sa_transform(
    trajectories: list,
    template: np.ndarray,
    save_path: str = "data/analysis/sa_transform_verification.png",
    seed: int | None = None,
) -> None:
    """
    Picks one random wingbeat, round-trips it through _segment_to_sa → _sa_to_segment,
    and plots original, reconstruction, and golden template for visual verification.
    """
    rng = np.random.default_rng(seed)

    # Collect all valid wingbeats across trajectories
    candidates = []
    for traj in trajectories:
        peaks = _wingbeat_peaks(traj)
        for i in range(len(peaks) - 1):
            candidates.append((traj, peaks[i], peaks[i + 1]))

    if not candidates:
        raise ValueError("No valid wingbeats found in any trajectory.")

    traj, start, end = candidates[rng.integers(len(candidates))]
    original = traj[start:end]  # (n, 6)

    # Round-trip: forward → inverse
    sa            = _segment_to_sa(original, template)
    reconstruction = _sa_to_segment(sa, template)

    # Interpolate template onto the same phase grid as the segment
    n              = original.shape[0]
    phase_seg      = np.linspace(0, 1, n)
    phase_template = np.linspace(0, 1, template.shape[0])
    matched        = interp1d(phase_template, template, axis=0, kind='cubic')(phase_seg)

    angle_labels = ['Stroke φ [rad]', 'Deviation θ [rad]', 'Rotation ψ [rad]']
    left_cols    = [0, 1, 2]
    right_cols   = [3, 4, 5]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("S/A Transform Verification — Original vs Reconstruction vs Template", fontsize=14)

    for ax, label, lc, rc in zip(axes, angle_labels, left_cols, right_cols):
        ax.plot(phase_seg, original[:, lc],       color='blue', lw=2,   ls='-',  label='Left — original')
        ax.plot(phase_seg, reconstruction[:, lc], color='blue', lw=1.5, ls='--', label='Left — reconstruction')
        ax.plot(phase_seg, matched[:, lc],         color='blue', lw=1,   ls=':',  alpha=0.5, label='Left — template')

        ax.plot(phase_seg, original[:, rc],       color='red',  lw=2,   ls='-',  label='Right — original')
        ax.plot(phase_seg, reconstruction[:, rc], color='red',  lw=1.5, ls='--', label='Right — reconstruction')
        ax.plot(phase_seg, matched[:, rc],         color='red',  lw=1,   ls=':',  alpha=0.5, label='Right — template')

        ax.set_ylabel(label)
        ax.grid(True, alpha=0.4)

    axes[0].legend(loc='upper right', fontsize=8, ncol=2)
    axes[2].set_xlabel('Normalized Phase [0 — 1]')

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Verification plot saved → {save_path}")


def generate_average_wingbeat_template(trajectories, template_res=100, plot_template=True, save_path="data/analysis/golden_template.png"):
    """
    trajectories: List of (N, 6) arrays
    template_res: The resolution of our 'Golden' cycle
    plot_template: Boolean, if True, generates and displays a plot of the template
    save_path: String, path to save the plotted figure
    """
    all_cycles = []

    for traj in trajectories:
        peaks = _wingbeat_peaks(traj)

        for i in range(len(peaks) - 1):
            start, end = peaks[i], peaks[i+1]
            segment = traj[start:end, :] # Shape (varies 61-75, 6)

            # Create a relative time scale [0, 1] for this specific segment
            actual_len = segment.shape[0]
            relative_time = np.linspace(0, 1, actual_len)
            
            # Create the fixed phase grid [0, 0.01, ..., 1.0]
            phase_grid = np.linspace(0, 1, template_res)

            # Interpolate all 6 angles onto the 100-point grid
            f = interp1d(relative_time, segment, axis=0, kind='cubic')
            normalized_cycle = f(phase_grid)
            
            all_cycles.append(normalized_cycle)

    # Calculate the 'Golden' Mean and the per-phase, per-channel std across all cycles.
    # all_cycles_arr shape: (n_cycles, template_res, 6).
    all_cycles_arr = np.stack(all_cycles, axis=0)
    template = all_cycles_arr.mean(axis=0).astype(np.float32)
    template_std = all_cycles_arr.std(axis=0).astype(np.float32)  # (template_res, 6)

    # ---------------------------------------------------------
    # Plotting and Saving Logic
    # Assumes column order: [L_Stroke, L_Dev, L_Rot, R_Stroke, R_Dev, R_Rot]
    # A ±1 std band is shaded around each mean curve to show inter-cycle spread.
    # ---------------------------------------------------------
    if plot_template:
        phase = np.linspace(0, 1, template_res)
        fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        fig.suptitle(f"Normalized 'Golden' Hover Template (n_cycles={len(all_cycles)}, shaded = ±1 std)", fontsize=14)

        def _plot_pair(ax, angle_idx_l: int, angle_idx_r: int, ylabel: str, l_label: str, r_label: str):
            ax.plot(phase, template[:, angle_idx_l], label=l_label, color='blue', linewidth=2)
            ax.fill_between(
                phase,
                template[:, angle_idx_l] - template_std[:, angle_idx_l],
                template[:, angle_idx_l] + template_std[:, angle_idx_l],
                color='blue', alpha=0.2, linewidth=0,
            )
            ax.plot(phase, template[:, angle_idx_r], label=r_label, color='red', linestyle='--', linewidth=2)
            ax.fill_between(
                phase,
                template[:, angle_idx_r] - template_std[:, angle_idx_r],
                template[:, angle_idx_r] + template_std[:, angle_idx_r],
                color='red', alpha=0.2, linewidth=0,
            )
            ax.set_ylabel(ylabel)
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.5)

        _plot_pair(axes[0], 0, 3, 'Stroke [rad]',    'Left Stroke',    'Right Stroke')
        _plot_pair(axes[1], 1, 4, 'Deviation [rad]', 'Left Deviation', 'Right Deviation')
        _plot_pair(axes[2], 2, 5, 'Rotation [rad]',  'Left Rotation',  'Right Rotation')
        axes[2].set_xlabel('Normalized Phase [0.0 - 1.0]')

        plt.tight_layout()
        
        # Save the figure if a path is provided
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Template plot saved to: {save_path}")

            # 3D wing-angle-space view of the template (next to the .png/.npy). The
            # template is one representative wingbeat per wing, drawn as L/R loops.
            # A missing plotly or render error must not break the template build.
            try:
                from wingbeat_angle_space import make_angle_space_figure, write_html, WING_COLORS
                html_path = os.path.splitext(save_path)[0] + "_angle_space.html"
                fig3d = make_angle_space_figure(
                    loops=[
                        dict(name="Left wing",  angles=template[:, 0:3], units="rad",
                             color=WING_COLORS["left"],  markers=True, close=True, mark_start=True, width=5),
                        dict(name="Right wing", angles=template[:, 3:6], units="rad",
                             color=WING_COLORS["right"], markers=True, close=True, mark_start=True, width=5),
                    ],
                    title=f"Golden template — wing angle space (n_cycles={len(all_cycles)})",
                )
                write_html(fig3d, html_path)
                logger.info(f"Template angle-space 3D plot saved → {html_path}")
            except Exception as e:
                logger.warning(f"Skipped template angle-space 3D plot: {e}")

        plt.show()
    
    return template


def generate_single_wing_template(trajectories, template_res=69, plot_template=True,
                                  save_path="data/analysis/golden_template_single_wing.png"):
    """
    Build a single-wing golden template (template_res, 3) [phi, theta, psi] by pooling
    every wingbeat's LEFT wing (cols 0:3) and RIGHT wing (cols 3:6) as independent
    single-wing cycles and averaging on a normalized [0, 1] phase grid.

    Left and right wing angles are stored in the same sign convention (the 6-channel
    golden template's L/R curves nearly overlap), so the two wings are directly poolable.
    Mirrors generate_average_wingbeat_template's cycle-collection loop.
    """
    all_cycles = []
    phase_grid = np.linspace(0, 1, template_res)

    for traj in trajectories:
        peaks = _wingbeat_peaks(traj)
        for i in range(len(peaks) - 1):
            start, end = peaks[i], peaks[i + 1]
            segment = traj[start:end, :]                 # (n, 6)
            n = segment.shape[0]
            if n < 2:
                continue
            relative_time = np.linspace(0, 1, n)
            for cols in (slice(0, 3), slice(3, 6)):      # left wing, then right wing
                wing = segment[:, cols]                  # (n, 3)
                f = interp1d(relative_time, wing, axis=0, kind='cubic')
                all_cycles.append(f(phase_grid))

    all_cycles_arr = np.stack(all_cycles, axis=0)        # (2 * n_cycles, template_res, 3)
    template3 = all_cycles_arr.mean(axis=0).astype(np.float32)
    template3_std = all_cycles_arr.std(axis=0).astype(np.float32)

    if plot_template:
        phase = phase_grid
        fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        fig.suptitle(
            f"Single-Wing 'Golden' Hover Template (n_cycles={len(all_cycles)}, shaded = ±1 std)",
            fontsize=14,
        )
        labels = ['Stroke φ [rad]', 'Deviation θ [rad]', 'Rotation ψ [rad]']
        for ax, k, label in zip(axes, range(3), labels):
            ax.plot(phase, template3[:, k], color='purple', linewidth=2, label='single-wing mean')
            ax.fill_between(
                phase,
                template3[:, k] - template3_std[:, k],
                template3[:, k] + template3_std[:, k],
                color='purple', alpha=0.2, linewidth=0,
            )
            ax.set_ylabel(label)
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.5)
        axes[2].set_xlabel('Normalized Phase [0.0 - 1.0]')
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Single-wing template plot saved to: {save_path}")

            # 3D wing-angle-space view: one wing, one phase-colored loop. Guarded so a
            # missing plotly or render error never breaks the template build.
            try:
                from wingbeat_angle_space import plot_single_wingbeat
                html_path = os.path.splitext(save_path)[0] + "_angle_space.html"
                plot_single_wingbeat(
                    template3, html_path, units="rad", name="single-wing template",
                    title=f"Single-wing golden template — wing angle space (n_cycles={len(all_cycles)})",
                )
                logger.info(f"Single-wing template angle-space 3D plot saved → {html_path}")
            except Exception as e:
                logger.warning(f"Skipped single-wing template angle-space 3D plot: {e}")
        plt.close(fig)

    return template3


def transform_to_symmetric_asymmetric(trajectories, template, stroke_idx=0):
    """
    Transforms continuous wing angle trajectories into Symmetric (S) and Asymmetric (A) components.
    
    Args:
    - trajectories: List of (N, 6) arrays [L_phi, L_theta, L_psi, R_phi, R_theta, R_psi]
    - template: (100, 6) array representing the golden wingbeat template
    - stroke_idx: Index of the stroke angle to find peaks (default 0 for Left Stroke)
    
    Returns:
    - transformed_trajectories: List of (M, 6) arrays containing [S_phi, S_theta, S_psi, A_phi, A_theta, A_psi] 
                                for the valid wingbeat periods (dropping the incomplete ends).
    """
    transformed_trajectories = []
    
    for traj in trajectories:
        # Find peaks to define complete wingbeats
        peaks, _ = find_peaks(traj[:, stroke_idx], distance=50)
        
        if len(peaks) < 2:
            continue
            
        valid_start, valid_end = peaks[0], peaks[-1]
        valid_length = valid_end - valid_start
        transformed_traj = np.zeros((valid_length, 6), dtype=np.float32)
        
        for i in range(len(peaks) - 1):
            start = peaks[i]
            end = peaks[i+1]
            out_start = start - valid_start
            out_end   = end   - valid_start
            transformed_traj[out_start:out_end] = _segment_to_sa(traj[start:end], template)
            
        transformed_trajectories.append(transformed_traj)

    return transformed_trajectories


def _load_wing_and_body_trajectories(
    processed_dir: str, use_radians: bool = True,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Returns (wing_trajectories, body_trajectories) — one (T, 6) wing-angle and
    one (T, 12) body-kinematic array per processed H5 file, in the same order.

    Body columns from _extract_features_and_targets are [v(3), a(3), ω(3), α(3)].
    """
    # Gather condensed files from all experiment subfolders (recursive), sorted by
    # path so the trajectory order matches build_regressor_dataset.py.
    paths = gather_condensed_h5(processed_dir)
    if not paths:
        raise FileNotFoundError(f"No .h5 files found under {processed_dir}")

    wing_trajectories: list[np.ndarray] = []
    body_trajectories: list[np.ndarray] = []
    for path in paths:
        body_matrix, wing_matrix = _extract_features_and_targets(
            path,
            forces_indication_vector=None,  # None → keep all 12 body columns
            use_radians=use_radians,
        )
        wing_trajectories.append(wing_matrix)
        body_trajectories.append(body_matrix.astype(np.float32))
        logger.info(f"  {os.path.relpath(path, processed_dir)}: wing={wing_matrix.shape} body={body_matrix.shape}")

    return wing_trajectories, body_trajectories


# ---------------------------------------------------------------------------
# Fixed-length wingbeat dataset (for the new architecture where the autoencoder
# always sees length L).
# ---------------------------------------------------------------------------


_FIXED_LEN_INTERP_METHOD = "cubic_spline"   # what we use to resample SA wingbeats to L
_FIXED_LEN_SCHEMA_VERSION = 4                # bump when on-disk schema changes
                                              # v2: added body_means + maneuver_scores arrays
                                              #     and matching sidecar fields
                                              # v3: maneuver_scores narrowed from 6 channels
                                              #     to 3 (angular acceleration only; linear
                                              #     acceleration excluded until gravity is
                                              #     subtracted upstream)
                                              # v4: single-wing dataset adds
                                              #     body_omega_endpoints (2N, 3, 2): ω at the
                                              #     first and last interpolated sample of each
                                              #     wingbeat (for within-beat Δω proxies)

# Body kinematics column layout from _extract_features_and_targets:
# [v(3), a(3), ω(3), α(3)]. _BODY_A_COLS is kept here as documentation; linear
# acceleration is not currently fed into maneuver scoring.
_BODY_A_COLS     = slice(3, 6)    # linear acceleration in body frame (unused, contains gravity)
_BODY_OMEGA_COLS = slice(6, 9)    # angular velocity in body frame (yaw, pitch, roll)
_BODY_ALPHA_COLS = slice(9, 12)   # angular acceleration in body frame


def _file_md5(path: str) -> str:
    """Stable file fingerprint for sidecar drift detection. Empty string if missing."""
    if not path or not os.path.exists(path):
        return ""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _cubic_resample(arr: np.ndarray, L: int) -> np.ndarray:
    """Resample a (n, C) signal to (L, C) along axis 0 via CubicSpline."""
    n = arr.shape[0]
    if n == L:
        return arr.astype(np.float32, copy=False)
    x_old = np.linspace(0.0, 1.0, n)
    x_new = np.linspace(0.0, 1.0, L)
    cs    = CubicSpline(x_old, arr, axis=0)
    return cs(x_new).astype(np.float32)


def _repr_tag(representation: str) -> str:
    """Filename infix distinguishing representation variants. 'sa' (default) is empty
    so existing wingbeats_L<L>.npz paths are unchanged; 'single_wing' tags the file."""
    if representation == "sa":
        return ""
    if representation == "single_wing":
        return "single_wing_"
    raise ValueError(f"Unknown representation {representation!r}. Options: sa, single_wing.")


def fixed_len_dataset_path(data_dir: str, L: int, representation: str = "sa") -> str:
    return os.path.join(data_dir, f"wingbeats_{_repr_tag(representation)}L{L}.npz")


def fixed_len_sidecar_path(data_dir: str, L: int, representation: str = "sa") -> str:
    return os.path.join(data_dir, f"wingbeats_{_repr_tag(representation)}L{L}.json")


def single_wing_template_path(template_path: str) -> str:
    """Sibling path of the 6-ch golden template holding the single-wing (3-ch) template,
    e.g. data/analysis/golden_template.npy → data/analysis/golden_template_single_wing.npy."""
    root, ext = os.path.splitext(template_path)
    return f"{root}_single_wing{ext or '.npy'}"


def body_kinematics_path(data_dir: str) -> str:
    """Per-trajectory body kinematics live alongside trajectories.npy. Same indexing."""
    return os.path.join(data_dir, "body_kinematics.npy")


def build_fixed_len_dataset(
    L: int,
    trajectories: list,
    body_trajectories: list,
    template: np.ndarray,
    output_path: str,
    *,
    trajectories_path: str = "",
    body_kinematics_path_str: str = "",
    template_path: str = "",
    use_radians: bool = True,
    asymmetry_max_multiple: float | None = None,
    maneuver_W: int = 4,
    maneuver_T_coh: float = 0.75,
    maneuver_T_mag_percentile: float = 75.0,
) -> dict:
    """
    Build the fixed-length wingbeat dataset and its sidecar JSON.

    Each wingbeat (between consecutive peaks) is SA-transformed against the
    golden template at native length, normalized by SA_PHYSICAL_SCALE, and then
    CubicSpline-resampled along the time axis to length L. The result is stored
    channels-first so it can be fed straight into the encoder as (B, 6, L).

    The corresponding slice of each trajectory's body-kinematic matrix is
    averaged per wingbeat to produce a (N, 12) body_means array. The angular-
    and linear-acceleration sub-vectors of those means are then passed through
    the windowed maneuver detector (data_handling/maneuver_scoring.py) to emit
    a (N, 6) graded score per wingbeat — one channel per body axis. This score
    is purely body-kinematic; wing angles never influence it.

    Atomic write: a .tmp file is created and renamed in place to avoid leaving
    a half-written .npz on disk if the process is killed.

    Returns the sidecar metadata dict (also written to disk alongside .npz).
    """
    if len(body_trajectories) != len(trajectories):
        raise ValueError(
            f"body/wing trajectory count mismatch: {len(body_trajectories)} vs {len(trajectories)}"
        )

    sa_wingbeats:   list[np.ndarray] = []
    body_means:     list[np.ndarray] = []
    durations:      list[int]        = []
    trajectory_ids: list[int]        = []
    n_skipped = 0

    for traj_id, (traj, body) in enumerate(zip(trajectories, body_trajectories)):
        if body.shape[0] != traj.shape[0]:
            raise ValueError(
                f"trajectory {traj_id}: body length {body.shape[0]} ≠ wing length {traj.shape[0]}"
            )
        peaks = _wingbeat_peaks(traj)
        for i in range(len(peaks) - 1):
            start, end = int(peaks[i]), int(peaks[i + 1])
            n = end - start
            if n <= 1:
                n_skipped += 1
                continue
            sa_native = _segment_to_sa(traj[start:end], template)       # (n, 6)
            sa_L      = _cubic_resample(sa_native, L)                   # (L, 6)
            sa_L_norm = sa_L / SA_PHYSICAL_SCALE                        # broadcast (6,) → (L, 6)
            sa_wingbeats.append(sa_L_norm.T.astype(np.float32))         # (6, L)
            # Per-wingbeat mean of the 12-d body-kinematic vector. Mean-of-vectors
            # (not mean-of-magnitudes) — preserves direction so the across-wingbeat
            # sign-consistency check in the maneuver detector is meaningful.
            body_means.append(body[start:end].mean(axis=0).astype(np.float32))
            durations.append(n)
            trajectory_ids.append(traj_id)

    if not sa_wingbeats:
        raise RuntimeError("No wingbeats produced — check trajectories input.")

    sa_arr   = np.stack(sa_wingbeats)                                    # (N, 6, L)
    bm_arr   = np.stack(body_means)                                      # (N, 12)
    dur_arr  = np.asarray(durations,      dtype=np.int32)
    tid_arr  = np.asarray(trajectory_ids, dtype=np.int32)

    # --- Maneuver scoring (angular acceleration only; see maneuver_scoring.py) ---
    from data_handling.maneuver_scoring import compute_maneuver_scores  # local import; no cycle
    maneuver_scores, maneuver_meta = compute_maneuver_scores(
        mean_alpha       = bm_arr[:, _BODY_ALPHA_COLS],
        trajectory_ids   = tid_arr,
        W                = maneuver_W,
        T_coh            = maneuver_T_coh,
        T_mag_percentile = maneuver_T_mag_percentile,
    )

    # --- Sidecar metadata (for drift detection in consumers) ---
    sidecar = {
        "schema_version":         _FIXED_LEN_SCHEMA_VERSION,
        "L":                      int(L),
        "interpolation":          _FIXED_LEN_INTERP_METHOD,
        "sa_physical_scale":      SA_PHYSICAL_SCALE.tolist(),
        "use_radians":            bool(use_radians),
        "asymmetry_max_multiple": asymmetry_max_multiple,
        "n_wingbeats":            int(sa_arr.shape[0]),
        "n_trajectories":         int(len(trajectories)),
        "duration_min":           int(dur_arr.min()),
        "duration_max":           int(dur_arr.max()),
        "duration_mean":          float(dur_arr.mean()),
        "duration_std":           float(dur_arr.std()),
        "n_skipped":              int(n_skipped),
        "trajectories_path":      trajectories_path,
        "trajectories_md5":       _file_md5(trajectories_path),
        "body_kinematics_path":   body_kinematics_path_str,
        "body_kinematics_md5":    _file_md5(body_kinematics_path_str),
        "template_path":          template_path,
        "template_md5":           _file_md5(template_path),
        "built_at":               datetime.now().isoformat(timespec="seconds"),
        **maneuver_meta,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Atomic write: temp file in the same directory, then rename.
    # np.savez auto-appends .npz if the path doesn't already end in it, so the temp
    # path needs to end in .npz too — otherwise the file we write and the file we
    # try to rename are different paths.
    tmp_path = output_path[:-len(".npz")] + ".tmp.npz" if output_path.endswith(".npz") else output_path + ".tmp.npz"
    np.savez(
        tmp_path,
        sa_wingbeats    = sa_arr,
        body_means      = bm_arr,
        maneuver_scores = maneuver_scores,
        durations       = dur_arr,
        trajectory_ids  = tid_arr,
    )
    os.replace(tmp_path, output_path)

    sidecar_path = os.path.splitext(output_path)[0] + ".json"
    tmp_sidecar  = sidecar_path + ".tmp"
    with open(tmp_sidecar, "w") as f:
        json.dump(sidecar, f, indent=2)
    os.replace(tmp_sidecar, sidecar_path)

    logger.info(
        f"Fixed-length dataset (L={L}): {sa_arr.shape[0]} wingbeats from "
        f"{len(trajectories)} trajectories → {output_path}"
    )
    if n_skipped:
        logger.warning(f"  Skipped {n_skipped} wingbeats with non-positive duration.")
    return sidecar


def build_single_wing_fixed_len_dataset(
    L: int,
    trajectories: list,
    body_trajectories: list,
    template3: np.ndarray,
    output_path: str,
    *,
    trajectories_path: str = "",
    body_kinematics_path_str: str = "",
    template_path: str = "",
    use_radians: bool = True,
    asymmetry_max_multiple: float | None = None,
    maneuver_W: int = 4,
    maneuver_T_coh: float = 0.75,
    maneuver_T_mag_percentile: float = 75.0,
) -> dict:
    """
    Build the single-wing fixed-length dataset and its sidecar JSON.

    Every wingbeat contributes TWO samples — its LEFT wing (cols 0:3) and its
    RIGHT wing (cols 3:6). Each single-wing segment is residual-transformed
    against the single-wing template at native length, normalized by
    SINGLE_WING_PHYSICAL_SCALE, CubicSpline-resampled to length L, and stored
    channels-first as (3, L) for the 3-channel encoder. So N wingbeats yield 2N
    single-wing samples.

    body_means and maneuver_scores are computed once per wingbeat (the body state
    is shared by both wings) and duplicated across the left/right rows so every
    array is aligned at the 2N sample level. Atomic write, same as the 6-ch builder.

    Also stores body_omega_endpoints (2N, 3, 2): the body-frame angular velocity ω
    (yaw, pitch, roll) at the first and last sample of each wingbeat *after* the same
    CubicSpline resampling to L applied to the wing angles — i.e. read off the
    interpolated n-samples-per-wingbeat grid, not the raw native samples. Lets a
    consumer form the within-beat change Δω = ω_last − ω_first per axis.
    """
    if len(body_trajectories) != len(trajectories):
        raise ValueError(
            f"body/wing trajectory count mismatch: {len(body_trajectories)} vs {len(trajectories)}"
        )

    wing_samples:    list[np.ndarray] = []   # (3, L) per single wing
    wing_side:       list[int]        = []   # 0 = left, 1 = right
    body_means_wb:   list[np.ndarray] = []   # (12,) per wingbeat (N-level)
    omega_ep_wb:     list[np.ndarray] = []   # (3, 2) per wingbeat: ω at [first, last] interp sample
    durations_wb:    list[int]        = []   # N-level
    trajectory_wb:   list[int]        = []   # N-level
    n_skipped = 0

    for traj_id, (traj, body) in enumerate(zip(trajectories, body_trajectories)):
        if body.shape[0] != traj.shape[0]:
            raise ValueError(
                f"trajectory {traj_id}: body length {body.shape[0]} ≠ wing length {traj.shape[0]}"
            )
        peaks = _wingbeat_peaks(traj)
        for i in range(len(peaks) - 1):
            start, end = int(peaks[i]), int(peaks[i + 1])
            n = end - start
            if n <= 1:
                n_skipped += 1
                continue
            segment = traj[start:end]                                   # (n, 6)
            for side, cols in enumerate((slice(0, 3), slice(3, 6))):    # left, then right
                res_native = _segment_to_single_wing(segment[:, cols], template3)   # (n, 3)
                res_L      = _cubic_resample(res_native, L)                          # (L, 3)
                res_L_norm = res_L / SINGLE_WING_PHYSICAL_SCALE                      # (L, 3)
                wing_samples.append(res_L_norm.T.astype(np.float32))                # (3, L)
                wing_side.append(side)
            body_means_wb.append(body[start:end].mean(axis=0).astype(np.float32))
            # Angular velocity ω resampled to L exactly like the wing angles, so its first/last
            # sample come from the same interpolated n-samples-per-wingbeat grid (not the raw
            # native samples). CubicSpline preserves endpoints, so these equal the segment ends.
            omega_L = _cubic_resample(body[start:end, _BODY_OMEGA_COLS], L)          # (L, 3)
            omega_ep_wb.append(np.stack([omega_L[0], omega_L[-1]], axis=1).astype(np.float32))  # (3, 2)
            durations_wb.append(n)
            trajectory_wb.append(traj_id)

    if not wing_samples:
        raise RuntimeError("No wingbeats produced — check trajectories input.")

    # 2N sample-level arrays. Each wingbeat appended left then right, so np.repeat(·, 2)
    # of the N-level arrays aligns body/duration/trajectory/maneuver with the L/R rows.
    sw_arr   = np.stack(wing_samples)                                    # (2N, 3, L)
    side_arr = np.asarray(wing_side, dtype=np.int8)                     # (2N,)
    bm_wb    = np.stack(body_means_wb)                                   # (N, 12)
    dur_wb   = np.asarray(durations_wb,  dtype=np.int32)               # (N,)
    tid_wb   = np.asarray(trajectory_wb, dtype=np.int32)               # (N,)

    oep_wb   = np.stack(omega_ep_wb)                                     # (N, 3, 2)

    bm_arr   = np.repeat(bm_wb,  2, axis=0)                              # (2N, 12)
    oep_arr  = np.repeat(oep_wb, 2, axis=0)                              # (2N, 3, 2)
    dur_arr  = np.repeat(dur_wb, 2, axis=0)                              # (2N,)
    tid_arr  = np.repeat(tid_wb, 2, axis=0)                              # (2N,)

    # --- Maneuver scoring at the wingbeat level, then duplicated per wing ---
    from data_handling.maneuver_scoring import compute_maneuver_scores  # local import; no cycle
    maneuver_scores_wb, maneuver_meta = compute_maneuver_scores(
        mean_alpha       = bm_wb[:, _BODY_ALPHA_COLS],
        trajectory_ids   = tid_wb,
        W                = maneuver_W,
        T_coh            = maneuver_T_coh,
        T_mag_percentile = maneuver_T_mag_percentile,
    )
    maneuver_scores = np.repeat(maneuver_scores_wb, 2, axis=0)

    sidecar = {
        "schema_version":         _FIXED_LEN_SCHEMA_VERSION,
        "representation":         "single_wing",
        "L":                      int(L),
        "interpolation":          _FIXED_LEN_INTERP_METHOD,
        "single_wing_physical_scale": SINGLE_WING_PHYSICAL_SCALE.tolist(),
        "use_radians":            bool(use_radians),
        "asymmetry_max_multiple": asymmetry_max_multiple,
        "n_samples":              int(sw_arr.shape[0]),
        "n_wingbeats":            int(bm_wb.shape[0]),
        "n_trajectories":         int(len(trajectories)),
        "duration_min":           int(dur_wb.min()),
        "duration_max":           int(dur_wb.max()),
        "duration_mean":          float(dur_wb.mean()),
        "duration_std":           float(dur_wb.std()),
        "n_skipped":              int(n_skipped),
        "trajectories_path":      trajectories_path,
        "trajectories_md5":       _file_md5(trajectories_path),
        "body_kinematics_path":   body_kinematics_path_str,
        "body_kinematics_md5":    _file_md5(body_kinematics_path_str),
        "template_path":          template_path,
        "template_md5":           _file_md5(template_path),
        "built_at":               datetime.now().isoformat(timespec="seconds"),
        **maneuver_meta,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    tmp_path = output_path[:-len(".npz")] + ".tmp.npz" if output_path.endswith(".npz") else output_path + ".tmp.npz"
    np.savez(
        tmp_path,
        single_wing_wingbeats = sw_arr,
        wing_side             = side_arr,
        body_means            = bm_arr,
        body_omega_endpoints  = oep_arr,
        maneuver_scores       = maneuver_scores,
        durations             = dur_arr,
        trajectory_ids        = tid_arr,
    )
    os.replace(tmp_path, output_path)

    sidecar_path = os.path.splitext(output_path)[0] + ".json"
    tmp_sidecar  = sidecar_path + ".tmp"
    with open(tmp_sidecar, "w") as f:
        json.dump(sidecar, f, indent=2)
    os.replace(tmp_sidecar, sidecar_path)

    logger.info(
        f"Single-wing fixed-length dataset (L={L}): {sw_arr.shape[0]} single wings from "
        f"{bm_wb.shape[0]} wingbeats / {len(trajectories)} trajectories → {output_path}"
    )
    if n_skipped:
        logger.warning(f"  Skipped {n_skipped} wingbeats with non-positive duration.")
    return sidecar


def ensure_single_wing_template(
    template_path: str,
    trajectories: list,
    template_res: int = 69,
) -> np.ndarray:
    """Load the single-wing template sibling of `template_path`, building it from
    `trajectories` (and saving it next to the 6-ch template) if it doesn't exist."""
    sw_path = single_wing_template_path(template_path)
    if os.path.exists(sw_path):
        return np.load(sw_path)
    logger.info(f"Single-wing template missing — building → {sw_path}")
    plot_path = os.path.splitext(sw_path)[0] + ".png"
    template3 = generate_single_wing_template(
        trajectories, template_res=template_res, plot_template=True, save_path=plot_path,
    )
    os.makedirs(os.path.dirname(os.path.abspath(sw_path)), exist_ok=True)
    np.save(sw_path, template3)
    return template3


def build_fixed_len_dataset_from_disk(
    L: int,
    trajectories_path: str,
    template_path: str,
    output_path: str,
    *,
    representation: str = "sa",
    body_kinematics_path_str: str | None = None,
    use_radians: bool = True,
    asymmetry_max_multiple: float | None = None,
    maneuver_W: int = 4,
    maneuver_T_coh: float = 0.75,
    maneuver_T_mag_percentile: float = 75.0,
) -> dict:
    """
    Convenience wrapper that loads trajectories + body kinematics + template
    from disk and builds the fixed-length dataset for the given representation.

    representation="sa" builds the 6-channel S/A dataset against `template_path`.
    representation="single_wing" builds the 3-channel single-wing dataset against
    the single-wing template sibling of `template_path` (built on demand if missing).

    If body_kinematics_path_str is None, the loader derives it from
    trajectories_path by convention (body_kinematics.npy in the same directory).
    """
    if body_kinematics_path_str is None:
        body_kinematics_path_str = body_kinematics_path(
            os.path.dirname(os.path.abspath(trajectories_path))
        )
    if not os.path.exists(body_kinematics_path_str):
        raise FileNotFoundError(
            f"body kinematics file not found: {body_kinematics_path_str}\n"
            f"This file is required for maneuver scoring in the fixed-length dataset. "
            f"It is produced by transform_data.py alongside trajectories.npy. "
            f"Run:  python code/transform_data.py --fixed_len {L}"
        )
    trajectories      = np.load(trajectories_path,        allow_pickle=True).tolist()
    body_trajectories = np.load(body_kinematics_path_str, allow_pickle=True).tolist()

    if representation == "single_wing":
        template3       = ensure_single_wing_template(template_path, trajectories)
        sw_template_path = single_wing_template_path(template_path)
        return build_single_wing_fixed_len_dataset(
            L                          = L,
            trajectories               = trajectories,
            body_trajectories          = body_trajectories,
            template3                  = template3,
            output_path                = output_path,
            trajectories_path          = trajectories_path,
            body_kinematics_path_str   = body_kinematics_path_str,
            template_path              = sw_template_path,
            use_radians                = use_radians,
            asymmetry_max_multiple     = asymmetry_max_multiple,
            maneuver_W                 = maneuver_W,
            maneuver_T_coh             = maneuver_T_coh,
            maneuver_T_mag_percentile  = maneuver_T_mag_percentile,
        )

    template = np.load(template_path)
    return build_fixed_len_dataset(
        L                          = L,
        trajectories               = trajectories,
        body_trajectories          = body_trajectories,
        template                   = template,
        output_path                = output_path,
        trajectories_path          = trajectories_path,
        body_kinematics_path_str   = body_kinematics_path_str,
        template_path              = template_path,
        use_radians                = use_radians,
        asymmetry_max_multiple     = asymmetry_max_multiple,
        maneuver_W                 = maneuver_W,
        maneuver_T_coh             = maneuver_T_coh,
        maneuver_T_mag_percentile  = maneuver_T_mag_percentile,
    )


def fixed_len_dataset_is_valid(
    output_path: str,
    sidecar_path: str,
    *,
    L: int,
    trajectories_path: str,
    template_path: str,
    representation: str = "sa",
    body_kinematics_path_str: str | None = None,
    maneuver_W: int | None = None,
    maneuver_T_coh: float | None = None,
    maneuver_T_mag_percentile: float | None = None,
) -> tuple[bool, str]:
    """
    Returns (is_valid, reason). is_valid=True means the on-disk dataset can be loaded
    as-is; False means it's missing or stale and the caller should rebuild.

    `template_path` is the 6-ch golden template path; for representation="single_wing"
    the single-wing template sibling is the one whose md5 is checked.

    Maneuver params (W / T_coh / T_mag_percentile) are checked against the sidecar
    only when the caller passes a non-None value. This lets autoencoder configs
    that don't care about maneuver scoring stay valid against any sidecar settings.
    """
    if not os.path.exists(output_path):
        return False, f"missing dataset {output_path}"
    if not os.path.exists(sidecar_path):
        return False, f"missing sidecar {sidecar_path}"
    try:
        with open(sidecar_path) as f:
            meta = json.load(f)
    except json.JSONDecodeError as e:
        return False, f"corrupt sidecar: {e}"
    if meta.get("schema_version") != _FIXED_LEN_SCHEMA_VERSION:
        return False, f"schema_version mismatch ({meta.get('schema_version')} ≠ {_FIXED_LEN_SCHEMA_VERSION})"
    if meta.get("representation", "sa") != representation:
        return False, f"representation mismatch ({meta.get('representation', 'sa')} ≠ {representation})"
    if int(meta.get("L", -1)) != int(L):
        return False, f"L mismatch ({meta.get('L')} ≠ {L})"
    template_md5_path = (
        single_wing_template_path(template_path) if representation == "single_wing" else template_path
    )
    if meta.get("template_md5") != _file_md5(template_md5_path):
        return False, "template file changed since build"
    if meta.get("trajectories_md5") != _file_md5(trajectories_path):
        return False, "trajectories file changed since build"

    if body_kinematics_path_str is None:
        body_kinematics_path_str = body_kinematics_path(
            os.path.dirname(os.path.abspath(trajectories_path))
        )
    if not os.path.exists(body_kinematics_path_str):
        return False, f"missing body kinematics file {body_kinematics_path_str}"
    if meta.get("body_kinematics_md5") != _file_md5(body_kinematics_path_str):
        return False, "body kinematics file changed since build"

    if maneuver_W is not None and int(meta.get("maneuver_W", -1)) != int(maneuver_W):
        return False, f"maneuver_W mismatch ({meta.get('maneuver_W')} ≠ {maneuver_W})"
    if maneuver_T_coh is not None and float(meta.get("maneuver_T_coh", -1)) != float(maneuver_T_coh):
        return False, f"maneuver_T_coh mismatch ({meta.get('maneuver_T_coh')} ≠ {maneuver_T_coh})"
    if maneuver_T_mag_percentile is not None and float(meta.get("maneuver_T_mag_percentile", -1)) != float(maneuver_T_mag_percentile):
        return False, f"maneuver_T_mag_percentile mismatch ({meta.get('maneuver_T_mag_percentile')} ≠ {maneuver_T_mag_percentile})"

    return True, "ok"


def main() -> None:
    """
    Loads wing trajectories from processed H5 files, generates the golden wingbeat
    template, and saves both to the paths specified in the autoencoder config.

    Run from the project root:
        python code/transform_data.py --config code/autoencoder_config.json
    """
    parser = argparse.ArgumentParser(description="Generate golden wingbeat template for autoencoder training.")
    parser.add_argument(
        "--config",
        default="code/autoencoder_config.json",
        help="Path to autoencoder_config.json (provides data_path, template_path, stroke_idx)",
    )
    parser.add_argument(
        "--processed_dir",
        type=str,
        default=None,
        help="Directory of condensed H5 files to load trajectories from. Default: the shared "
             "train_processed_data. Point this at an isolated dir to build trajectories.npy + "
             "template for a data subset (e.g. a new-data-only AE workspace).",
    )
    parser.add_argument(
        "--template_res",
        type=int,
        default=69,
        help="Number of phase points in the golden template (default: 100)",
    )
    parser.add_argument(
        "--no_radians",
        action="store_true",
        help="Keep wing angles in degrees instead of converting to radians",
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="Skip saving the template plot",
    )
    parser.add_argument(
        "--asymmetry_max_multiple",
        type=float,
        default=10.0,
        help="Drop trajectories whose L/R asymmetry score exceeds this multiple of the "
             "dataset median. Set to 0 to disable garbage filtering. Default: 10.0.",
    )
    parser.add_argument(
        "--fixed_len",
        type=int,
        default=0,
        help="If > 0, additionally build wingbeats_L<fixed_len>.npz — every wingbeat's "
             "SA representation CubicSpline-resampled to this many samples. The variable-length "
             "trajectories.npy and template are still produced. Default: 0 (no fixed-length file).",
    )
    parser.add_argument(
        "--single_wing_fixed_len",
        type=int,
        default=0,
        help="If > 0, additionally build the single-wing dataset wingbeats_single_wing_L<N>.npz — "
             "every wingbeat split into its left and right wing (3 channels each), residual against "
             "the single-wing template, CubicSpline-resampled to this many samples. Also builds "
             "golden_template_single_wing.npy. Default: 0 (not built).",
    )
    parser.add_argument(
        "--maneuver_W", type=int, default=4,
        help="Sliding-window size (wingbeats) for maneuver detection inside the fixed-length build. Default: 4.",
    )
    parser.add_argument(
        "--maneuver_T_coh", type=float, default=0.75,
        help="Sign-consistency threshold for the maneuver detector, in [0, 1]. Default: 0.75.",
    )
    parser.add_argument(
        "--maneuver_T_mag_percentile", type=float, default=75.0,
        help="Per-channel magnitude threshold expressed as a percentile of |x^k| across the corpus. Default: 75.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    # List-valued keys are grid-search params — read just the scalar value for paths/indices
    def scalar(v):
        return v[0] if isinstance(v, list) else v

    data_path     = scalar(config['data_path'])
    template_path = scalar(config['template_path'])

    os.makedirs(os.path.dirname(os.path.abspath(data_path)),     exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(template_path)), exist_ok=True)

    # --- Load wing + body trajectories from processed H5 files ---
    # Precedence: explicit --processed_dir > config "processed_dir" key > global default.
    # The config key lets a dataset config (e.g. autoencoder_dataset) self-describe its
    # condensed source, so a bare `transform_data --config <cfg>` rebuilds the right
    # dataset instead of silently falling back to the shared train_processed_data.
    processed_dir = args.processed_dir or config.get('processed_dir') or PROCESSED_DATA_DIR
    logger.info(f"Loading trajectories from {processed_dir} ...")
    trajectories, body_trajectories = _load_wing_and_body_trajectories(
        processed_dir, use_radians=not args.no_radians
    )
    logger.info(f"Loaded {len(trajectories)} trajectories (wing + body).")

    # --- Drop garbage trajectories by L/R asymmetry score ---
    # Asymmetry is a wing-angle property, but the resulting keep_mask is applied to both
    # the wing and body arrays so their indexing stays in lockstep through everything downstream.
    if args.asymmetry_max_multiple > 0 and len(trajectories) > 0:
        scores = np.array([trajectory_asymmetry_score(t) for t in trajectories], dtype=np.float64)
        median_score = float(np.median(scores))
        if median_score > 0:
            threshold = args.asymmetry_max_multiple * median_score
            keep_mask = scores <= threshold
            n_dropped = int((~keep_mask).sum())
            if n_dropped > 0:
                dropped = [(i, float(scores[i])) for i in range(len(scores)) if not keep_mask[i]]
                logger.info(
                    f"Asymmetry filter: dropping {n_dropped}/{len(trajectories)} trajectories "
                    f"with score > {args.asymmetry_max_multiple}× median = {threshold:.4f} "
                    f"(median = {median_score:.4f})."
                )
                logger.info("  Dropped idx → score: " + ", ".join(f"{i}→{s:.3f}" for i, s in dropped))
                trajectories      = [t for t, keep in zip(trajectories,      keep_mask) if keep]
                body_trajectories = [b for b, keep in zip(body_trajectories, keep_mask) if keep]
            else:
                logger.info(
                    f"Asymmetry filter: no trajectories exceed {args.asymmetry_max_multiple}× median "
                    f"({threshold:.4f}); nothing dropped."
                )

    # Save as object arrays so variable-length per-trajectory arrays survive np.load(allow_pickle=True).
    np.save(data_path, np.array(trajectories, dtype=object))
    logger.info(f"Saved {len(trajectories)} trajectories → {data_path}")

    body_path = body_kinematics_path(os.path.dirname(os.path.abspath(data_path)))
    np.save(body_path, np.array(body_trajectories, dtype=object))
    logger.info(f"Saved body kinematics ({len(body_trajectories)} arrays) → {body_path}")

    # --- Generate golden template and save both the plot and the .npy ---
    plot_path = os.path.splitext(template_path)[0] + ".png"
    template = generate_average_wingbeat_template(
        trajectories  = trajectories,
        template_res  = args.template_res,
        plot_template = not args.no_plot,
        save_path     = plot_path if not args.no_plot else None,
    )

    np.save(template_path, template)
    logger.info(f"Saved golden template {template.shape} → {template_path}")

    # --- Verify round-trip correctness ---
    verify_path = os.path.join(os.path.dirname(template_path), "sa_transform_verification.png")
    verify_sa_transform(trajectories, template, save_path=verify_path)

    # --- Optionally build the fixed-length wingbeat dataset for the new AE architecture ---
    if args.fixed_len and args.fixed_len > 0:
        L = int(args.fixed_len)
        data_dir = os.path.dirname(os.path.abspath(data_path))
        out_path = fixed_len_dataset_path(data_dir, L)
        logger.info(f"Building fixed-length dataset (L={L}) → {out_path}")
        build_fixed_len_dataset(
            L                          = L,
            trajectories               = trajectories,
            body_trajectories          = body_trajectories,
            template                   = template,
            output_path                = out_path,
            trajectories_path          = data_path,
            body_kinematics_path_str   = body_path,
            template_path              = template_path,
            use_radians                = not args.no_radians,
            asymmetry_max_multiple     = args.asymmetry_max_multiple,
            maneuver_W                 = args.maneuver_W,
            maneuver_T_coh             = args.maneuver_T_coh,
            maneuver_T_mag_percentile  = args.maneuver_T_mag_percentile,
        )

    # --- Optionally build the single-wing dataset (one wing at a time, 3 channels) ---
    if args.single_wing_fixed_len and args.single_wing_fixed_len > 0:
        L = int(args.single_wing_fixed_len)
        data_dir = os.path.dirname(os.path.abspath(data_path))
        sw_template_path = single_wing_template_path(template_path)
        template3 = generate_single_wing_template(
            trajectories  = trajectories,
            template_res  = args.template_res,
            plot_template = not args.no_plot,
            save_path     = (os.path.splitext(sw_template_path)[0] + ".png") if not args.no_plot else None,
        )
        np.save(sw_template_path, template3)
        logger.info(f"Saved single-wing template {template3.shape} → {sw_template_path}")

        out_path = fixed_len_dataset_path(data_dir, L, representation="single_wing")
        logger.info(f"Building single-wing fixed-length dataset (L={L}) → {out_path}")
        build_single_wing_fixed_len_dataset(
            L                          = L,
            trajectories               = trajectories,
            body_trajectories          = body_trajectories,
            template3                  = template3,
            output_path                = out_path,
            trajectories_path          = data_path,
            body_kinematics_path_str   = body_path,
            template_path              = sw_template_path,
            use_radians                = not args.no_radians,
            asymmetry_max_multiple     = args.asymmetry_max_multiple,
            maneuver_W                 = args.maneuver_W,
            maneuver_T_coh             = args.maneuver_T_coh,
            maneuver_T_mag_percentile  = args.maneuver_T_mag_percentile,
        )


if __name__ == '__main__':
    main()
