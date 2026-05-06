import argparse
import json
import os
import sys

import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from loguru import logger

# Allow imports from data_handling/ when run from the project root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_handling'))
from data_handling.process_data import PROCESSED_TRAIN_FLIGHT_DATA_DIR, _extract_features_and_targets

# Column indices in the (N, 6) wing matrix: [L_phi, L_theta, L_psi, R_phi, R_theta, R_psi]
_L_PHI = 0
_R_PHI = 3


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

    # Calculate the 'Golden' Mean
    # Resulting shape: (100, 6)
    template = np.mean(all_cycles, axis=0).astype(np.float32)
    
    # ---------------------------------------------------------
    # Plotting and Saving Logic
    # Assumes column order: [L_Stroke, L_Dev, L_Rot, R_Stroke, R_Dev, R_Rot]
    # ---------------------------------------------------------
    if plot_template:
        phase = np.linspace(0, 1, template_res)
        fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        fig.suptitle("Normalized 'Golden' Hover Template", fontsize=16)

        # 1. Stroke Angle (phi)
        axes[0].plot(phase, template[:, 0], label='Left Stroke', color='blue', linewidth=2)
        axes[0].plot(phase, template[:, 3], label='Right Stroke', color='red', linestyle='--', linewidth=2)
        axes[0].set_ylabel('Stroke [rad]')
        axes[0].legend(loc="upper right")
        axes[0].grid(True, alpha=0.5)

        # 2. Deviation Angle (theta)
        axes[1].plot(phase, template[:, 1], label='Left Deviation', color='blue', linewidth=2)
        axes[1].plot(phase, template[:, 4], label='Right Deviation', color='red', linestyle='--', linewidth=2)
        axes[1].set_ylabel('Deviation [rad]')
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.5)

        # 3. Rotation Angle (psi)
        axes[2].plot(phase, template[:, 2], label='Left Rotation', color='blue', linewidth=2)
        axes[2].plot(phase, template[:, 5], label='Right Rotation', color='red', linestyle='--', linewidth=2)
        axes[2].set_ylabel('Rotation [rad]')
        axes[2].set_xlabel('Normalized Phase [0.0 - 1.0]')
        axes[2].legend(loc="upper right")
        axes[2].grid(True, alpha=0.5)

        plt.tight_layout()
        
        # Save the figure if a path is provided
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Template plot saved to: {save_path}")
            
        plt.show()
    
    return template


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


def _load_wing_trajectories(processed_dir: str, use_radians: bool = True) -> list[np.ndarray]:
    """Returns one (N, 6) float32 wing-angle array per processed H5 file."""
    files = sorted(f for f in os.listdir(processed_dir) if f.endswith('.h5'))
    if not files:
        raise FileNotFoundError(f"No .h5 files found in {processed_dir}")

    trajectories = []
    for fname in files:
        _, wing_matrix = _extract_features_and_targets(
            os.path.join(processed_dir, fname),
            forces_indication_vector=None,  # None → skip body-column filtering
            use_radians=use_radians,
        )
        trajectories.append(wing_matrix)
        logger.info(f"  {fname}: {wing_matrix.shape}")

    return trajectories


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
        "--template_res",
        type=int,
        default=100,
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

    # --- Load wing trajectories from processed H5 files ---
    logger.info(f"Loading trajectories from {PROCESSED_TRAIN_FLIGHT_DATA_DIR} ...")
    trajectories = _load_wing_trajectories(PROCESSED_TRAIN_FLIGHT_DATA_DIR, use_radians=not args.no_radians)
    logger.info(f"Loaded {len(trajectories)} trajectories.")

    # Save as object array so variable-length (N, 6) arrays survive np.load(allow_pickle=True)
    np.save(data_path, np.array(trajectories, dtype=object))
    logger.info(f"Saved trajectories → {data_path}")

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


if __name__ == '__main__':
    main()
