import os
import glob
import argparse
import h5py
import numpy as np
from scipy.signal import savgol_filter
from loguru import logger
import shutil

SAMPLING_RATE = 16000
VELOCITY_POLYORDER = 2
ACCELERATION_POLYORDER = 3
PROCESSED_FILE_SUFFIX = "_condensed_data.h5"
UNPROCESSED_FILE_SUFFIX = "_analysis_smoothed.h5"

GENERAL_DATASET_DIR = "./data/"
UNPROCESSED_FLIGHT_DATA_DIR = os.path.join(GENERAL_DATASET_DIR, "unprocessed_data")
# Condensed output lives one subfolder per experiment under PROCESSED_DATA_DIR
# (e.g. data/processed/roni_60ms/). transform_data + build_regressor_dataset
# gather all of them recursively via gather_condensed_h5().
PROCESSED_DATA_DIR = os.path.join(GENERAL_DATASET_DIR, "processed")

class FeaturesConstants:
    CENTER_OF_MASS_RAW = 'center_of_mass_raw'
    CENTER_OF_MASS_SMOOTHED = 'center_of_mass_smoothed'
    LINEAR_VELOCITY_BODY = 'linear_velocity_body_vec'
    LINEAR_ACCEL_BODY = 'linear_accel_body_vec'
    ANGULAR_VELOCITY_BODY = 'angular_velocity_body_vec'
    ANGULAR_ACCEL_BODY = 'angular_accel_body_vec'
    WINGS_LEFT_PHI = 'wings/left/phi'
    WINGS_LEFT_THETA = 'wings/left/theta'
    WINGS_LEFT_PSI = 'wings/left/psi'
    WINGS_RIGHT_PHI = 'wings/right/phi'
    WINGS_RIGHT_THETA = 'wings/right/theta'
    WINGS_RIGHT_PSI = 'wings/right/psi'

def compute_angular_kinematics_poisson(x_body, y_body, z_body, fps=SAMPLING_RATE, body_frame=True, window_length=351, polyorder=2):
    Rs = np.stack([x_body, y_body, z_body], axis=-1)
    N = len(Rs)
    dRdt = np.zeros_like(Rs)
    for i in range(3):
        for j in range(3):
            dRdt[:, i, j] = np.gradient(Rs[:, i, j])
            
    w_x, w_y, w_z = np.zeros((3, N))
    for frame in range(N):
        A_dot = dRdt[frame]
        A = Rs[frame]
        omega_mat = A_dot @ A.T
        w_x[frame] = np.rad2deg((omega_mat[2, 1] - omega_mat[1, 2]) / 2) * fps
        w_y[frame] = np.rad2deg((omega_mat[0, 2] - omega_mat[2, 0]) / 2) * fps
        w_z[frame] = np.rad2deg((omega_mat[1, 0] - omega_mat[0, 1]) / 2) * fps

    omega_lab = np.column_stack((w_x, w_y, w_z))
    if not body_frame:
        return omega_lab, None
        
    omega_body = np.zeros_like(omega_lab)
    for frame in range(N):
        omega_body[frame, :] = Rs[frame].T @ omega_lab[frame, :]

    alpha_body = np.zeros_like(omega_body)
    dt = 1 / fps
    for i in range(3):
        alpha_body[:, i] = savgol_filter(omega_body[:, i], window_length, polyorder, deriv=1, delta=dt)
        
    return omega_body, alpha_body

def derive_signal(signal, window_length, polyorder, derivation_order, delta=1/SAMPLING_RATE):
    return savgol_filter(signal, window_length, polyorder, deriv=derivation_order, delta=delta)

def compute_linear_kinematics(pos_raw, window_length=351, polyorder=2):
    position_smoothed = np.zeros_like(pos_raw)
    v_smoothed = np.zeros_like(pos_raw)
    a_smoothed = np.zeros_like(pos_raw)
    for i in range(3):
        position_smoothed[:, i] = derive_signal(pos_raw[:, i], window_length, polyorder, derivation_order=0)
        v_smoothed[:, i] = derive_signal(pos_raw[:, i], window_length, polyorder, derivation_order=1)
        a_smoothed[:, i] = derive_signal(pos_raw[:, i], window_length, polyorder, derivation_order=2)
    return position_smoothed, v_smoothed, a_smoothed

def transform_velocity_to_body_frame(cm_dot_world, x_body, y_body, z_body):
     cm_dot_body = np.zeros_like(cm_dot_world)
     cm_dot_body[:, 0] = np.einsum('ij,ij->i', cm_dot_world, x_body)
     cm_dot_body[:, 1] = np.einsum('ij,ij->i', cm_dot_world, y_body)
     cm_dot_body[:, 2] = np.einsum('ij,ij->i', cm_dot_world, z_body)
     return cm_dot_body

def augment_dataset(unaugmented_dataset_file_path):
    augmented_file_path = unaugmented_dataset_file_path.replace('.h5', '_augmented.h5')
    
    with h5py.File(unaugmented_dataset_file_path, 'r') as src, h5py.File(augmented_file_path, 'w') as dest:
        linear_vel_body_aug = src[FeaturesConstants.LINEAR_VELOCITY_BODY][:]
        linear_vel_body_aug[:, 1] *= -1
        dest.create_dataset(FeaturesConstants.LINEAR_VELOCITY_BODY, data=linear_vel_body_aug)

        linear_accel_body_aug = src[FeaturesConstants.LINEAR_ACCEL_BODY][:]
        linear_accel_body_aug[:, 1] *= -1
        dest.create_dataset(FeaturesConstants.LINEAR_ACCEL_BODY, data=linear_accel_body_aug)

        angular_vel_body_aug = src[FeaturesConstants.ANGULAR_VELOCITY_BODY][:]
        angular_vel_body_aug[:, 0] *= -1
        angular_vel_body_aug[:, 2] *= -1
        dest.create_dataset(FeaturesConstants.ANGULAR_VELOCITY_BODY, data=angular_vel_body_aug)

        angular_accel_body_aug = src[FeaturesConstants.ANGULAR_ACCEL_BODY][:]
        angular_accel_body_aug[:, 0] *= -1
        angular_accel_body_aug[:, 2] *= -1
        dest.create_dataset(FeaturesConstants.ANGULAR_ACCEL_BODY, data=angular_accel_body_aug)

        src_wings = src['wings']
        dest_wings = dest.create_group('wings')
        for side in ['left', 'right']:
            other_side = 'right' if side == 'left' else 'left'
            src_side_data = src_wings[side]
            dest_side_group = dest_wings.create_group(other_side)
            dest_side_group.create_dataset('phi', data=src_side_data['phi'][:])
            dest_side_group.create_dataset('theta', data=src_side_data['theta'][:])
            dest_side_group.create_dataset('psi', data=src_side_data['psi'][:])

    logger.info(f"Augmentation complete. Saved at: {augmented_file_path}")

def _process_single_h5(h5_path, output_dir, window_length=351, polyorder=2):
    file_name = os.path.basename(h5_path)
    movie_name = file_name.replace(UNPROCESSED_FILE_SUFFIX, '')
    save_path = os.path.join(output_dir, f"{movie_name}{PROCESSED_FILE_SUFFIX}")

    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(h5_path, 'r') as src:
        wings_raw = {
            'left_phi': src['wings_phi_left'][:], 'left_theta': src['wings_theta_left'][:], 'left_psi': src['wings_psi_left'][:],
            'right_phi': src['wings_phi_right'][:], 'right_theta': src['wings_theta_right'][:], 'right_psi': src['wings_psi_right'][:]
        }
        stacked_wings = np.vstack(list(wings_raw.values()))
        valid_mask = ~np.isnan(stacked_wings).any(axis=0)
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            logger.warning(f"Skipping {file_name}: No valid wing frames found.")
            return None

        start_idx, end_idx = valid_indices[0], valid_indices[-1] + 1
        sequence_length = end_idx - start_idx
        if sequence_length < window_length:
            logger.warning(f"Skipping {file_name}: Sequence length ({sequence_length}) is shorter than the filter window length ({window_length}).")
            return None

        with h5py.File(save_path, 'w') as dest:
        
            cm_raw = src['center_of_mass'][start_idx:end_idx]
            xb, yb, zb = src['x_body'][start_idx:end_idx], src['y_body'][start_idx:end_idx], src['z_body'][start_idx:end_idx]
            
            omega_body, alpha_body = compute_angular_kinematics_poisson(xb, yb, zb, body_frame=True, window_length=window_length, polyorder=polyorder)
            dest.create_dataset(FeaturesConstants.ANGULAR_VELOCITY_BODY, data=omega_body)
            dest.create_dataset(FeaturesConstants.ANGULAR_ACCEL_BODY, data=alpha_body)

            cm_smoothed, v_world, a_world = compute_linear_kinematics(cm_raw, window_length=window_length, polyorder=polyorder)
            dest.create_dataset(FeaturesConstants.CENTER_OF_MASS_RAW, data=cm_raw)
            dest.create_dataset(FeaturesConstants.CENTER_OF_MASS_SMOOTHED, data=cm_smoothed)

            dest.create_dataset(FeaturesConstants.LINEAR_VELOCITY_BODY, data=transform_velocity_to_body_frame(v_world, xb, yb, zb))
            dest.create_dataset(FeaturesConstants.LINEAR_ACCEL_BODY, data=transform_velocity_to_body_frame(a_world, xb, yb, zb))

            wings = dest.create_group('wings')
            for side in ['left', 'right']:
                side_grp = wings.create_group(side)
                side_grp.create_dataset('phi', data=wings_raw[f'{side}_phi'][start_idx:end_idx])
                side_grp.create_dataset('theta', data=wings_raw[f'{side}_theta'][start_idx:end_idx])
                side_grp.create_dataset('psi', data=wings_raw[f'{side}_psi'][start_idx:end_idx])
            
    logger.info(f"Extraction complete: {save_path}")
    return save_path

def _extract_features_and_targets(h5_path, forces_indication_vector, use_radians=False):
    """Helper to pull the raw numpy matrices out of a processed H5 file."""
    with h5py.File(h5_path, 'r') as src:
        body_matrix = np.hstack([
            src[FeaturesConstants.LINEAR_VELOCITY_BODY][:],
            src[FeaturesConstants.LINEAR_ACCEL_BODY][:],
            src[FeaturesConstants.ANGULAR_VELOCITY_BODY][:],
            src[FeaturesConstants.ANGULAR_ACCEL_BODY][:]
        ])
        
        if forces_indication_vector:
            include_indices = [3*i + j for i, bit in enumerate(forces_indication_vector) if bit == '1' for j in range(3)]
            body_matrix = body_matrix[:, include_indices]

        wing_matrix = np.column_stack([
            src[FeaturesConstants.WINGS_LEFT_PHI][:], src[FeaturesConstants.WINGS_LEFT_THETA][:], src[FeaturesConstants.WINGS_LEFT_PSI][:],
            src[FeaturesConstants.WINGS_RIGHT_PHI][:], src[FeaturesConstants.WINGS_RIGHT_THETA][:], src[FeaturesConstants.WINGS_RIGHT_PSI][:]
        ])

        # --- CONVERT TO RADIANS IF FLAG IS TRUE ---
        if use_radians:
            wing_matrix = np.deg2rad(wing_matrix)
            
    return body_matrix.astype(np.float32), wing_matrix.astype(np.float32)

def _clear_directory(dir_path: str):
    """Safely deletes all files and subdirectories inside a given path."""
    if os.path.exists(dir_path):
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                logger.error(f"Failed to delete {file_path}. Reason: {e}")
    else:
        # If it doesn't exist yet, just create it so the rest of the code doesn't crash
        os.makedirs(dir_path, exist_ok=True)

def gather_condensed_h5(processed_dir: str) -> list[str]:
    """
    All condensed *.h5 paths under processed_dir, recursively and sorted.

    The condensed data is laid out one subfolder per experiment
    (e.g. data/processed/roni_60ms/...), so this gathers every experiment into a
    single deterministic, path-sorted list. Also works for a flat directory.
    Sorting by full path keeps the trajectory order identical across
    transform_data.py and build_regressor_dataset.py (both call this), so their
    trajectory_ids stay in lockstep.
    """
    return sorted(glob.glob(os.path.join(processed_dir, "**", "*.h5"), recursive=True))


def condense_experiment(experiment_dir: str, output_root: str = PROCESSED_DATA_DIR) -> str:
    """
    Condense every raw movie in a single experiment dir into
    <output_root>/<experiment name>/ (auto-named from the input dir). Clears only
    that experiment's output dir, so re-running one experiment never touches the
    others. Returns the output dir.
    """
    experiment = os.path.basename(os.path.normpath(experiment_dir))
    output_dir = os.path.join(output_root, experiment)
    raw_files = sorted(f for f in os.listdir(experiment_dir) if f.endswith(UNPROCESSED_FILE_SUFFIX))
    if not raw_files:
        logger.warning(f"No {UNPROCESSED_FILE_SUFFIX} files in {experiment_dir}; skipping.")
        return output_dir

    logger.info(f"Condensing experiment '{experiment}' ({len(raw_files)} movies) -> {output_dir}")
    _clear_directory(output_dir)
    n = 0
    for fname in raw_files:
        save_path = _process_single_h5(os.path.join(experiment_dir, fname), output_dir)
        if save_path:
            augment_dataset(save_path)
            n += 1
    logger.info(f"  '{experiment}': condensed {n}/{len(raw_files)} movies (+ L/R-mirror augmented).")
    return output_dir


def _has_raw_h5(d: str) -> bool:
    return any(f.endswith(UNPROCESSED_FILE_SUFFIX) for f in os.listdir(d))


def run_condense(unprocessed_dir: str = UNPROCESSED_FLIGHT_DATA_DIR, output_root: str = PROCESSED_DATA_DIR):
    """
    Condense raw data into per-experiment condensed dirs. Auto-detects:
      - if `unprocessed_dir` holds raw *_analysis_smoothed.h5 files directly, it is
        treated as ONE experiment;
      - otherwise each immediate subdirectory that contains raw files is treated as
        its own experiment.
    """
    if _has_raw_h5(unprocessed_dir):
        experiments = [unprocessed_dir]
    else:
        experiments = sorted(
            os.path.join(unprocessed_dir, name)
            for name in os.listdir(unprocessed_dir)
            if os.path.isdir(os.path.join(unprocessed_dir, name))
            and _has_raw_h5(os.path.join(unprocessed_dir, name))
        )
        if not experiments:
            raise ValueError(
                f"No {UNPROCESSED_FILE_SUFFIX} files found in {unprocessed_dir} "
                f"(neither directly nor in any immediate subdirectory)."
            )

    logger.info(f"Condensing {len(experiments)} experiment(s) into {output_root}/")
    for exp in experiments:
        condense_experiment(exp, output_root)
    logger.info("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Condense raw fly-kinematics H5 into per-experiment condensed dirs."
    )
    parser.add_argument(
        '--unprocessed_dir', type=str, default=UNPROCESSED_FLIGHT_DATA_DIR,
        help="Raw data dir. Either one experiment (has *_analysis_smoothed.h5 directly) "
             "or a parent of experiment subdirs. Default: data/unprocessed_data (all experiments).")
    parser.add_argument(
        '--output_root', type=str, default=PROCESSED_DATA_DIR,
        help="Parent dir for the per-experiment condensed output. Default: data/processed.")
    args = parser.parse_args()

    run_condense(args.unprocessed_dir, args.output_root)