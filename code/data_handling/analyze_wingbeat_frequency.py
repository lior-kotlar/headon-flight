"""
Analyze wingbeat frequencies across all raw flight trajectories.

Gathers all raw H5 files, detects wingbeat intervals, and creates a histogram
of wingbeat durations (in samples) to estimate flapping frequency.
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from scipy.signal import find_peaks

UNPROCESSED_DATA_DIR = "data/unprocessed_data"
OUTPUT_DIR = "data/analysis"


def _detect_wingbeat_intervals(
    wing_matrix: np.ndarray,
    min_peak_distance: int = 20,
    min_peak_prominence: float = None,
    split_by: str = "min",
) -> List[Tuple[int, int]]:
    """
    Detect [start, end) frame intervals for wingbeats.

    Args:
        wing_matrix: Array of shape [T, 6] with wing angles
        min_peak_distance: Minimum frames between consecutive extrema
        min_peak_prominence: Peak prominence threshold (auto-computed if None)
        split_by: 'min' for trough-based or 'max' for peak-based boundaries

    Returns:
        List of (start, end) tuples for each wingbeat interval
    """
    if wing_matrix.ndim != 2 or wing_matrix.shape[1] != 6:
        raise ValueError(f"Expected wing_matrix shape [T, 6], got {wing_matrix.shape}")

    left_phi = wing_matrix[:, 0]
    right_phi = wing_matrix[:, 3]
    stroke_signal = 0.5 * (left_phi + right_phi)

    if min_peak_prominence is None:
        signal_std = float(np.std(stroke_signal))
        min_peak_prominence = max(1e-6, 0.05 * signal_std)

    # Choose what to find based on split_by mode
    if split_by == "max":
        signal_to_search = stroke_signal
    else:  # split_by == 'min' (default)
        signal_to_search = -stroke_signal

    peaks, _ = find_peaks(
        signal_to_search,
        distance=max(1, int(min_peak_distance)),
        prominence=min_peak_prominence,
    )

    if peaks.size < 2:
        return []

    intervals: List[Tuple[int, int]] = []
    for i in range(len(peaks) - 1):
        start = int(peaks[i])
        end = int(peaks[i + 1])
        if end > start:
            intervals.append((start, end))
    return intervals


def _extract_wing_data(h5_path: str) -> Tuple[np.ndarray, str]:
    """
    Extract wing angle data from raw H5 file.

    Args:
        h5_path: Path to H5 file

    Returns:
        (wing_matrix, filename) tuple where wing_matrix has shape [T, 6]
    """
    try:
        with h5py.File(h5_path, "r") as f:
            # Extract wing angles: left (phi, theta, psi), right (phi, theta, psi)
            left_phi = f["wings_phi_left"][:]
            left_theta = f["wings_theta_left"][:]
            left_psi = f["wings_psi_left"][:]
            right_phi = f["wings_phi_right"][:]
            right_theta = f["wings_theta_right"][:]
            right_psi = f["wings_psi_right"][:]

            # Remove frames with NaN wing data
            valid_mask = ~(
                np.isnan(left_phi)
                | np.isnan(left_theta)
                | np.isnan(left_psi)
                | np.isnan(right_phi)
                | np.isnan(right_theta)
                | np.isnan(right_psi)
            )

            wing_matrix = np.column_stack(
                [left_phi, left_theta, left_psi, right_phi, right_theta, right_psi]
            )
            wing_matrix = wing_matrix[valid_mask]

            filename = os.path.basename(h5_path)
            return wing_matrix, filename

    except Exception as e:
        logger.warning(f"Failed to extract wing data from {h5_path}: {e}")
        return np.empty((0, 6)), os.path.basename(h5_path)


def analyze_wingbeat_frequencies(
    min_peak_distance: int = 20,
    min_peak_prominence: float = None,
    split_by: str = "min",
) -> None:
    """
    Analyze wingbeat frequencies across all raw trajectories.

    Creates a histogram of wingbeat durations (in samples) and saves as image.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Gather all raw H5 files
    h5_files = sorted(
        f
        for f in os.listdir(UNPROCESSED_DATA_DIR)
        if f.endswith("_analysis_smoothed.h5")
    )

    if not h5_files:
        logger.error(f"No H5 files found in {UNPROCESSED_DATA_DIR}")
        return

    logger.info(f"Found {len(h5_files)} raw trajectory files")

    # Collect wingbeat durations across all files
    all_wingbeat_durations = []
    all_wingbeat_counts = []

    for h5_file in h5_files:
        h5_path = os.path.join(UNPROCESSED_DATA_DIR, h5_file)
        wing_matrix, filename = _extract_wing_data(h5_path)

        if wing_matrix.shape[0] == 0:
            logger.warning(f"No valid wing data in {filename}")
            continue

        # Detect wingbeats
        intervals = _detect_wingbeat_intervals(
            wing_matrix,
            min_peak_distance=min_peak_distance,
            min_peak_prominence=min_peak_prominence,
            split_by=split_by,
        )

        if not intervals:
            logger.warning(f"No wingbeats detected in {filename}")
            continue

        # Compute durations for each wingbeat
        durations = [end - start for start, end in intervals]
        all_wingbeat_durations.extend(durations)
        all_wingbeat_counts.append(len(intervals))

        logger.info(
            f"{filename}: {len(intervals)} wingbeats, "
            f"avg={np.mean(durations):.1f} samples, "
            f"std={np.std(durations):.1f}"
        )

    if not all_wingbeat_durations:
        logger.error("No wingbeats found in any files")
        return

    # Create histogram
    fig, ax = plt.subplots(figsize=(12, 6))

    all_wingbeat_durations = np.array(all_wingbeat_durations)
    n, bins, patches = ax.hist(
        all_wingbeat_durations,
        bins=50,
        edgecolor="black",
        alpha=0.7,
        color="steelblue",
    )

    # Add statistics to plot
    mean_duration = np.mean(all_wingbeat_durations)
    median_duration = np.median(all_wingbeat_durations)
    std_duration = np.std(all_wingbeat_durations)

    ax.axvline(mean_duration, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_duration:.1f}")
    ax.axvline(median_duration, color="green", linestyle="--", linewidth=2, label=f"Median: {median_duration:.1f}")

    ax.set_xlabel("Wingbeat Duration (samples)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(
        f"Wingbeat Duration Distribution\n"
        f"Total wingbeats: {len(all_wingbeat_durations)}, "
        f"Std: {std_duration:.1f}",
        fontsize=14,
    )
    ax.legend()
    ax.grid(alpha=0.3)

    # Save figure
    output_path = os.path.join(OUTPUT_DIR, "wingbeat_frequency_histogram.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved histogram to {output_path}")

    # Print summary statistics
    sampling_rate = 16000  # Hz
    framerate = 16000 # Assuming 8kHz camera framerate (16kHz sampling / 2)

    mean_freq_hz = framerate / mean_duration
    median_freq_hz = framerate / median_duration

    logger.info("\n" + "=" * 60)
    logger.info("WINGBEAT FREQUENCY ANALYSIS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total files analyzed: {len(h5_files)}")
    logger.info(f"Total wingbeats detected: {len(all_wingbeat_durations)}")
    logger.info(f"Total files with wingbeats: {len(all_wingbeat_counts)}")
    logger.info(f"\nDuration statistics (in samples):")
    logger.info(f"  Min:    {np.min(all_wingbeat_durations):.1f}")
    logger.info(f"  Max:    {np.max(all_wingbeat_durations):.1f}")
    logger.info(f"  Mean:   {mean_duration:.1f}")
    logger.info(f"  Median: {median_duration:.1f}")
    logger.info(f"  Std:    {std_duration:.1f}")
    logger.info(f"\nFlapping frequency (assuming {framerate} Hz camera):")
    logger.info(f"  Mean freq:   {mean_freq_hz:.1f} Hz")
    logger.info(f"  Median freq: {median_freq_hz:.1f} Hz")
    logger.info("=" * 60 + "\n")

    plt.close()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze wingbeat frequencies across all raw flight trajectories."
    )
    parser.add_argument(
        "--min_peak_distance",
        type=int,
        default=20,
        help="Minimum frame distance between consecutive extrema",
    )
    parser.add_argument(
        "--min_peak_prominence",
        type=float,
        default=None,
        help="Optional peak prominence threshold",
    )
    parser.add_argument(
        "--split_by",
        type=str,
        choices=["min", "max"],
        default="min",
        help="Wingbeat boundary definition: 'min' for troughs or 'max' for peaks",
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    analyze_wingbeat_frequencies(
        min_peak_distance=args.min_peak_distance,
        min_peak_prominence=args.min_peak_prominence,
        split_by=args.split_by,
    )
