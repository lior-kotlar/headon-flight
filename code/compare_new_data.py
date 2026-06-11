#!/usr/bin/env python3
"""
Compare new unprocessed data against existing data to verify compatibility.

Loads raw H5 files from both directories, computes wingbeat statistics and
distributions, generates comparison plots, and produces a JSON summary with
KS test results.

Usage:
  python code/compare_new_data.py
  python code/compare_new_data.py --old_dir ... --new_dir ... --out_dir ... --template ...
"""

import sys
import os
import logging
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, gaussian_kde
from sklearn.decomposition import PCA

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CODE_DIR = os.path.join(_PROJECT_ROOT, "code")
sys.path.insert(0, _CODE_DIR)
sys.path.insert(0, os.path.join(_CODE_DIR, "data_handling"))

from transform_data import (
    _wingbeat_peaks, trajectory_asymmetry_score,
    generate_average_wingbeat_template, _segment_to_sa,
    _cubic_resample, SA_PHYSICAL_SCALE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

SAMPLING_RATE = 16000
RESAMPLE_L = 69
MIN_WINGBEATS = 2
ASYM_THRESHOLD_MULTIPLE = 10.0


def load_raw_wings(path: str) -> tuple[np.ndarray, int] | tuple[None, int]:
    """
    Load 6-channel wing angle time series from H5 file.

    Returns: (traj, total_frames) where traj is a (T, 6) array in radians
    (NaN-endpoints trimmed), or (None, total_frames) if load fails / no valid
    frames. Column order: [L_phi, L_theta, L_psi, R_phi, R_theta, R_psi].

    Matches the NaN-trimming and deg->rad conversion used by the main pipeline
    in process_data._process_single_h5, so stats here are directly comparable to
    the processed datasets.
    """
    try:
        with h5py.File(path, "r") as f:
            keys = ["wings_phi_left", "wings_theta_left", "wings_psi_left",
                   "wings_phi_right", "wings_theta_right", "wings_psi_right"]

            wing_angles = np.array([f[k][:] for k in keys], dtype=np.float64)  # (6, T)
            total_frames = wing_angles.shape[1]

            valid_mask = ~np.isnan(wing_angles).any(axis=0)
            if not np.any(valid_mask):
                logger.warning(f"Skipping {path}: no valid frames (all NaN)")
                return None, total_frames

            valid_indices = np.where(valid_mask)[0]
            start_idx, end_idx = valid_indices[0], valid_indices[-1] + 1
            wing_angles_trimmed = wing_angles[:, start_idx:end_idx]

            wing_angles_rad = np.deg2rad(wing_angles_trimmed.T).astype(np.float32)

            if np.isnan(wing_angles_rad).any():
                logger.warning(f"Skipping {path}: NaN found in interior after trimming")
                return None, total_frames

            return wing_angles_rad, total_frames
    except Exception as e:
        logger.warning(f"Skipping {path}: {e}")
        return None, 0


def compute_file_stats(path: str, traj: np.ndarray | None, total_frames: int) -> dict:
    """
    Compute statistics for a single H5 file from a pre-loaded trajectory.

    Returns dict with keys:
      - valid (bool)
      - path (str)
      - n_wingbeats (int)
      - valid_frames (int)
      - total_frames (int)
      - nan_fraction (float)
      - wb_durations (list of ints, not exported to JSON)
      - wb_duration_mean_samples (float)
      - wb_duration_std_samples (float)
      - wb_freq_hz (float or None)
      - asymmetry_score (float)
    """
    if traj is None:
        return {"valid": False, "path": path}

    valid_frames = len(traj)

    peaks = _wingbeat_peaks(traj)
    n_wingbeats = max(0, len(peaks) - 1)

    wb_durations = []
    if len(peaks) > 1:
        wb_durations = np.diff(peaks).tolist()

    wb_duration_mean = np.mean(wb_durations) if wb_durations else np.nan
    wb_duration_std = np.std(wb_durations) if len(wb_durations) > 1 else np.nan
    wb_freq_hz = SAMPLING_RATE / wb_duration_mean if not np.isnan(wb_duration_mean) else None

    asymmetry = trajectory_asymmetry_score(traj)

    return {
        "valid": True,
        "path": path,
        "n_wingbeats": n_wingbeats,
        "valid_frames": int(valid_frames),
        "total_frames": int(total_frames),
        "nan_fraction": 1.0 - (valid_frames / total_frames) if total_frames > 0 else 0.0,
        "wb_durations": wb_durations,
        "wb_duration_mean_samples": float(wb_duration_mean) if wb_durations else None,
        "wb_duration_std_samples": float(wb_duration_std) if wb_durations else None,
        "wb_freq_hz": float(wb_freq_hz) if wb_freq_hz is not None else None,
        "asymmetry_score": float(asymmetry),
    }


def collect_dataset_stats(data_dir: str) -> tuple[list[dict], list[np.ndarray]]:
    """
    Collect statistics for all H5 files in data_dir (non-recursive).

    Returns:
      - stats_list: list of per-file stat dicts
      - trajectories: list of (T, 6) arrays for files with n_wingbeats >= MIN_WINGBEATS
    """
    stats_list = []
    trajectories = []

    h5_files = sorted([
        f.path for f in os.scandir(data_dir)
        if f.is_file() and f.name.endswith(".h5")
    ])

    logger.info(f"Found {len(h5_files)} H5 files in {data_dir}")

    for h5_file in h5_files:
        traj, total_frames = load_raw_wings(h5_file)
        stats = compute_file_stats(h5_file, traj, total_frames)
        stats_list.append(stats)

        if stats["valid"] and stats["n_wingbeats"] >= MIN_WINGBEATS:
            trajectories.append(traj)

    n_excluded = len([s for s in stats_list if not s["valid"] or s["n_wingbeats"] < MIN_WINGBEATS])
    logger.info(f"Loaded {len(trajectories)} valid trajectories, excluded {n_excluded}")

    return stats_list, trajectories


def collect_all_wingbeats(
    trajectories: list[np.ndarray],
    template: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """
    Segment all trajectories into wingbeats and compute S/A representation.

    Returns:
      - raw_segments: list of (n, 6) arrays at native length
      - sa_norms: (N, 6*RESAMPLE_L) flattened normalized SA wingbeats for PCA
      - durations: (N,) array of wingbeat lengths in samples
    """
    raw_segments = []
    sa_list = []
    durations = []

    for traj in trajectories:
        peaks = _wingbeat_peaks(traj)

        for i in range(len(peaks) - 1):
            start, end = peaks[i], peaks[i+1]
            seg = traj[start:end]

            sa = _segment_to_sa(seg, template)
            sa_norm = sa / SA_PHYSICAL_SCALE
            sa_L = _cubic_resample(sa_norm, RESAMPLE_L)

            raw_segments.append(seg)
            sa_list.append(sa_L)
            durations.append(end - start)

    sa_array = np.array(sa_list, dtype=np.float32)
    durations_array = np.array(durations, dtype=np.int32)

    logger.info(f"Extracted {len(sa_array)} wingbeats")

    return raw_segments, sa_array, durations_array


def extract_per_wingbeat_metrics(raw_segments: list[np.ndarray]) -> dict:
    """
    Extract scalar metrics from raw wingbeat segments.

    Returns dict with keys: max_phi_left, max_phi_right, theta_range, psi_range
    Each value is a numpy array of length len(raw_segments).
    """
    max_phi_left = np.array([seg[:, 0].max() for seg in raw_segments])
    max_phi_right = np.array([seg[:, 3].max() for seg in raw_segments])
    theta_range = np.array([np.ptp(seg[:, 1]) for seg in raw_segments])
    psi_range = np.array([np.ptp(seg[:, 2]) for seg in raw_segments])

    return {
        "max_phi_left": max_phi_left,
        "max_phi_right": max_phi_right,
        "theta_range": theta_range,
        "psi_range": psi_range,
    }


def run_ks_tests(
    old_metrics: dict[str, np.ndarray],
    new_metrics: dict[str, np.ndarray],
    old_asym: np.ndarray,
    new_asym: np.ndarray,
    old_sa_norms: np.ndarray,
    new_sa_norms: np.ndarray,
    old_durations: np.ndarray,
    new_durations: np.ndarray,
) -> dict:
    """
    Run Kolmogorov-Smirnov test on key metrics.

    Returns dict mapping metric name to {"statistic": float, "p_value": float}.
    """
    tests = {
        "wb_duration": (old_durations, new_durations),
        "max_phi_left": (old_metrics["max_phi_left"], new_metrics["max_phi_left"]),
        "max_phi_right": (old_metrics["max_phi_right"], new_metrics["max_phi_right"]),
        "theta_range": (old_metrics["theta_range"], new_metrics["theta_range"]),
        "psi_range": (old_metrics["psi_range"], new_metrics["psi_range"]),
        "asymmetry_score": (old_asym, new_asym),
        "sa_residual_norm": (old_sa_norms, new_sa_norms),
    }

    results = {}
    for name, (old_vals, new_vals) in tests.items():
        stat, p_val = ks_2samp(old_vals, new_vals)
        results[name] = {
            "statistic": float(stat),
            "p_value": float(p_val),
            "significant_at_0.05": p_val < 0.05,
        }

    return results


def compute_sa_residual_norms(sa_array: np.ndarray) -> np.ndarray:
    """Compute per-wingbeat L2 norm in normalized SA space."""
    norms = np.linalg.norm(sa_array, axis=(1, 2))
    return norms


def classify_effect_size(ks_statistic: float) -> str:
    """
    Classify a KS statistic (max distance between the two CDFs, range [0, 1])
    into a qualitative effect-size band. This is the practically-meaningful
    measure of "how different" two distributions are, and unlike the p-value it
    does not collapse to "significant" just because the sample is large.

    Bands (rule-of-thumb for KS): <0.1 negligible, 0.1-0.2 small,
    0.2-0.35 moderate, >=0.35 large.
    """
    if ks_statistic < 0.1:
        return "negligible"
    elif ks_statistic < 0.2:
        return "small"
    elif ks_statistic < 0.35:
        return "moderate"
    else:
        return "large"


# ============================================================================
# Plotting functions
# ============================================================================

def plot_duration_histogram(old_durs: np.ndarray, new_durs: np.ndarray, out_path: str, ks_result: dict):
    fig, ax = plt.subplots(figsize=(10, 5))

    bins = np.linspace(min(old_durs.min(), new_durs.min()),
                       max(old_durs.max(), new_durs.max()), 40)

    ax.hist(old_durs, bins=bins, alpha=0.6, color="steelblue", label=f"Old (n={len(old_durs)})")
    ax.hist(new_durs, bins=bins, alpha=0.6, color="darkorange", label=f"New (n={len(new_durs)})")

    ax.axvline(old_durs.mean(), color="steelblue", linestyle="--", linewidth=2, alpha=0.7, label=f"Old mean: {old_durs.mean():.1f}")
    ax.axvline(new_durs.mean(), color="darkorange", linestyle="--", linewidth=2, alpha=0.7, label=f"New mean: {new_durs.mean():.1f}")

    ax.set_xlabel("Wingbeat Duration (samples)")
    ax.set_ylabel("Count")
    p_val = ks_result["p_value"]
    ax.set_title(f"Wingbeat Duration Histogram — KS p={p_val:.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"Saved {out_path}")


def plot_amplitude_distributions(old_segs: list[np.ndarray], new_segs: list[np.ndarray], out_path: str, ks_results: dict):
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))

    old_metrics = extract_per_wingbeat_metrics(old_segs)
    new_metrics = extract_per_wingbeat_metrics(new_segs)

    metrics_names = ["max_phi_left", "max_phi_right", "theta_range", "psi_range"]
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for (i, j), metric_name in zip(positions, metrics_names):
        old_vals = old_metrics[metric_name]
        new_vals = new_metrics[metric_name]

        x_range = np.linspace(
            min(old_vals.min(), new_vals.min()),
            max(old_vals.max(), new_vals.max()),
            200
        )

        if len(old_vals) > 2:
            try:
                kde_old = gaussian_kde(old_vals)
                ax[i, j].fill_between(x_range, kde_old(x_range), alpha=0.4, color="steelblue", label="Old")
            except:
                ax[i, j].hist(old_vals, bins=20, alpha=0.4, color="steelblue", label="Old", density=True)

        if len(new_vals) > 2:
            try:
                kde_new = gaussian_kde(new_vals)
                ax[i, j].fill_between(x_range, kde_new(x_range), alpha=0.4, color="darkorange", label="New")
            except:
                ax[i, j].hist(new_vals, bins=20, alpha=0.4, color="darkorange", label="New", density=True)

        p_val = ks_results[metric_name]["p_value"]
        ax[i, j].set_title(f"{metric_name} — KS p={p_val:.4f}")
        ax[i, j].legend()
        ax[i, j].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"Saved {out_path}")


def plot_asymmetry_distribution(old_asym: np.ndarray, new_asym: np.ndarray, threshold: float, out_path: str, ks_result: dict):
    fig, ax = plt.subplots(figsize=(10, 5))

    bins = np.linspace(min(old_asym.min(), new_asym.min()),
                       max(old_asym.max(), new_asym.max()), 30)

    ax.hist(old_asym, bins=bins, alpha=0.6, color="steelblue", label=f"Old (n={len(old_asym)})")
    ax.hist(new_asym, bins=bins, alpha=0.6, color="darkorange", label=f"New (n={len(new_asym)})")

    ax.axvline(threshold, color="red", linestyle="--", linewidth=2, label=f"Auto-exclude threshold: {threshold:.2f}")

    ax.set_xlabel("Asymmetry Score")
    ax.set_ylabel("Count")
    p_val = ks_result["p_value"]
    ax.set_title(f"Asymmetry Scores — KS p={p_val:.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if max(old_asym.max(), new_asym.max()) > threshold * 2:
        ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"Saved {out_path}")


def plot_template_comparison(old_template: np.ndarray, new_template: np.ndarray, out_path: str, frob_dist: float):
    fig, ax = plt.subplots(3, 1, figsize=(12, 10))

    phase = np.linspace(0, 1, old_template.shape[0])
    angle_names = ["Phi (Stroke)", "Theta (Deviation)", "Psi (Rotation)"]

    for row, angle_idx in enumerate([0, 1, 2]):
        ax[row].plot(phase, old_template[:, angle_idx], "b-", label="Old (left)", linewidth=2)
        ax[row].plot(phase, old_template[:, angle_idx+3], "r-", label="Old (right)", linewidth=2)
        ax[row].plot(phase, new_template[:, angle_idx], "b--", label="New (left)", linewidth=2)
        ax[row].plot(phase, new_template[:, angle_idx+3], "r--", label="New (right)", linewidth=2)

        ax[row].set_ylabel(f"{angle_names[row]} (rad)")
        ax[row].legend(loc="best")
        ax[row].grid(True, alpha=0.3)

    ax[2].set_xlabel("Normalized Phase [0, 1]")
    fig.suptitle(f"Golden Template Comparison — Frobenius Distance: {frob_dist:.4f} rad", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"Saved {out_path}")


def plot_sa_residual_norms(old_norms: np.ndarray, new_norms: np.ndarray, out_path: str, ks_result: dict):
    fig, ax = plt.subplots(figsize=(10, 5))

    bins = np.linspace(min(old_norms.min(), new_norms.min()),
                       max(old_norms.max(), new_norms.max()), 40)

    ax.hist(old_norms, bins=bins, alpha=0.6, color="steelblue", label=f"Old (n={len(old_norms)})")
    ax.hist(new_norms, bins=bins, alpha=0.6, color="darkorange", label=f"New (n={len(new_norms)})")

    ax.axvline(old_norms.mean(), color="steelblue", linestyle="--", linewidth=2, alpha=0.7)
    ax.axvline(new_norms.mean(), color="darkorange", linestyle="--", linewidth=2, alpha=0.7)

    ax.set_xlabel("SA Residual L2 Norm (normalized)")
    ax.set_ylabel("Count")
    p_val = ks_result["p_value"]
    ax.set_title(f"S/A Residual Norms (projected on old template) — KS p={p_val:.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"Saved {out_path}")


def plot_pca(old_sa: np.ndarray, new_sa: np.ndarray, out_path: str):
    old_flat = old_sa.reshape(len(old_sa), -1)
    new_flat = new_sa.reshape(len(new_sa), -1)

    combined = np.vstack([old_flat, new_flat])

    pca = PCA(n_components=2)
    combined_pca = pca.fit_transform(combined)

    old_pca = combined_pca[:len(old_flat)]
    new_pca = combined_pca[len(old_flat):]

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(old_pca[:, 0], old_pca[:, 1], c="steelblue", alpha=0.3, s=8, label=f"Old (n={len(old_flat)})")
    ax.scatter(new_pca[:, 0], new_pca[:, 1], c="darkorange", alpha=0.3, s=8, label=f"New (n={len(new_flat)})")

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title("2D PCA of Normalized S/A Wingbeats")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"Saved {out_path}")


def plot_ks_summary(ks_results: dict, out_path: str):
    """
    Bar chart of the KS *statistic* (effect size) per metric, NOT the p-value.

    With thousands of wingbeats every p-value is astronomically small
    (e.g. 1e-96), so a p-value bar chart renders as empty slivers and tells us
    nothing about magnitude. The KS statistic is the max gap between the two
    CDFs, bounded in [0, 1], and directly answers "how different are they".
    Bars are sorted and colored by effect-size band; the p-value is annotated.
    """
    # Sort metrics by effect size so the biggest differences are at the top.
    items = sorted(ks_results.items(), key=lambda kv: kv[1]["statistic"])
    metric_names = [k for k, _ in items]
    stats = [v["statistic"] for _, v in items]
    p_values = [v["p_value"] for _, v in items]

    band_colors = {
        "negligible": "#2ca02c",  # green
        "small": "#bcbd22",       # yellow-green
        "moderate": "#ff7f0e",    # orange
        "large": "#d62728",       # red
    }
    colors = [band_colors[classify_effect_size(s)] for s in stats]

    fig, ax = plt.subplots(figsize=(11, 6))
    y_pos = np.arange(len(metric_names))
    ax.barh(y_pos, stats, color=colors, alpha=0.85)

    # Effect-size guide lines.
    ax.axvline(0.1, color="gray", linestyle=":", linewidth=1.5)
    ax.axvline(0.2, color="gray", linestyle="--", linewidth=1.5)
    ax.axvline(0.35, color="black", linestyle="--", linewidth=1.5)
    ax.text(0.1, len(metric_names) - 0.4, " small", color="gray", fontsize=9, va="top")
    ax.text(0.2, len(metric_names) - 0.4, " moderate", color="gray", fontsize=9, va="top")
    ax.text(0.35, len(metric_names) - 0.4, " large", color="black", fontsize=9, va="top")

    # Annotate each bar with its KS statistic and p-value.
    for y, s, p in zip(y_pos, stats, p_values):
        ax.text(s + 0.005, y, f"D={s:.3f}, p={p:.1e}", va="center", fontsize=9)

    # Legend for the effect-size bands.
    from matplotlib.patches import Patch
    legend_handles = [Patch(color=c, label=b) for b, c in band_colors.items()]
    ax.legend(handles=legend_handles, title="Effect size", loc="lower right")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(metric_names)
    ax.set_xlabel("KS statistic D (max distance between CDFs)")
    ax.set_xlim(0, max(0.6, max(stats) * 1.25))
    ax.set_title("Old vs New: Distributional Difference by Metric (KS effect size)")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"Saved {out_path}")


def save_summary_json(
    old_stats: list[dict],
    new_stats: list[dict],
    old_trajs_count: int,
    new_trajs_count: int,
    ks_results: dict,
    frob_dist: float,
    old_metrics: dict,
    new_metrics: dict,
    old_asym: np.ndarray,
    new_asym: np.ndarray,
    old_sa_norms: np.ndarray,
    new_sa_norms: np.ndarray,
    out_dir: str,
):
    """Save aggregated comparison summary to JSON."""

    def compute_recommendation(ks_results: dict) -> dict:
        """
        Compatibility verdict based on EFFECT SIZE (KS statistic), not p-value.

        With N in the thousands, p-values are always tiny and "significant", so a
        p-value-based verdict would always say INCOMPATIBLE even when the
        distributions are practically identical (e.g. wingbeat duration here:
        D=0.05, negligible). We instead judge by the KS statistic magnitude.

        sa_residual_norm and asymmetry_score are the most decision-relevant: the
        autoencoder is trained on the normalized S/A representation, so a large
        shift there directly breaks compatibility with the existing model.

        Verdict:
          INCOMPATIBLE - any large (D>=0.35) difference on a key metric
          BORDERLINE   - any moderate (0.2<=D<0.35) difference
          COMPATIBLE   - all differences small/negligible (D<0.2)
        """
        bands = {m: classify_effect_size(v["statistic"]) for m, v in ks_results.items()}
        drivers = sorted(
            [m for m, b in bands.items() if b in ("moderate", "large")],
            key=lambda m: ks_results[m]["statistic"],
            reverse=True,
        )

        if any(b == "large" for b in bands.values()):
            verdict = "INCOMPATIBLE"
        elif any(b == "moderate" for b in bands.values()):
            verdict = "BORDERLINE"
        else:
            verdict = "COMPATIBLE"

        return {
            "verdict": verdict,
            "driver_metrics": drivers,
            "effect_size_bands": bands,
        }

    old_wb_durs = [d for s in old_stats if s.get("valid") for d in s.get("wb_durations", [])]
    new_wb_durs = [d for s in new_stats if s.get("valid") for d in s.get("wb_durations", [])]

    summary = {
        "timestamp": datetime.now().isoformat(),
        "metadata": {
            "sampling_rate_hz": SAMPLING_RATE,
            "resample_length": RESAMPLE_L,
            "min_wingbeats_per_file": MIN_WINGBEATS,
            "template_frobenius_distance_units": "radians",
        },
        "files": {
            "old": {
                "directory": "data/unprocessed_data",
                "n_files_loaded": len([s for s in old_stats if s.get("valid")]),
                "n_files_excluded": len([s for s in old_stats if not s.get("valid") or s.get("n_wingbeats", 0) < MIN_WINGBEATS]),
                "n_valid_trajectories": old_trajs_count,
            },
            "new": {
                "directory": "data/unprocessed_data/new_unprocessed_data",
                "n_files_loaded": len([s for s in new_stats if s.get("valid")]),
                "n_files_excluded": len([s for s in new_stats if not s.get("valid") or s.get("n_wingbeats", 0) < MIN_WINGBEATS]),
                "n_valid_trajectories": new_trajs_count,
            },
        },
        "wingbeat_statistics": {
            "old": {
                "n_wingbeats": len(old_wb_durs),
                "duration_mean_samples": float(np.mean(old_wb_durs)) if old_wb_durs else None,
                "duration_std_samples": float(np.std(old_wb_durs)) if old_wb_durs else None,
                "duration_mean_hz": SAMPLING_RATE / float(np.mean(old_wb_durs)) if old_wb_durs else None,
                "max_phi_left_mean": float(old_metrics["max_phi_left"].mean()),
                "max_phi_right_mean": float(old_metrics["max_phi_right"].mean()),
                "theta_range_mean": float(old_metrics["theta_range"].mean()),
                "psi_range_mean": float(old_metrics["psi_range"].mean()),
                "asymmetry_score_mean": float(old_asym.mean()),
                "sa_residual_norm_mean": float(old_sa_norms.mean()),
            },
            "new": {
                "n_wingbeats": len(new_wb_durs),
                "duration_mean_samples": float(np.mean(new_wb_durs)) if new_wb_durs else None,
                "duration_std_samples": float(np.std(new_wb_durs)) if new_wb_durs else None,
                "duration_mean_hz": SAMPLING_RATE / float(np.mean(new_wb_durs)) if new_wb_durs else None,
                "max_phi_left_mean": float(new_metrics["max_phi_left"].mean()),
                "max_phi_right_mean": float(new_metrics["max_phi_right"].mean()),
                "theta_range_mean": float(new_metrics["theta_range"].mean()),
                "psi_range_mean": float(new_metrics["psi_range"].mean()),
                "asymmetry_score_mean": float(new_asym.mean()),
                "sa_residual_norm_mean": float(new_sa_norms.mean()),
            },
        },
        "template_comparison": {
            "frobenius_distance": float(frob_dist),
        },
        "ks_test_results": {
            k: {
                "statistic": float(v["statistic"]),
                "p_value": float(v["p_value"]),
                "significant_at_0.05": bool(v["significant_at_0.05"]),
                "effect_size": classify_effect_size(v["statistic"]),
            }
            for k, v in ks_results.items()
        },
        "recommendation": compute_recommendation(ks_results),
    }

    out_path = os.path.join(out_dir, "comparison_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved {out_path}")
    rec = summary["recommendation"]
    logger.info(f"Recommendation: {rec['verdict']} (drivers: {', '.join(rec['driver_metrics']) or 'none'})")


def write_guide(out_dir: str):
    """Write a human-readable guide explaining every figure and test."""
    guide = """\
========================================================================
 DATASET COMPARISON GUIDE
 Old data (data/unprocessed_data) vs New data
 (data/unprocessed_data/new_unprocessed_data)
 Generated by code/compare_new_data.py
========================================================================

PURPOSE
-------
Before mixing ~100 newly-recorded movies into the training set, this report
checks whether they are statistically similar enough to the existing data.
The concern: if the new wingbeats differ in shape, timing, or quality, mixing
them would shift the "golden template", invalidate the S/A representation the
autoencoder was trained on, and degrade every downstream model.

Every file is loaded with the SAME logic as the real pipeline
(process_data._process_single_h5): the 6 wing angles are read, NaN endpoints
are trimmed, and degrees are converted to radians. Wingbeats are segmented with
transform_data._wingbeat_peaks. So the numbers here are directly comparable to
what the training pipeline actually sees.

Column order everywhere: [L_phi, L_theta, L_psi, R_phi, R_theta, R_psi]
  phi   = stroke angle (front-to-back sweep, the big motion)
  theta = deviation angle (up-down out of stroke plane)
  psi   = rotation/pitch angle (wing twist)
S = symmetric component (L+R)/2,  A = asymmetric component (L-R)/2.

------------------------------------------------------------------------
HOW TO READ THE VERDICT
------------------------------------------------------------------------
The headline metric is the KS statistic D, NOT the p-value.

  - The Kolmogorov-Smirnov test compares two distributions. It returns:
      * D (the statistic): the largest gap between the two cumulative
        distributions. Range [0,1]. This is the EFFECT SIZE = "how different".
      * p-value: probability the difference is due to chance.

  - With thousands of wingbeats, the p-value is ALWAYS astronomically small
    (1e-90 and below) even for tiny, meaningless differences. So p-values are
    useless for ranking here. We rank by D instead.

  - Effect-size bands used for D:
      D < 0.10  negligible  (practically identical)
      0.10-0.20 small
      0.20-0.35 moderate
      D >= 0.35 large       (genuinely different distribution)

  - Overall verdict (in comparison_summary.json -> recommendation):
      COMPATIBLE   : all metrics D < 0.20
      BORDERLINE   : at least one metric in 0.20-0.35
      INCOMPATIBLE : at least one metric D >= 0.35
    The "driver_metrics" field lists which metrics caused the verdict.

------------------------------------------------------------------------
THE FIGURES
------------------------------------------------------------------------

fig1_duration_histogram.png
  Wingbeat duration (in samples; 16000 Hz, so 70 samples ~= 229 Hz). Overlaid
  histograms, old vs new, with mean lines. THE primary sanity check: flies of
  the same kind in the same rig should beat their wings at the same frequency.
  Heavy overlap = good.

fig2_amplitude_distributions.png
  2x2 KDE (smoothed density) of per-wingbeat amplitude features:
    - max_phi_left / max_phi_right : peak stroke angle (stroke amplitude)
    - theta_range                  : peak-to-peak deviation
    - psi_range                    : peak-to-peak wing rotation
  Each title shows the KS p-value. Look for shifted peaks or different spread.

fig3_asymmetry_scores.png
  One asymmetry score PER FILE (not per wingbeat): mean |L-R| / physical scale,
  max over the 3 angles. This is the exact metric the pipeline uses to auto-flag
  "garbage" trajectories. The red line is the 10x-median auto-exclude threshold.
  y-axis is log-scaled. A right-shifted new distribution means the new movies
  have more left/right asymmetry (could be real maneuvers, or a tracking issue).

fig4_template_comparison.png
  The "golden template" = the average wingbeat shape over one normalized cycle.
  3 panels (phi, theta, psi), each with old (solid) vs new (dashed), left (blue)
  and right (red). Title shows the Frobenius distance (total shape difference,
  in radians). The old template is the one on disk
  (data/analysis/golden_template.npy); the new one is computed from new data
  only. If these diverge, mixing the data WILL move the template the whole
  S/A representation is built on.

fig5_sa_residual_norms.png
  THE most decision-relevant plot. Each wingbeat is converted to its normalized
  S/A residual against the OLD template (exactly what the autoencoder is fed),
  and we take its L2 norm. Old and new are both projected on the OLD template
  for a fair comparison. If the new histogram sits far to the right, new
  wingbeats do not fit the existing template basis -> the trained AE would see
  them as high-error outliers.

fig6_pca.png
  2D PCA of every wingbeat's flattened normalized S/A representation
  (69 x 6 = 414 dims), fit on the combined set, colored by source. This is the
  global "do they live in the same space" view. Intermingled clouds = similar;
  separate clusters or new-only outliers = different.

fig7_ks_summary.png
  The one-glance summary. Horizontal bars = KS statistic D per metric, sorted,
  colored by effect-size band (green negligible -> red large), each annotated
  with D and the p-value. Vertical guide lines mark the small/moderate/large
  thresholds. Read top bars first: those are the metrics that differ most.

comparison_summary.json
  Machine-readable: file counts, per-dataset wingbeat statistics, the full KS
  results (statistic, p-value, effect_size band), the template Frobenius
  distance, and the recommendation block (verdict + driver_metrics +
  per-metric effect_size_bands).

------------------------------------------------------------------------
THE METRICS TESTED (KS, old vs new)
------------------------------------------------------------------------
  wb_duration       per-wingbeat length in samples      -> timing / frequency
  max_phi_left      peak left stroke angle              -> stroke amplitude (L)
  max_phi_right     peak right stroke angle             -> stroke amplitude (R)
  theta_range       peak-to-peak deviation              -> out-of-plane motion
  psi_range         peak-to-peak rotation               -> wing twist
  asymmetry_score   per-file mean |L-R| / scale         -> L/R symmetry / quality
  sa_residual_norm  L2 norm of S/A vs old template      -> fit to existing basis

------------------------------------------------------------------------
IF THE VERDICT IS NOT "COMPATIBLE"
------------------------------------------------------------------------
Differences are not necessarily errors in the new data. Options:
  1. Investigate the drivers physically (e.g. why more asymmetry / wing twist?
     different fly line, rig, or genuinely more aggressive maneuvers?).
  2. Filter the new data (e.g. drop files above the fig3 asymmetry threshold)
     and re-run this comparison.
  3. Keep the datasets separate, each with its own template.
  4. Merge and RETRAIN everything from scratch (new golden template + AE +
     regressors) rather than reusing models trained on the old data.

Re-run anytime with:  python code/compare_new_data.py
"""
    out_path = os.path.join(out_dir, "GUIDE.txt")
    with open(out_path, "w") as f:
        f.write(guide)
    logger.info(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare new unprocessed data against existing data."
    )
    parser.add_argument(
        "--old_dir",
        default="data/unprocessed_data",
        help="Path to old unprocessed data",
    )
    parser.add_argument(
        "--new_dir",
        default="data/unprocessed_data/new_unprocessed_data",
        help="Path to new unprocessed data",
    )
    parser.add_argument(
        "--out_dir",
        default="data/analysis/dataset_comparison",
        help="Output directory for plots and summary",
    )
    parser.add_argument(
        "--template",
        default="data/analysis/golden_template.npy",
        help="Path to golden template (old)",
    )

    args = parser.parse_args()

    old_dir = os.path.join(_PROJECT_ROOT, args.old_dir)
    new_dir = os.path.join(_PROJECT_ROOT, args.new_dir)
    out_dir = os.path.join(_PROJECT_ROOT, args.out_dir)
    template_path = os.path.join(_PROJECT_ROOT, args.template)

    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Output directory: {out_dir}")

    # Load data
    logger.info("Loading old data...")
    old_stats, old_trajs = collect_dataset_stats(old_dir)

    logger.info("Loading new data...")
    new_stats, new_trajs = collect_dataset_stats(new_dir)

    if not old_trajs or not new_trajs:
        logger.error("Not enough valid trajectories to proceed")
        return

    # Load/compute templates
    logger.info(f"Loading old golden template from {template_path}...")
    old_template = np.load(template_path)
    logger.info(f"Old template shape: {old_template.shape}")

    logger.info("Computing new golden template...")
    new_template = generate_average_wingbeat_template(
        new_trajs, template_res=RESAMPLE_L, plot_template=False
    )
    logger.info(f"New template shape: {new_template.shape}")

    frob_dist = np.linalg.norm(old_template - new_template)
    logger.info(f"Frobenius distance between templates: {frob_dist:.4f} rad")

    # Collect wingbeats
    logger.info("Collecting wingbeats from old data...")
    old_segs, old_sa, old_durations = collect_all_wingbeats(old_trajs, old_template)

    logger.info("Collecting wingbeats from new data (projected on old template)...")
    new_segs, new_sa, new_durations = collect_all_wingbeats(new_trajs, old_template)

    # Extract metrics
    logger.info("Extracting per-wingbeat metrics...")
    old_metrics = extract_per_wingbeat_metrics(old_segs)
    new_metrics = extract_per_wingbeat_metrics(new_segs)

    old_asym = np.array([trajectory_asymmetry_score(traj) for traj in old_trajs])
    new_asym = np.array([trajectory_asymmetry_score(traj) for traj in new_trajs])

    old_sa_norms = compute_sa_residual_norms(old_sa)
    new_sa_norms = compute_sa_residual_norms(new_sa)

    # Run KS tests
    logger.info("Running KS tests...")
    ks_results = run_ks_tests(
        old_metrics, new_metrics,
        old_asym, new_asym,
        old_sa_norms, new_sa_norms,
        old_durations, new_durations,
    )

    # Plot
    logger.info("Generating plots...")
    plot_duration_histogram(
        old_durations, new_durations,
        os.path.join(out_dir, "fig1_duration_histogram.png"),
        ks_results["wb_duration"],
    )

    plot_amplitude_distributions(
        old_segs, new_segs,
        os.path.join(out_dir, "fig2_amplitude_distributions.png"),
        ks_results,
    )

    asym_threshold = np.median(np.concatenate([old_asym, new_asym])) * ASYM_THRESHOLD_MULTIPLE
    plot_asymmetry_distribution(
        old_asym, new_asym, asym_threshold,
        os.path.join(out_dir, "fig3_asymmetry_scores.png"),
        ks_results["asymmetry_score"],
    )

    plot_template_comparison(
        old_template, new_template,
        os.path.join(out_dir, "fig4_template_comparison.png"),
        frob_dist,
    )

    plot_sa_residual_norms(
        old_sa_norms, new_sa_norms,
        os.path.join(out_dir, "fig5_sa_residual_norms.png"),
        ks_results["sa_residual_norm"],
    )

    plot_pca(
        old_sa, new_sa,
        os.path.join(out_dir, "fig6_pca.png"),
    )

    plot_ks_summary(
        ks_results,
        os.path.join(out_dir, "fig7_ks_summary.png"),
    )

    # Save summary JSON
    save_summary_json(
        old_stats, new_stats,
        len(old_trajs), len(new_trajs),
        ks_results, frob_dist,
        old_metrics, new_metrics,
        old_asym, new_asym,
        old_sa_norms, new_sa_norms,
        out_dir,
    )

    write_guide(out_dir)

    logger.info("Done!")


if __name__ == "__main__":
    main()
