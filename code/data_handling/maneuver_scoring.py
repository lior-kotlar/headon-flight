"""
Maneuver scoring for wingbeats.

Detects "maneuvering" windows in a sequence of per-wingbeat body-kinematic
vectors and propagates the detection back to a per-wingbeat graded score in
[0, 1]. Body-kinematics-only by design.

Currently angular-acceleration only — the three signed components
    (alpha_yaw, alpha_pitch, alpha_roll)
Linear acceleration is intentionally excluded for now because body-frame
linear acceleration carries a persistent gravity-projection baseline that
trivially passes the sign-consistency check (the fly is always "accelerating
downward" in body coords when upright). Gravity-subtracted linear accel can
be added back later.

A window of W consecutive wingbeats (within a single trajectory) is labeled
a maneuver, per channel k, if:
  - all W per-wingbeat magnitudes |x^k_i| are at or above a per-channel
    magnitude threshold T_mag[k]  (rejects single-spike noise: 3 low + 1 high)
  - the sign-consistency ratio  |Σ x^k_i| / Σ |x^k_i|  is at or above
    T_coh                          (rejects axis-flipping: yaw-right + yaw-left
                                    or strong α_yaw then strong α_pitch)

Output: per-wingbeat (N, 3) graded score in {0, 1/W, ..., 1}, equal to the
fraction of overlapping in-trajectory windows that flagged this wingbeat on
each channel. Wingbeats whose trajectory is shorter than W get score 0.
"""

import numpy as np

CHANNEL_LABELS = ('alpha_yaw', 'alpha_pitch', 'alpha_roll')


def _trajectory_runs(trajectory_ids: np.ndarray) -> list[tuple[int, int]]:
    """
    Return [(start, end_exclusive), ...] for each maximal run of identical
    trajectory_ids. Assumes wingbeats from one trajectory are contiguous in
    the array — true by construction in transform_data's fixed-L builder.
    """
    runs = []
    n = len(trajectory_ids)
    if n == 0:
        return runs
    run_start = 0
    for i in range(1, n + 1):
        if i == n or trajectory_ids[i] != trajectory_ids[i - 1]:
            runs.append((run_start, i))
            run_start = i
    return runs


def per_window_flags(
    channel_scalars: np.ndarray,        # (N, C) signed per-wingbeat scalars
    trajectory_ids:  np.ndarray,        # (N,) integer trajectory tag
    W:               int,
    T_mag:           np.ndarray,        # (C,) per-channel magnitude threshold
    T_coh:           float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slide a length-W window across each trajectory's run of wingbeats and flag
    per-channel maneuvers. Windows never cross trajectory boundaries.

    Returns:
        window_starts: (n_windows,) start index of each emitted window
        flags:         (n_windows, C) bool — maneuver flag per (window, channel)
    """
    n, C = channel_scalars.shape
    if T_mag.shape != (C,):
        raise ValueError(f"T_mag shape {T_mag.shape} ≠ ({C},)")

    starts_list: list[int] = []
    flags_list:  list[np.ndarray] = []
    for run_start, run_end in _trajectory_runs(np.asarray(trajectory_ids)):
        run_len = run_end - run_start
        if run_len < W:
            continue
        for ws in range(run_start, run_end - W + 1):
            block      = channel_scalars[ws:ws + W]         # (W, C)
            abs_block  = np.abs(block)
            min_mag    = abs_block.min(axis=0)              # (C,)
            sum_signed = block.sum(axis=0)                  # (C,)
            sum_abs    = abs_block.sum(axis=0)              # (C,)
            # Coherence is undefined when sum_abs == 0; treat as 0 (vector cancels).
            with np.errstate(divide='ignore', invalid='ignore'):
                coh = np.where(sum_abs > 0, np.abs(sum_signed) / sum_abs, 0.0)
            flag = (min_mag >= T_mag) & (coh >= T_coh)
            starts_list.append(ws)
            flags_list.append(flag)

    if not starts_list:
        return np.empty((0,), dtype=np.int64), np.empty((0, C), dtype=bool)
    return np.asarray(starts_list, dtype=np.int64), np.stack(flags_list, axis=0)


def graded_scores(
    window_starts: np.ndarray,
    window_flags:  np.ndarray,
    n_wingbeats:   int,
    W:             int,
) -> np.ndarray:
    """
    For each wingbeat, score[i, c] = (# windows containing i that were flagged on
    channel c) / (# windows containing i). Wingbeats covered by no valid window
    (trajectories shorter than W) get score 0 on every channel.
    """
    C = window_flags.shape[1] if window_flags.size else len(CHANNEL_LABELS)

    flag_sum  = np.zeros((n_wingbeats, C), dtype=np.int32)
    n_overlap = np.zeros(n_wingbeats,       dtype=np.int32)
    for j, s in enumerate(window_starts):
        e = int(s) + W
        flag_sum[s:e] += window_flags[j].astype(np.int32)
        n_overlap[s:e] += 1
    safe_overlap = np.maximum(n_overlap, 1)
    scores = flag_sum.astype(np.float32) / safe_overlap[:, None].astype(np.float32)
    scores[n_overlap == 0] = 0.0
    return scores


def compute_maneuver_scores(
    mean_alpha:       np.ndarray,    # (N, 3) per-wingbeat mean angular accel vector
    trajectory_ids:   np.ndarray,    # (N,) per-wingbeat trajectory tag
    W:                int   = 4,
    T_coh:            float = 0.75,
    T_mag_percentile: float = 75.0,
) -> tuple[np.ndarray, dict]:
    """
    Top-level entry point.

    Takes the (N, 3) per-wingbeat mean angular-acceleration vectors
    (yaw, pitch, roll), derives per-channel magnitude thresholds as percentiles
    over the corpus, runs the windowed detector inside each trajectory, and
    returns:
        scores:   (N, 3) float32 graded score in {0, 1/W, ..., 1}
        meta:     dict with resolved thresholds and build parameters (suitable
                  to merge into the sidecar JSON for drift detection)
    """
    if mean_alpha.shape[-1] != 3:
        raise ValueError(f"mean_alpha must be (N, 3); got {mean_alpha.shape}")
    if mean_alpha.shape[0] != len(trajectory_ids):
        raise ValueError(
            f"shape mismatch: mean_alpha {mean_alpha.shape}, "
            f"trajectory_ids {np.asarray(trajectory_ids).shape}"
        )

    channel_scalars = np.asarray(mean_alpha, dtype=np.float64)            # (N, 3)
    T_mag = np.percentile(np.abs(channel_scalars), T_mag_percentile, axis=0).astype(np.float64)

    starts, flags = per_window_flags(
        channel_scalars = channel_scalars,
        trajectory_ids  = np.asarray(trajectory_ids),
        W               = W,
        T_mag           = T_mag,
        T_coh           = T_coh,
    )
    scores = graded_scores(starts, flags, len(channel_scalars), W)

    meta = {
        "maneuver_W":                int(W),
        "maneuver_T_coh":            float(T_coh),
        "maneuver_T_mag_percentile": float(T_mag_percentile),
        "maneuver_T_mag":            [float(v) for v in T_mag],
        "maneuver_channel_labels":   list(CHANNEL_LABELS),
        "maneuver_n_windows":        int(len(starts)),
        "maneuver_n_wingbeats_flagged_any": int((scores.max(axis=1) > 0).sum()),
    }
    return scores.astype(np.float32), meta
