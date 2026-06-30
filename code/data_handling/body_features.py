"""
Body-kinematics feature-set definitions: the single source of truth for *which*
columns of the 12-d body-mean vector a body→latent regressor consumes, and how a
serialized body-scaler is applied to them.

The 12-d body vector layout is ``[v(3), a(3), ω(3), α(3)]`` in body frame; the ω
and α blocks are ordered ``(yaw, pitch, roll)`` (see
``process_data._extract_features_and_targets`` and
``transform_data._BODY_ALPHA_COLS``). So pitch angular velocity is index 7 and
pitch angular acceleration is index 10.

A regressor selects a feature set (a list of these indices), applied identically
to the current and next wingbeat halves of the input. The body-scaler dict stored
in a checkpoint carries everything needed to reproduce the selection + scaling at
inference time; ``scaler_to_offset_scale`` and ``apply_body_scaler_np`` are the
shared contract used by both the trainer and the inference adapter.
"""
from __future__ import annotations

import numpy as np

# Physical labels for the 12 core body-mean channels. The angular blocks use
# yaw/pitch/roll rather than x/y/z so feature sets read physically.
_CORE_BODY_CHANNEL_NAMES = [
    "v_x", "v_y", "v_z",
    "a_x", "a_y", "a_z",
    "w_yaw", "w_pitch", "w_roll",
    "alpha_yaw", "alpha_pitch", "alpha_roll",
]
N_CORE_BODY_CHANNELS = len(_CORE_BODY_CHANNEL_NAMES)  # 12

# Within-beat angular-velocity change appended by build_regressor_dataset.py as 3 extra
# channels (12, 13, 14): Δω = ω(wingbeat's last sample) − ω(first sample), per axis
# (yaw, pitch, roll). This is a finite-difference proxy for the mean angular acceleration
# α over the beat, carried in the SAME selectable feature vector so a regressor can use it
# *instead of* (or alongside) the dataset's α channels. Same definition as the
# `dvel_within` proxy in wing_asymmetry_vs_body_accel.py.
_DWITHIN_CHANNEL_NAMES = ["dwithin_yaw", "dwithin_pitch", "dwithin_roll"]

# Full per-wingbeat body feature vector = 12 core kinematics + 3 within-beat Δω channels.
BODY_CHANNEL_NAMES = _CORE_BODY_CHANNEL_NAMES + _DWITHIN_CHANNEL_NAMES
N_BODY_CHANNELS = len(BODY_CHANNEL_NAMES)  # 15

# Named feature sets → indices into the body vector. "full" reproduces the original
# 24-d (current+next) regressor input (the 12 core kinematics, no Δω proxy). The
# *_dwithin sets swap in / add the within-beat Δω proxy so each has a direct α-based
# counterpart to compare against (e.g. angular_accel ↔ dwithin, pitch_accel ↔ pitch_dwithin).
FEATURE_SETS: dict[str, list[int]] = {
    "full":          list(range(N_CORE_BODY_CHANNELS)),
    "pitch":         [7, 10],          # w_pitch, alpha_pitch
    "pitch_accel":   [10],             # alpha_pitch only
    "angular_accel": [9, 10, 11],      # yaw/pitch/roll angular acceleration
    "angular":       [6, 7, 8, 9, 10, 11],  # angular velocity + acceleration
    # --- within-beat Δω proxies for angular acceleration (channels 12–14) ---
    "dwithin":             [12, 13, 14],            # Δω all axes        (↔ angular_accel)
    "pitch_dwithin":       [13],                    # Δω pitch only      (↔ pitch_accel)
    "pitch_vel_dwithin":   [7, 13],                 # w_pitch + Δω_pitch (↔ pitch)
    "angular_vel_dwithin": [6, 7, 8, 12, 13, 14],   # angular velocity + Δω (↔ angular)
    "full_dwithin":        [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14] # full but use dwithin instead of α
}

DEFAULT_FEATURE_SET = "full"


def resolve_feature_set(
    feature_set: str | None = None,
    body_feature_indices: list[int] | None = None,
) -> tuple[list[int], list[str]]:
    """Resolve a feature-set spec to ``(indices, names)``.

    An explicit ``body_feature_indices`` wins if given; otherwise the named
    ``feature_set`` is looked up (defaulting to ``"full"``). Indices are validated
    against the 12-channel layout.
    """
    if body_feature_indices is not None:
        indices = [int(i) for i in body_feature_indices]
    else:
        name = feature_set or DEFAULT_FEATURE_SET
        if name not in FEATURE_SETS:
            raise ValueError(
                f"Unknown feature_set {name!r}. Options: {sorted(FEATURE_SETS)} "
                f"or pass body_feature_indices explicitly."
            )
        indices = list(FEATURE_SETS[name])
    if not indices:
        raise ValueError("Feature set resolved to an empty index list.")
    for i in indices:
        if not (0 <= i < N_BODY_CHANNELS):
            raise ValueError(f"Body feature index {i} out of range [0, {N_BODY_CHANNELS}).")
    names = [BODY_CHANNEL_NAMES[i] for i in indices]
    return indices, names


def default_scaler_type(indices: list[int]) -> str:
    """The scaler used when the config doesn't specify one: ``vector_norm`` for the
    full 12-channel core set (3-vector groups, original behavior), else ``standardize``
    (per-channel z-score, which works for any subset including the Δω proxy channels)."""
    return "vector_norm" if list(indices) == list(range(N_CORE_BODY_CHANNELS)) else "standardize"


def scaler_to_offset_scale(scaler: dict) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Convert a serialized body-scaler dict to ``(indices, offset, scale)`` so a
    raw 12-d body half is transformed as ``(x[:, indices] - offset) / scale``.

    Supports both stored forms:
      * ``{"type": "vector_norm", "scale_factors": [...]}`` — divide by the
        per-channel scale (no offset). ``indices`` defaults to all channels the
        ``scale_factors`` covers (legacy full-input checkpoints omit it).
      * ``{"type": "standardize", "indices": [...], "mean": [...], "std": [...]}``
        — per-channel z-score over the selected indices.
    """
    stype = scaler.get("type")
    if stype == "vector_norm":
        scale_full = np.asarray(scaler["scale_factors"], dtype=np.float32)
        if "indices" in scaler:
            indices = [int(i) for i in scaler["indices"]]
            scale = scale_full[indices]
        else:
            indices = list(range(len(scale_full)))
            scale = scale_full
        offset = np.zeros(len(indices), dtype=np.float32)
        return indices, offset, scale.astype(np.float32)
    if stype == "standardize":
        indices = [int(i) for i in scaler["indices"]]
        offset = np.asarray(scaler["mean"], dtype=np.float32)
        scale = np.asarray(scaler["std"], dtype=np.float32)
        return indices, offset, scale
    raise ValueError(f"Unknown body_scaler type {stype!r}.")


def apply_body_scaler_np(
    body_means: np.ndarray,
    next_body_means: np.ndarray,
    scaler: dict,
) -> np.ndarray:
    """Select + scale both halves of a full 12-d body input and concatenate.

    ``body_means`` / ``next_body_means``: ``(N, 12)`` each. Returns ``(N, 2*k)``
    float32 where ``k = len(indices)``.
    """
    indices, offset, scale = scaler_to_offset_scale(scaler)
    idx = np.asarray(indices, dtype=np.int64)
    cur = (body_means[:, idx] - offset) / scale
    nxt = (next_body_means[:, idx] - offset) / scale
    return np.concatenate([cur, nxt], axis=1).astype(np.float32)
