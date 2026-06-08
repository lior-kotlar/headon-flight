"""
Builds the training dataset for the body-kinematics → wingbeat-latent regressor.

For every wingbeat in every (filtered) trajectory, records:
  - body_mean:       (12,)  mean body-kinematics vector over the CURRENT wingbeat
  - next_body_mean:  (12,)  mean body-kinematics vector over the NEXT wingbeat
  - target_latent:   (D,)   the trained encoder's output for the current wingbeat
  - sa_wingbeat:     (6, L) the exact normalized SA fed to the encoder (ground
                            truth for decode(latent) RMSE; avoids re-deriving the
                            wingbeat by position from wingbeats_L<L>.npz)
  - duration:        int    the current wingbeat length in samples
  - trajectory_id:   int    post-filter trajectory index (matches trajectories.npy)

The last wingbeat of every trajectory has no "next" so it is skipped — both during
training (this script) and at inference (a segmenter that emits wingbeat boundaries
also won't have a next for the trajectory's final wingbeat). The two-vector input
exposes 1st-order temporal evolution of the body state to the regressor.

The trajectory ordering and asymmetry filter mirror `transform_data.py`, so
`trajectory_id` here is the same index used by `val_indices.json` written by
the autoencoder grid search — splitting on this matches the autoencoder's
train/val split.

Usage (from project root):
    python code/data_handling/build_regressor_dataset.py
    python code/data_handling/build_regressor_dataset.py \
        --model_dir data/models/autoencoder/some_run \
        --output data/wingbeat_regressor_dataset.npz
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

# Allow imports from sibling code/ and from this dir (process_data.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from autoencoder import WingbeatAutoencoder
from transform_data import (
    _wingbeat_peaks,
    _segment_to_sa,
    _cubic_resample,
    SA_PHYSICAL_SCALE,
    trajectory_asymmetry_score,
)
from process_data import PROCESSED_TRAIN_FLIGHT_DATA_DIR, _extract_features_and_targets


def _resolve_model_dir(model_dir: str) -> str:
    """If model_dir has best_autoencoder.pt directly, use it; otherwise pick the
    latest run_*/ subdirectory that contains a checkpoint."""
    if os.path.exists(os.path.join(model_dir, "best_autoencoder.pt")):
        return model_dir
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"{model_dir} does not exist.")
    candidates = sorted(
        d for d in os.listdir(model_dir)
        if os.path.isdir(os.path.join(model_dir, d))
        and os.path.exists(os.path.join(model_dir, d, "best_autoencoder.pt"))
    )
    if not candidates:
        raise FileNotFoundError(
            f"No best_autoencoder.pt found under {model_dir} or any of its subdirectories."
        )
    latest = os.path.join(model_dir, candidates[-1])
    print(f"No checkpoint directly in {model_dir} — using latest: {latest}")
    return latest


def _load_autoencoder(model_dir: str, device: str) -> WingbeatAutoencoder:
    ckpt_path = os.path.join(model_dir, "best_autoencoder.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Dropout is irrelevant in eval mode; we still construct with the saved value for parity.
    model = WingbeatAutoencoder(
        latent_dim          = ckpt["latent_dim"],
        activation          = ckpt.get("activation", "gelu"),
        dropout             = ckpt.get("dropout", 0.0),
        base_channels       = ckpt.get("base_channels", 128),
        bottleneck_len      = ckpt.get("bottleneck_len", 12),
        decoder_kernel_size = ckpt.get("decoder_kernel_size", 5),
        output_len          = ckpt["output_len"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    print(
        f"Autoencoder loaded: latent_dim={model.latent_dim}  "
        f"base_channels={model.base_channels}  bottleneck_len={model.bottleneck_len}  "
        f"output_len={model.output_len}  "
        f"val_loss={ckpt.get('val_loss', 'unknown')}"
    )
    return model


def _load_filtered_trajectories(
    processed_dir: str,
    asymmetry_max_multiple: float,
    use_radians: bool,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Reads paired (body, wing) per trajectory in the same sorted H5 order used by
    transform_data._load_wing_trajectories, then applies the same asymmetry filter
    used in transform_data.py:main(). Returns body and wing lists aligned with
    the post-filter trajectory indices.
    """
    files = sorted(f for f in os.listdir(processed_dir) if f.endswith(".h5"))
    if not files:
        raise FileNotFoundError(f"No .h5 files found in {processed_dir}")

    body_list: list[np.ndarray] = []
    wing_list: list[np.ndarray] = []
    for fname in files:
        body, wing = _extract_features_and_targets(
            os.path.join(processed_dir, fname),
            forces_indication_vector=None,   # None → no body-column filtering, full 12
            use_radians=use_radians,
        )
        body_list.append(body)
        wing_list.append(wing)
    print(f"Loaded {len(body_list)} raw trajectories from {processed_dir}")

    # Mirror the asymmetry filter in transform_data.py:main().
    if asymmetry_max_multiple > 0 and len(wing_list) > 0:
        scores = np.array([trajectory_asymmetry_score(w) for w in wing_list], dtype=np.float64)
        median_score = float(np.median(scores))
        if median_score > 0:
            threshold = asymmetry_max_multiple * median_score
            keep_mask = scores <= threshold
            n_dropped = int((~keep_mask).sum())
            if n_dropped:
                dropped = [(i, float(scores[i])) for i in range(len(scores)) if not keep_mask[i]]
                print(
                    f"Asymmetry filter: dropping {n_dropped}/{len(wing_list)} trajectories "
                    f"(>{asymmetry_max_multiple}× median = {threshold:.4f}, median={median_score:.4f})."
                )
                print("  Dropped idx → score: " + ", ".join(f"{i}→{s:.3f}" for i, s in dropped))
                body_list = [b for b, k in zip(body_list, keep_mask) if k]
                wing_list = [w for w, k in zip(wing_list, keep_mask) if k]

    return body_list, wing_list


def regressor_dataset_path(data_dir: str, latent_dim: int, L: int) -> str:
    """Canonical dataset path, keyed on the (latent_dim, L) of the source autoencoder.
    Lets callers derive the file a given autoencoder needs instead of hardcoding it."""
    return os.path.join(data_dir, f"wingbeat_regressor_dataset_dim{latent_dim}_L{L}.npz")


def regressor_dataset_sidecar_path(data_dir: str, latent_dim: int, L: int) -> str:
    return os.path.splitext(regressor_dataset_path(data_dir, latent_dim, L))[0] + ".json"


def regressor_dataset_is_valid(
    output_path: str,
    sidecar_path: str,
    *,
    latent_dim: int,
    L: int,
    autoencoder_model_dir: str,
) -> tuple[bool, str]:
    """Returns (is_valid, reason). False means the dataset is missing or was built
    against a different autoencoder (latent_dim / L / model dir) and must be rebuilt.
    Keying on autoencoder_model_dir (not just latent_dim) means pointing at a
    *different* dim-D autoencoder also forces a rebuild rather than reusing stale latents."""
    if not os.path.exists(output_path):
        return False, f"missing dataset {output_path}"
    if not os.path.exists(sidecar_path):
        return False, f"missing sidecar {sidecar_path}"
    try:
        with open(sidecar_path) as f:
            meta = json.load(f)
    except json.JSONDecodeError as e:
        return False, f"corrupt sidecar: {e}"
    if int(meta.get("latent_dim", -1)) != int(latent_dim):
        return False, f"latent_dim mismatch ({meta.get('latent_dim')} ≠ {latent_dim})"
    if int(meta.get("L", -1)) != int(L):
        return False, f"L mismatch ({meta.get('L')} ≠ {L})"
    if meta.get("autoencoder_model_dir") != autoencoder_model_dir:
        return False, (f"autoencoder_model_dir mismatch "
                       f"({meta.get('autoencoder_model_dir')!r} ≠ {autoencoder_model_dir!r})")
    return True, "ok"


def build_regressor_dataset(
    *,
    model_dir: str,
    output: str,
    processed_dir: str = PROCESSED_TRAIN_FLIGHT_DATA_DIR,
    template_path: str = "data/analysis/golden_template.npy",
    asymmetry_max_multiple: float = 10.0,
    device: str = "auto",
) -> dict:
    """Build and write the body→latent regressor dataset for one autoencoder.
    Returns the metadata dict (also written as the .json sidecar). Callable directly
    so body_latent_regressor.py can auto-build a missing/stale dataset in-process."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    template = np.load(template_path)
    print(f"Template: {template.shape}")

    model_dir = _resolve_model_dir(model_dir)
    model     = _load_autoencoder(model_dir, device=device)
    latent_dim = model.latent_dim

    body_list, wing_list = _load_filtered_trajectories(
        processed_dir, asymmetry_max_multiple, use_radians=True,
    )
    n_traj = len(body_list)
    print(f"{n_traj} trajectories after filtering — processing wingbeats...")

    sa_scale = torch.from_numpy(SA_PHYSICAL_SCALE).view(6, 1).to(device)
    L = int(model.output_len)
    print(f"Resampling each wingbeat to L={L} samples (CubicSpline) before encoding.")

    body_means_list:      list[np.ndarray] = []
    next_body_means_list: list[np.ndarray] = []
    target_latents_list:  list[np.ndarray] = []
    sa_wingbeats_list:    list[np.ndarray] = []   # (6, L) normalized SA actually fed to the encoder
    durations_list:       list[int]        = []
    trajectory_ids_list:  list[int]        = []
    n_skipped         = 0    # current or next wingbeat had non-positive duration
    n_dropped_last    = 0    # last wingbeat of trajectory, no "next" available

    with torch.no_grad():
        for traj_id, (body, wing) in enumerate(zip(body_list, wing_list)):
            # Defensive: if body and wing time-lengths happen to disagree by 1, align them.
            n_aligned = min(body.shape[0], wing.shape[0])
            body = body[:n_aligned]
            wing = wing[:n_aligned]

            peaks = _wingbeat_peaks(wing)
            # i indexes the CURRENT wingbeat (peaks[i] → peaks[i+1]); the NEXT wingbeat
            # is peaks[i+1] → peaks[i+2], so we need i+2 < len(peaks). The trajectory's
            # final wingbeat (one per trajectory) is dropped on purpose.
            if len(peaks) >= 2:
                n_dropped_last += 1
            for i in range(len(peaks) - 2):
                start, end = int(peaks[i]),     int(peaks[i + 1])
                next_end   = int(peaks[i + 2])
                duration      = end - start
                next_duration = next_end - end
                if duration <= 0 or next_duration <= 0:
                    n_skipped += 1
                    continue

                body_segment      = body[start:end]                               # (n, 12)
                next_body_segment = body[end:next_end]                            # (m, 12)
                wing_segment      = wing[start:end]                               # (n, 6)

                body_mean      = body_segment.mean(axis=0).astype(np.float32)     # (12,)
                next_body_mean = next_body_segment.mean(axis=0).astype(np.float32)  # (12,)

                # Build the same input the encoder was trained on: SA at native length,
                # CubicSpline-resampled to L, transposed to (6, L), divided by SA_PHYSICAL_SCALE.
                sa = _segment_to_sa(wing_segment, template)                       # (n, 6)
                sa_L = _cubic_resample(sa, L)                                      # (L, 6)
                sa_t = torch.as_tensor(sa_L.T, device=device).unsqueeze(0) / sa_scale  # (1, 6, L)

                latent = model.encoder(sa_t).squeeze(0).cpu().numpy().astype(np.float32)

                body_means_list.append(body_mean)
                next_body_means_list.append(next_body_mean)
                target_latents_list.append(latent)
                # Store the exact (6, L) normalized SA the encoder consumed, so the
                # evaluator can compute decode(latent)-vs-ground-truth without having
                # to re-derive the wingbeat by position from wingbeats_L<L>.npz (whose
                # filtering/skip logic differs and silently misaligns the rows).
                sa_wingbeats_list.append(sa_t.squeeze(0).cpu().numpy().astype(np.float32))
                durations_list.append(duration)
                trajectory_ids_list.append(traj_id)

            if (traj_id + 1) % 20 == 0 or traj_id == n_traj - 1:
                print(f"  {traj_id + 1}/{n_traj} trajs processed — "
                      f"{len(body_means_list)} wingbeats so far")

    if n_skipped:
        print(f"Note: skipped {n_skipped} wingbeats with non-positive current or next duration.")
    if n_dropped_last:
        print(f"Note: dropped {n_dropped_last} trajectory-final wingbeats (no next wingbeat).")

    body_means      = np.stack(body_means_list)                         # (N, 12) float32
    next_body_means = np.stack(next_body_means_list)                    # (N, 12) float32
    target_latents  = np.stack(target_latents_list)                     # (N, D)  float32
    sa_wingbeats    = np.stack(sa_wingbeats_list)                       # (N, 6, L) float32
    durations       = np.asarray(durations_list, dtype=np.int32)        # (N,)
    trajectory_ids  = np.asarray(trajectory_ids_list, dtype=np.int32)   # (N,)

    out_dir = os.path.dirname(os.path.abspath(output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    np.savez(
        output,
        body_means      = body_means,
        next_body_means = next_body_means,
        target_latents  = target_latents,
        sa_wingbeats    = sa_wingbeats,
        durations       = durations,
        trajectory_ids  = trajectory_ids,
    )

    meta = {
        "n_wingbeats":           int(len(body_means)),
        "n_trajectories":        int(n_traj),
        "n_dropped_last":        int(n_dropped_last),
        "n_skipped":             int(n_skipped),
        "latent_dim":            int(latent_dim),
        "autoencoder_output_len": int(L),
        "L":                     int(L),
        "has_next_body_mean":    True,
        "has_sa_wingbeats":      True,
        "duration_min":          int(durations.min()),
        "duration_max":          int(durations.max()),
        "duration_mean":         float(durations.mean()),
        "duration_std":          float(durations.std()),
        "autoencoder_model_dir": model_dir,
        "processed_dir":         processed_dir,
        "template_path":         template_path,
        "asymmetry_max_multiple": asymmetry_max_multiple,
        "sa_scale":              SA_PHYSICAL_SCALE.tolist(),
    }
    meta_path = os.path.splitext(output)[0] + ".json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print()
    print(f"Saved {len(body_means)} wingbeats to {output}")
    print(f"  body_means:      {body_means.shape}  {body_means.dtype}")
    print(f"  next_body_means: {next_body_means.shape}  {next_body_means.dtype}")
    print(f"  target_latents: {target_latents.shape}  {target_latents.dtype}")
    print(f"  sa_wingbeats:   {sa_wingbeats.shape}  {sa_wingbeats.dtype}")
    print(f"  durations:      {durations.shape}  range [{durations.min()}, {durations.max()}]  "
          f"mean {durations.mean():.2f}  std {durations.std():.2f}")
    print(f"  trajectory_ids: {trajectory_ids.shape}  ({len(np.unique(trajectory_ids))} unique)")
    print(f"Metadata: {meta_path}")

    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Build body→latent regressor dataset.")
    parser.add_argument("--processed_dir", default=PROCESSED_TRAIN_FLIGHT_DATA_DIR,
                        help="Directory containing the processed H5 files.")
    parser.add_argument("--template_path", default="data/analysis/golden_template.npy",
                        help="Path to the golden wingbeat template .npy.")
    parser.add_argument("--model_dir", default="data/models/autoencoder",
                        help="Either a directory containing best_autoencoder.pt, "
                             "or a parent of run_<timestamp>/ subdirectories (latest is used).")
    parser.add_argument("--output", default="data/wingbeat_regressor_dataset.npz",
                        help="Output .npz file path.")
    parser.add_argument("--asymmetry_max_multiple", type=float, default=10.0,
                        help="Trajectories with asymmetry score > this × dataset-median are "
                             "filtered out. Match what transform_data.py used to build "
                             "trajectories.npy. Set to 0 to disable.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    build_regressor_dataset(
        model_dir              = args.model_dir,
        output                 = args.output,
        processed_dir          = args.processed_dir,
        template_path          = args.template_path,
        asymmetry_max_multiple = args.asymmetry_max_multiple,
        device                 = args.device,
    )


if __name__ == "__main__":
    main()
