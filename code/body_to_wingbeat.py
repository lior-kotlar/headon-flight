"""
BodyToWingbeat: trained body-latent regressor + frozen autoencoder decoder.

Composes a BodyLatentRegressor (body_mean → latent + duration) with the decoder
half of a trained WingbeatAutoencoder. Has no trainable parameters of its own.

Uses:
  - End-to-end evaluation: reconstruction quality from body kinematics alone.
  - Inference: combined with a wingbeat segmenter that supplies the body_mean
    per predicted wingbeat.

Run from project root for a smoke-test eval on the regressor val set:
    python code/body_to_wingbeat.py
    python code/body_to_wingbeat.py \
        --regressor_dir   data/models/body_latent_regressor \
        --autoencoder_dir data/models/autoencoder
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

from autoencoder import WingbeatAutoencoder
from body_latent_regressor import BodyLatentRegressor, _load_split
from transform_data import _cubic_resample


class BodyToWingbeat(nn.Module):
    """Wraps a trained regressor + frozen decoder; outputs S/A wingbeats.

    The regressor expects a 24-d input formed by concatenating the CURRENT and
    NEXT wingbeat's mean body kinematics, each divided by a per-3-vector
    VectorNormScaler scale factor fit on the training set.
    """
    def __init__(
        self,
        regressor: BodyLatentRegressor,
        autoencoder: WingbeatAutoencoder,
        body_scale_12: torch.Tensor,
        dur_mu: float,
        dur_sigma: float,
        min_duration: int = 2,
    ):
        super().__init__()
        self.regressor = regressor
        self.decoder   = autoencoder.decoder
        for p in self.decoder.parameters():
            p.requires_grad_(False)

        # The 12-d scale vector is applied identically to each half of the 24-d input.
        self.register_buffer("body_scale_12", body_scale_12)
        self.dur_mu       = float(dur_mu)
        self.dur_sigma    = float(dur_sigma)
        self.min_duration = int(min_duration)

    def scale_body(self, body_mean: torch.Tensor, next_body_mean: torch.Tensor) -> torch.Tensor:
        """Both inputs: (B, 12). Returns (B, 24) scaled by shared 12-d factor."""
        scaled_curr = body_mean      / self.body_scale_12
        scaled_next = next_body_mean / self.body_scale_12
        return torch.cat([scaled_curr, scaled_next], dim=-1)

    def predict_latent_and_duration(
        self,
        body_mean: torch.Tensor,
        next_body_mean: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Raw body kinematics → (latent (B, D), duration (B,) int).

        body_mean, next_body_mean: (B, 12) each.
        """
        x = self.scale_body(body_mean, next_body_mean)
        pred_l, pred_d_std = self.regressor(x)
        log_dur = pred_d_std * self.dur_sigma + self.dur_mu
        dur = torch.exp(log_dur).round().clamp(min=self.min_duration).long()
        return pred_l, dur

    def forward(
        self,
        body_mean: torch.Tensor,
        next_body_mean: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Decode at fixed L, then CubicSpline-resample each row to its predicted duration.

        Returns a list of (T_i, 6) S/A tensors (CPU). Lengths vary by row, so we cannot stack.
        The decoder always produces L samples; CubicSpline interpolation happens here in
        numpy because torch's `F.interpolate` has no 1D cubic mode.
        """
        latent, dur = self.predict_latent_and_duration(body_mean, next_body_mean)
        wing_L = self.decoder(latent)                                # (B, 6, L), L = self.decoder.output_len
        wing_L_np = wing_L.transpose(1, 2).detach().cpu().numpy()    # (B, L, 6) for the resampler
        outputs = []
        for i in range(wing_L_np.shape[0]):
            T_i = int(dur[i].item())
            wing_i = _cubic_resample(wing_L_np[i], T_i)        # (T_i, 6)
            outputs.append(torch.from_numpy(wing_i))
        return outputs


def load_body_to_wingbeat(
    regressor_ckpt_path: str,
    autoencoder_ckpt_path: str,
    device: str = "cpu",
) -> BodyToWingbeat:
    """Load regressor and autoencoder checkpoints and wire them into a BodyToWingbeat."""
    r_ckpt = torch.load(regressor_ckpt_path, map_location=device, weights_only=False)
    regressor = BodyLatentRegressor(
        in_dim      = r_ckpt["in_dim"],
        latent_dim  = r_ckpt["latent_dim"],
        hidden_dims = r_ckpt["hidden_dims"],
        activation  = r_ckpt["activation"],
        dropout     = r_ckpt["dropout"],
    )
    regressor.load_state_dict(r_ckpt["state_dict"])
    regressor.to(device).eval()

    a_ckpt = torch.load(autoencoder_ckpt_path, map_location=device, weights_only=False)
    ae = WingbeatAutoencoder(
        latent_dim          = a_ckpt["latent_dim"],
        activation          = a_ckpt.get("activation", "gelu"),
        dropout             = a_ckpt.get("dropout", 0.0),
        base_channels       = a_ckpt.get("base_channels", 128),
        bottleneck_len      = a_ckpt.get("bottleneck_len", 12),
        decoder_kernel_size = a_ckpt.get("decoder_kernel_size", 5),
        output_len          = a_ckpt["output_len"],
    )
    ae.load_state_dict(a_ckpt["state_dict"])
    ae.to(device).eval()

    if ae.latent_dim != regressor.latent_dim:
        raise ValueError(
            f"latent_dim mismatch: regressor={regressor.latent_dim} ae={ae.latent_dim}"
        )

    body_scaler   = r_ckpt["body_scaler"]
    if body_scaler.get("type") != "vector_norm":
        raise ValueError(
            f"Unexpected body_scaler.type {body_scaler.get('type')!r}; expected 'vector_norm'."
        )
    body_scale_12 = torch.tensor(body_scaler["scale_factors"], dtype=torch.float32)
    dur_mu        = float(r_ckpt["duration_standardizer"]["mu"])
    dur_sigma     = float(r_ckpt["duration_standardizer"]["sigma"])

    return BodyToWingbeat(
        regressor, ae, body_scale_12.to(device), dur_mu, dur_sigma,
    ).to(device)


def _resolve_run_dir(parent_dir: str, checkpoint_name: str) -> str:
    """Return parent_dir if it holds the checkpoint, else its latest subdirectory that does."""
    if os.path.exists(os.path.join(parent_dir, checkpoint_name)):
        return parent_dir
    candidates = sorted(
        d for d in os.listdir(parent_dir)
        if os.path.isdir(os.path.join(parent_dir, d))
        and os.path.exists(os.path.join(parent_dir, d, checkpoint_name))
    )
    if not candidates:
        raise FileNotFoundError(
            f"No {checkpoint_name} found in {parent_dir} or its subdirectories."
        )
    return os.path.join(parent_dir, candidates[-1])


def main():
    parser = argparse.ArgumentParser(description="End-to-end body→wingbeat smoke-test eval.")
    parser.add_argument("--regressor_dir",    default="data/models/body_latent_regressor")
    parser.add_argument("--autoencoder_dir",  default="data/models/autoencoder")
    parser.add_argument("--dataset_path",     default="data/wingbeat_regressor_dataset.npz")
    parser.add_argument("--device",           default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    r_dir = _resolve_run_dir(args.regressor_dir,   "best_body_latent_regressor.pt")
    a_dir = _resolve_run_dir(args.autoencoder_dir, "best_autoencoder.pt")
    print(f"Regressor:   {r_dir}")
    print(f"Autoencoder: {a_dir}")

    bw = load_body_to_wingbeat(
        os.path.join(r_dir, "best_body_latent_regressor.pt"),
        os.path.join(a_dir, "best_autoencoder.pt"),
        device=device,
    )
    print("Loaded BodyToWingbeat.")

    # Smoke test: latent + duration metrics on the val partition the regressor was trained against.
    val_indices_path = os.path.join(a_dir, "val_indices.json")
    splits = _load_split(args.dataset_path, val_indices_path)
    val_idx = splits["val_idx"]
    body      = torch.from_numpy(splits["body_means"][val_idx]).to(device)
    next_body = torch.from_numpy(splits["next_body_means"][val_idx]).to(device)
    gt_latent = torch.from_numpy(splits["target_latents"][val_idx]).to(device)
    gt_dur    = torch.from_numpy(splits["durations"][val_idx]).to(device).float()

    with torch.no_grad():
        pred_l, pred_d = bw.predict_latent_and_duration(body, next_body)
        latent_mse = ((pred_l - gt_latent) ** 2).mean().item()
        dur_mae    = (pred_d.float() - gt_dur).abs().mean().item()
        dur_mape   = ((pred_d.float() - gt_dur).abs() / gt_dur.clamp(min=1)).mean().item() * 100

    print(
        f"Val (n={len(val_idx)}): "
        f"latent_MSE={latent_mse:.5f}  duration_MAE={dur_mae:.2f} samples  ({dur_mape:.1f}%)"
    )


if __name__ == "__main__":
    main()
