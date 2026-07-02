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
from data_handling.body_features import scaler_to_offset_scale
from data_handling.bucket_eval import sa_to_lr_norm
from transform_data import (
    _cubic_resample,
    _single_wing_to_segment,
    SINGLE_WING_PHYSICAL_SCALE,
    single_wing_template_path,
)


class BodyToWingbeat(nn.Module):
    """Wraps a trained regressor + frozen decoder; composes a full wingbeat from body kinematics.

    The regressor consumes a configurable subset of the body feature vector
    (body_feature_indices), selected identically from the CURRENT and NEXT
    wingbeat's body kinematics and scaled per-channel ((x - offset) / scale).
    The vector is the 12 core mean kinematics plus the 3 within-beat Δω proxy
    channels (12–14; see data_handling/body_features.py). Callers always pass the
    full body vector; the selection happens here, so the same call site serves any
    feature set ("full" → 24-d, "pitch" → 4-d, "dwithin" → 6-d, ...).

    Two representations:
      * 'sa'          — regressor predicts one latent z; the frozen 6-ch decoder maps
                        z → (B, 6, L) normalized S/A. (original behavior)
      * 'single_wing' — regressor predicts (z_L, z_R); the ONE shared frozen 3-ch
                        decoder maps each to (B, 3, L), stacked into a full wingbeat.
    """
    def __init__(
        self,
        regressor: BodyLatentRegressor,
        autoencoder: WingbeatAutoencoder,
        body_indices: torch.Tensor,
        body_offset: torch.Tensor,
        body_scale: torch.Tensor,
        dur_mu: float,
        dur_sigma: float,
        min_duration: int = 2,
        representation: str = "sa",
        template3: np.ndarray | None = None,
        include_next: bool = True,
    ):
        super().__init__()
        self.regressor = regressor
        self.decoder   = autoencoder.decoder
        for p in self.decoder.parameters():
            p.requires_grad_(False)

        self.representation = representation
        self.n_wings        = int(getattr(regressor, "n_wings", 1))
        # Single-wing template (template_res, 3) for raw-angle reconstruction in forward().
        # Not needed for the L/R-normalized decode used by the evaluator (residuals cancel).
        if template3 is not None:
            self.register_buffer("template3", torch.from_numpy(np.asarray(template3, np.float32)))
        else:
            self.template3 = None
        self.register_buffer("single_wing_scale", torch.from_numpy(SINGLE_WING_PHYSICAL_SCALE))

        # Selection + per-channel scaling, applied identically to each half.
        self.register_buffer("body_indices", body_indices.long())
        self.register_buffer("body_offset",  body_offset.float())
        self.register_buffer("body_scale",   body_scale.float())
        self.dur_mu       = float(dur_mu)
        self.dur_sigma    = float(dur_sigma)
        self.min_duration = int(min_duration)
        # Temporal window the regressor was trained with: True → input is [current, next];
        # False → current wingbeat only (next_body_mean unused). Matched to the checkpoint.
        self.include_next = bool(include_next)

    def scale_body(self, body_mean: torch.Tensor, next_body_mean: torch.Tensor) -> torch.Tensor:
        """Both inputs: (B, n_body_channels). Selects body_indices, scales ((x - offset)
        / scale), and — when include_next — concatenates the next half → (B, 2*k); else
        returns the current half only → (B, k)."""
        scaled_curr = (body_mean.index_select(-1, self.body_indices) - self.body_offset) / self.body_scale
        if not self.include_next:
            return scaled_curr
        scaled_next = (next_body_mean.index_select(-1, self.body_indices) - self.body_offset) / self.body_scale
        return torch.cat([scaled_curr, scaled_next], dim=-1)

    def predict_latent_and_duration(
        self,
        body_mean: torch.Tensor,
        next_body_mean: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Raw body kinematics → (latent, duration (B,) int).

        latent is (B, D) for 'sa' and (B, 2, D) for 'single_wing'.
        body_mean, next_body_mean: (B, n_body_channels) each (full body feature vector).
        """
        x = self.scale_body(body_mean, next_body_mean)
        pred_l, pred_d_std = self.regressor(x)
        log_dur = pred_d_std * self.dur_sigma + self.dur_mu
        dur = torch.exp(log_dur).round().clamp(min=self.min_duration).long()
        return pred_l, dur

    def decode_lr_norm(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent(s) into a full wingbeat in the per-angle-normalized L/R-residual
        space [L_phi, L_theta, L_psi, R_phi, R_theta, R_psi] at fixed L — directly
        comparable to the ground-truth stored by build_regressor_dataset.

        'sa': sa_to_lr_norm(decoder(z)).  'single_wing': cat(decoder(z_L), decoder(z_R)),
        whose channels already ARE the per-angle-normalized left/right residuals.
        """
        if self.representation == "single_wing":
            z_l, z_r = latent[:, 0, :], latent[:, 1, :]
            left  = self.decoder(z_l)                 # (B, 3, L) normalized left residual
            right = self.decoder(z_r)                 # (B, 3, L) normalized right residual
            return torch.cat([left, right], dim=1)    # (B, 6, L)
        return sa_to_lr_norm(self.decoder(latent))    # (B, 6, L)

    def forward(
        self,
        body_mean: torch.Tensor,
        next_body_mean: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Predict and decode a full wingbeat, then CubicSpline-resample each row to its
        predicted duration. Returns a list of (T_i, 6) tensors (CPU); lengths vary so we
        cannot stack. Torch has no 1D cubic interpolation, so resampling is done in numpy.

        'sa' returns normalized S/A wingbeats (unchanged). 'single_wing' returns RAW
        wing angles [L_phi..R_psi] (residual × scale + template), ready for inference.
        """
        latent, dur = self.predict_latent_and_duration(body_mean, next_body_mean)

        if self.representation == "single_wing":
            z_l, z_r = latent[:, 0, :], latent[:, 1, :]
            left_res  = (self.decoder(z_l) * self.single_wing_scale.view(1, 3, 1))   # (B,3,L) raw residual
            right_res = (self.decoder(z_r) * self.single_wing_scale.view(1, 3, 1))
            left_np   = left_res.transpose(1, 2).detach().cpu().numpy()              # (B, L, 3)
            right_np  = right_res.transpose(1, 2).detach().cpu().numpy()
            tmpl3     = self.template3.cpu().numpy() if self.template3 is not None else None
            outputs = []
            for i in range(left_np.shape[0]):
                T_i = int(dur[i].item())
                if tmpl3 is not None:
                    left_ang  = _single_wing_to_segment(_cubic_resample(left_np[i],  T_i), tmpl3)
                    right_ang = _single_wing_to_segment(_cubic_resample(right_np[i], T_i), tmpl3)
                else:  # no template available → return residuals only
                    left_ang  = _cubic_resample(left_np[i],  T_i)
                    right_ang = _cubic_resample(right_np[i], T_i)
                outputs.append(torch.from_numpy(np.concatenate([left_ang, right_ang], axis=1)))  # (T_i, 6)
            return outputs

        wing_L = self.decoder(latent)                                # (B, 6, L)
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
    template_path: str = "data/analysis/golden_template.npy",
) -> BodyToWingbeat:
    """Load regressor and autoencoder checkpoints and wire them into a BodyToWingbeat.

    For a single-wing model the regressor is two-headed (n_wings=2) and the autoencoder
    is 3-channel; the single-wing template (sibling of template_path) is loaded so
    forward() can return raw wing angles. Both inferred from the checkpoints.
    """
    r_ckpt = torch.load(regressor_ckpt_path, map_location=device, weights_only=False)
    regressor = BodyLatentRegressor(
        in_dim      = r_ckpt["in_dim"],
        latent_dim  = r_ckpt["latent_dim"],
        hidden_dims = r_ckpt["hidden_dims"],
        activation  = r_ckpt["activation"],
        dropout     = r_ckpt["dropout"],
        n_wings     = int(r_ckpt.get("n_wings", 1)),
    )
    regressor.load_state_dict(r_ckpt["state_dict"])
    regressor.to(device).eval()

    a_ckpt = torch.load(autoencoder_ckpt_path, map_location=device, weights_only=False)
    representation = a_ckpt.get("representation", r_ckpt.get("representation", "sa"))
    ae = WingbeatAutoencoder(
        latent_dim          = a_ckpt["latent_dim"],
        in_channels         = a_ckpt.get("in_channels", 6),
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

    template3 = None
    if representation == "single_wing":
        sw_path = single_wing_template_path(template_path)
        if os.path.exists(sw_path):
            template3 = np.load(sw_path)
        else:
            print(f"  (single-wing template {sw_path} not found — forward() returns residuals)")

    # The body-scaler dict carries everything needed to reproduce selection +
    # scaling (both "vector_norm" full-input and "standardize" subset forms).
    indices, offset, scale = scaler_to_offset_scale(r_ckpt["body_scaler"])
    body_indices = torch.tensor(indices, dtype=torch.long)
    body_offset  = torch.from_numpy(offset)
    body_scale   = torch.from_numpy(scale)
    dur_mu       = float(r_ckpt["duration_standardizer"]["mu"])
    dur_sigma    = float(r_ckpt["duration_standardizer"]["sigma"])
    # Old checkpoints (pre-flag) were always current+next → default True.
    include_next = bool(r_ckpt.get("use_next_wingbeat", True))

    return BodyToWingbeat(
        regressor, ae,
        body_indices.to(device), body_offset.to(device), body_scale.to(device),
        dur_mu, dur_sigma,
        representation = representation,
        template3      = template3,
        include_next   = include_next,
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
    parser.add_argument("--dataset_path",     default="data/regressor_dataset/wingbeat_regressor_dataset.npz")
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
