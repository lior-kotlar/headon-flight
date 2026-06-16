"""
Body-kinematics → wingbeat-latent regressor.

A small MLP that maps the (current + next) mean body-kinematics vectors of a
wingbeat (24 dims) to that wingbeat's autoencoder latent (D dims) and
standardized log-duration. Body input is scaled by VectorNormScaler — each
physical 3-vector (v, a, ω, α) is divided by its average L2 magnitude on the
training set, preserving direction. The same 12-d scale factor is applied to
both halves of the 24-d input so current/next channels stay directly comparable.

Trains on data/regressor_dataset/wingbeat_regressor_dataset.npz produced by
code/data_handling/build_regressor_dataset.py. Train/val split is by
trajectory_id, using val_indices.json from the autoencoder run, so the
regressor's val set is wingbeats from trajectories the autoencoder also
never saw.

Run from project root:
    python code/body_latent_regressor.py
    python code/body_latent_regressor.py --config code/body_latent_regressor_config.json
"""

import argparse
import copy
import itertools
import json
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from normalizer import VectorNormScaler
from data_handling.body_features import (
    resolve_feature_set,
    default_scaler_type,
    apply_body_scaler_np,
)


_ACTIVATIONS = {
    "relu":      nn.ReLU,
    "gelu":      nn.GELU,
    "leakyrelu": nn.LeakyReLU,
    "silu":      nn.SiLU,
    "tanh":      nn.Tanh,
}

# Complex-valued config keys: a bare list-of-scalars is a single value;
# only a list-of-lists is a grid axis. Matches the autoencoder convention.
_COMPLEX_VALUE_KEYS = {"hidden_dims", "body_feature_indices"}


class BodyLatentRegressor(nn.Module):
    """MLP: scaled (body_mean, next_body_mean) → (latent, standardized log-duration).

    `n_wings` controls how many wingbeat latents are predicted from the shared trunk:
      * n_wings=1 (default, 'sa' representation): latent is (B, D), exactly the original
        behavior. The latent_head is Linear(prev, D), so old checkpoints load unchanged.
      * n_wings=2 ('single_wing'): the head emits 2·D and is reshaped to (B, 2, D) —
        z_L and z_R, both decoded by the one shared single-wing decoder downstream.
    """
    def __init__(
        self,
        in_dim: int = 24,
        latent_dim: int = 16,
        hidden_dims=(128, 128),
        activation: str = "gelu",
        dropout: float = 0.1,
        n_wings: int = 1,
    ):
        super().__init__()
        act_cls = _ACTIVATIONS[activation.lower()]
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(act_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        self.trunk         = nn.Sequential(*layers)
        self.latent_head   = nn.Linear(prev, n_wings * latent_dim)
        self.duration_head = nn.Linear(prev, 1)

        self.in_dim      = in_dim
        self.latent_dim  = latent_dim
        self.n_wings     = n_wings
        self.hidden_dims = list(hidden_dims)
        self.activation  = activation
        self.dropout     = dropout

    def forward(self, body_scaled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(body_scaled)
        latent = self.latent_head(h)
        if self.n_wings > 1:
            latent = latent.view(latent.size(0), self.n_wings, self.latent_dim)
        duration = self.duration_head(h).squeeze(-1)
        return latent, duration


def _load_split(dataset_path: str, ae_val_indices_path: str) -> dict:
    """Load the regressor dataset and mask wingbeats by autoencoder val trajectories.

    Detects the representation from the npz contents: a 'single_wing' dataset carries
    target_latents_left/right (stacked into target_latents (N, 2, D), n_wings=2); a
    legacy 'sa' dataset carries target_latents (N, D), n_wings=1.
    """
    data = np.load(dataset_path)
    body_means      = data["body_means"].astype(np.float32)
    next_body_means = data["next_body_means"].astype(np.float32)
    durations       = data["durations"].astype(np.int32)
    traj_ids        = data["trajectory_ids"].astype(np.int32)

    two_wing = "target_latents_left" in data.files
    if two_wing:
        # (N, 2, D): wing axis = [left, right].
        target_latents = np.stack(
            [data["target_latents_left"].astype(np.float32),
             data["target_latents_right"].astype(np.float32)],
            axis=1,
        )
        n_wings = 2
    else:
        target_latents = data["target_latents"].astype(np.float32)   # (N, D)
        n_wings = 1

    with open(ae_val_indices_path) as f:
        meta = json.load(f)
    val_trajs = set(int(i) for i in meta["val_indices"])
    val_mask = np.array([int(t) in val_trajs for t in traj_ids], dtype=bool)

    split = {
        "body_means":      body_means,
        "next_body_means": next_body_means,
        "target_latents":  target_latents,
        "n_wings":         n_wings,
        "durations":       durations,
        "trajectory_ids":  traj_ids,
        "train_idx":       np.where(~val_mask)[0],
        "val_idx":         np.where( val_mask)[0],
    }
    # Ground-truth wing arrays aligned 1:1 to each latent, used by the evaluator to
    # compute decode(latent)-vs-ground-truth without re-deriving wingbeats by position.
    # 'sa': (N, 6, L) normalized SA; 'single_wing': (N, 3, L) left and right residuals.
    if "sa_wingbeats" in data.files:
        split["sa_wingbeats"] = data["sa_wingbeats"].astype(np.float32)
    if "single_wing_left" in data.files:
        split["single_wing_left"]  = data["single_wing_left"].astype(np.float32)
        split["single_wing_right"] = data["single_wing_right"].astype(np.float32)
    return split


def _fit_standardizer(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu    = arr.mean(axis=0)
    sigma = arr.std(axis=0)
    sigma = np.where(sigma > 1e-8, sigma, 1.0)
    return mu.astype(np.float32), sigma.astype(np.float32)


def _fit_vector_norm_scale(body_means_train: np.ndarray) -> np.ndarray:
    """Fit VectorNormScaler on the current-wingbeat body_means (N, 12). Returns the
    12-d scale_factor vector (one scale per 3-vector, repeated 3× to align with channels).
    The same vector is reused for the next-wingbeat half by the caller."""
    scaler = VectorNormScaler(global_normalizer=True)
    scaler.fit(torch.from_numpy(body_means_train))
    return scaler.scale_factors.squeeze(0).numpy().astype(np.float32)  # (12,)


def _fit_body_scaler(body_means_train: np.ndarray, indices: list[int], scaler_type: str) -> dict:
    """Fit a serializable body-scaler over the selected channels of the TRAIN
    current-wingbeat body_means. The same scaler is reused for the next-wingbeat
    half by the caller (see apply_body_scaler_np).

    "vector_norm": divide each 3-vector by its mean L2 magnitude (original full-input
        behavior; requires a feature count that is a multiple of 3).
    "standardize": per-channel z-score (works for any subset).
    """
    sel = body_means_train[:, np.asarray(indices, dtype=np.int64)]
    if scaler_type == "vector_norm":
        if sel.shape[1] % 3 != 0:
            raise ValueError(
                f"body_scaler 'vector_norm' needs a feature count that is a multiple of 3; "
                f"got {sel.shape[1]} ({list(indices)}). Use 'standardize' for this feature set."
            )
        scale = _fit_vector_norm_scale(sel)                              # (k,)
        scaler = {"type": "vector_norm", "scale_factors": [float(v) for v in scale]}
        # Legacy full-input checkpoints omit indices; only record them for subsets.
        if list(indices) != list(range(body_means_train.shape[1])):
            scaler["indices"] = [int(i) for i in indices]
        return scaler
    if scaler_type == "standardize":
        mu = sel.mean(axis=0)
        sd = sel.std(axis=0)
        sd = np.where(sd > 1e-8, sd, 1.0)
        return {
            "type":    "standardize",
            "indices": [int(i) for i in indices],
            "mean":    [float(v) for v in mu],
            "std":     [float(v) for v in sd],
        }
    raise ValueError(f"Unknown body_scaler {scaler_type!r}. Options: vector_norm, standardize.")


def _make_loader(X, y_latent, y_dur, batch_size, shuffle):
    ds = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y_latent),
        torch.from_numpy(y_dur),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _print_diagnostics(splits: dict, dataset_path: str, ae_model_dir: str) -> None:
    """Sanity checks before training: train/val sizes, per-feature distribution
    shift, and baseline val losses (predict-the-mean). If the trained model can't
    beat these baselines on val, there's a deeper data problem no architecture
    or regularization knob will fix.
    """
    body  = splits["body_means"]
    nbody = splits["next_body_means"]
    lat   = splits["target_latents"]
    dur   = splits["durations"].astype(np.float32)
    tids  = splits["trajectory_ids"]
    tr    = splits["train_idx"]
    vl    = splits["val_idx"]

    n_tr_trajs = int(len(np.unique(tids[tr])))
    n_vl_trajs = int(len(np.unique(tids[vl])))
    pct_val = 100 * len(vl) / max(1, (len(tr) + len(vl)))
    print()
    print("=== Dataset diagnostics ===")
    print(f"Dataset:      {dataset_path}")
    print(f"AE split:     {ae_model_dir}")
    print(f"Wingbeats:    {len(tr)} train / {len(vl)} val  ({pct_val:.1f}% val)")
    print(f"Trajectories: {n_tr_trajs} train / {n_vl_trajs} val")
    print()

    body_channel_names = [
        "v_x", "v_y", "v_z",
        "a_x", "a_y", "a_z",
        "w_x", "w_y", "w_z",
        "alpha_x", "alpha_y", "alpha_z",
    ]
    print("Feature distribution comparison  (|Δμ|/σ_train > ~1 = noteworthy shift):")
    print(f"  {'channel':24s}  {'tr_mean':>10s}  {'tr_std':>10s}  {'va_mean':>10s}  {'va_std':>10s}  {'|Δμ|/σ':>7s}")

    def _row(name, t_arr, v_arr):
        tm, ts = float(t_arr.mean()), float(t_arr.std())
        vm, vs = float(v_arr.mean()), float(v_arr.std())
        shift = abs(vm - tm) / max(ts, 1e-8)
        print(f"  {name:24s}  {tm:10.4f}  {ts:10.4f}  {vm:10.4f}  {vs:10.4f}  {shift:7.2f}")

    for i, name in enumerate(body_channel_names):
        _row(f"body[{i:>2d}] {name}",  body[tr, i],  body[vl, i])
    for i, name in enumerate(body_channel_names):
        _row(f"next[{i:>2d}] {name}",  nbody[tr, i], nbody[vl, i])
    # Latent may be (N, D) ('sa') or (N, n_wings, D) ('single_wing'); flatten the
    # per-wingbeat latent for the per-dim distribution-shift readout.
    lat_flat = lat.reshape(lat.shape[0], -1)
    for i in range(lat_flat.shape[1]):
        _row(f"latent[{i:>2d}]", lat_flat[tr, i], lat_flat[vl, i])
    log_dur = np.log(dur)
    _row("duration (samples)", dur[tr], dur[vl])
    _row("log_duration",       log_dur[tr], log_dur[vl])

    # --- VectorNormScaler scale factors (computed on train, applied to both halves) ---
    scale_12 = _fit_vector_norm_scale(body[tr])
    vector_names = ["v", "a", "ω", "α"]
    print()
    print("VectorNormScaler scale factors (mean L2 magnitude per physical 3-vector, train set):")
    for i, vname in enumerate(vector_names):
        print(f"  ‖{vname}‖_mean = {scale_12[3 * i]:.4f}")

    # --- Baselines on val ---
    # The trained model needs to beat these. If it can't, the problem isn't the model.
    print()
    print("Val-set baselines (the trained model should be below these):")

    lat_train_mean = lat[tr].mean(axis=0, keepdims=True)
    base_lat_train = float(((lat[vl] - lat_train_mean) ** 2).mean())
    lat_val_mean   = lat[vl].mean(axis=0, keepdims=True)
    base_lat_val   = float(((lat[vl] - lat_val_mean) ** 2).mean())

    # Duration baselines in the SAME standardized log-space the model is trained on.
    train_log_mu  = float(log_dur[tr].mean())
    train_log_std = float(log_dur[tr].std())
    if train_log_std < 1e-8:
        train_log_std = 1.0
    val_in_train_z = (log_dur[vl] - train_log_mu) / train_log_std

    base_dur_zero    = float((val_in_train_z ** 2).mean())                    # predict 0 (= train mean)
    base_dur_valmean = float(((val_in_train_z - val_in_train_z.mean()) ** 2).mean())  # predict val mean

    print(f"  Latent: predict train_mean(latent)             → val MSE = {base_lat_train:.4f}")
    print(f"  Latent: predict val_mean(latent)   (oracle)    → val MSE = {base_lat_val:.4f}")
    print(f"  Duration: predict 0 (= train mean), log-z      → val MSE = {base_dur_zero:.4f}")
    print(f"  Duration: predict val_mean(log-dur) (oracle)   → val MSE = {base_dur_valmean:.4f}")
    print()


def _resolve_ae_model_dir(model_dir: str) -> str:
    """Same convention as evaluate_autoencoder.py: drop into the latest run_* if needed."""
    if os.path.exists(os.path.join(model_dir, "val_indices.json")):
        return model_dir
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"{model_dir} does not exist.")
    candidates = sorted(
        d for d in os.listdir(model_dir)
        if os.path.isdir(os.path.join(model_dir, d))
        and os.path.exists(os.path.join(model_dir, d, "val_indices.json"))
    )
    if not candidates:
        raise FileNotFoundError(
            f"No val_indices.json found in {model_dir} or its subdirectories."
        )
    latest = os.path.join(model_dir, candidates[-1])
    print(f"No val_indices.json directly in {model_dir} — using latest: {latest}")
    return latest


def _read_ae_dims(ae_model_dir: str) -> tuple[int, int, str]:
    """The (latent_dim, output_len, representation) the autoencoder at ae_model_dir
    produces — these fix the regressor dataset's target shape, wingbeat length, and
    which representation (and thus how many wing latents) it carries. Read from
    best_config.json (cheap); fall back to the checkpoint if a key is absent."""
    latent_dim = L = representation = None
    cfg_path = os.path.join(ae_model_dir, "best_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            c = json.load(f)
        latent_dim     = c.get("latent_dim")
        L              = c.get("fixed_len", c.get("output_len"))
        representation = c.get("representation")
    if latent_dim is None or L is None or representation is None:
        ckpt = torch.load(os.path.join(ae_model_dir, "best_autoencoder.pt"),
                          map_location="cpu", weights_only=False)
        latent_dim     = ckpt["latent_dim"]     if latent_dim is None else latent_dim
        L              = ckpt["output_len"]      if L is None else L
        representation = ckpt.get("representation", "sa") if representation is None else representation
    return int(latent_dim), int(L), str(representation)


def _resolve_dataset_path(fixed: dict, ae_model_dir: str, device: str) -> str:
    """Return the regressor dataset path for this autoencoder, building it if needed.

    An explicit fixed["dataset_path"] is honored as-is (trust the user's file).
    Otherwise the path is derived from the AE's (latent_dim, L) under fixed["data_dir"];
    if missing or built against a different AE it is rebuilt in-process (when
    auto_build_dataset is true), mirroring autoencoder.py's auto_build_dataset flow.
    """
    explicit = fixed.get("dataset_path")
    if explicit:
        if not os.path.exists(explicit):
            raise FileNotFoundError(f"Configured dataset_path {explicit} not found.")
        return explicit

    from data_handling.build_regressor_dataset import (
        build_regressor_dataset,
        regressor_dataset_path,
        regressor_dataset_sidecar_path,
        regressor_dataset_is_valid,
    )

    latent_dim, L, representation = _read_ae_dims(ae_model_dir)
    data_dir = fixed.get("data_dir", "data")
    path     = regressor_dataset_path(data_dir, latent_dim, L, representation)
    sidecar  = regressor_dataset_sidecar_path(data_dir, latent_dim, L, representation)
    is_valid, reason = regressor_dataset_is_valid(
        path, sidecar, latent_dim=latent_dim, L=L,
        autoencoder_model_dir=ae_model_dir, representation=representation,
    )
    if is_valid:
        print(f"Regressor dataset: {path}  (latent_dim={latent_dim}, L={L})")
        return path

    if not bool(fixed.get("auto_build_dataset", True)):
        raise FileNotFoundError(
            f"Regressor dataset unusable ({reason}). Set auto_build_dataset=true, or run:\n"
            f"  python code/data_handling/build_regressor_dataset.py "
            f"--model_dir {ae_model_dir} --output {path}"
        )

    print(f"Regressor dataset unusable ({reason}) — building from {ae_model_dir} ...", flush=True)
    build_regressor_dataset(
        model_dir              = ae_model_dir,
        output                 = path,
        template_path          = fixed.get("template_path", "data/analysis/golden_template.npy"),
        asymmetry_max_multiple = float(fixed.get("asymmetry_max_multiple", 10.0)),
        device                 = device,
    )
    return path


def train_one(
    config: dict,
    save_dir: str,
    device: str,
    run_label: str,
    run_idx: int,
):
    """Train a single regressor configuration. Returns (summary dict, best_state_dict)."""
    seed = int(config.get("random_seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)

    # --- Resolve AE checkpoint dir (for val split) ---
    ae_model_dir = _resolve_ae_model_dir(config["autoencoder_model_dir"])
    val_indices_path = os.path.join(ae_model_dir, "val_indices.json")

    # --- Load dataset and split ---
    splits = _load_split(config["dataset_path"], val_indices_path)
    train_idx = splits["train_idx"]
    val_idx   = splits["val_idx"]
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise RuntimeError(
            f"Empty split: train={len(train_idx)}, val={len(val_idx)}. "
            f"Check val_indices.json matches the trajectories used to build the dataset."
        )

    body_means      = splits["body_means"]
    next_body_means = splits["next_body_means"]
    target_latents  = splits["target_latents"]
    durations       = splits["durations"]
    n_wings             = int(splits["n_wings"])
    inferred_latent_dim = target_latents.shape[-1]

    # --- Feature selection + scaling. The dataset stores the full 12-d body means;
    # the regressor consumes a configurable subset (feature_set), applied identically
    # to the current and next halves. "full" + vector_norm reproduces the original
    # 24-d input. The scaler is fit on the TRAIN current-wingbeat selected channels
    # and reused for the next half (see apply_body_scaler_np). ---
    indices, feat_names = resolve_feature_set(
        config.get("feature_set"), config.get("body_feature_indices"),
    )
    scaler_type = config.get("body_scaler") or default_scaler_type(indices)
    body_scaler = _fit_body_scaler(body_means[train_idx], indices, scaler_type)
    X_train = apply_body_scaler_np(body_means[train_idx], next_body_means[train_idx], body_scaler)
    X_val   = apply_body_scaler_np(body_means[val_idx],   next_body_means[val_idx],   body_scaler)

    # --- Duration is standardized in log-space (independent of body scaling) ---
    log_dur = np.log(durations.astype(np.float32))
    dur_mu_arr, dur_sigma_arr = _fit_standardizer(log_dur[train_idx][:, None])
    dur_mu, dur_sigma = float(dur_mu_arr[0]), float(dur_sigma_arr[0])

    y_latent_train = target_latents[train_idx]
    y_latent_val   = target_latents[val_idx]
    y_dur_train = ((log_dur[train_idx] - dur_mu) / dur_sigma).astype(np.float32)
    y_dur_val   = ((log_dur[val_idx]   - dur_mu) / dur_sigma).astype(np.float32)

    print(f"  Train: {len(X_train)} wingbeats | Val: {len(X_val)} wingbeats")
    print(f"  Feature set: {feat_names} (indices {indices}) | scaler: {scaler_type}")
    print(f"  Input dim: {X_train.shape[1]}  |  Latent dim (from dataset): {inferred_latent_dim}"
          f"  |  n_wings: {n_wings}")

    # --- Model ---
    hidden_dims = config.get("hidden_dims", [128, 128])
    model = BodyLatentRegressor(
        in_dim      = X_train.shape[1],
        latent_dim  = inferred_latent_dim,
        hidden_dims = hidden_dims,
        activation  = config.get("activation", "gelu"),
        dropout     = float(config.get("dropout", 0.1)),
        n_wings     = n_wings,
    ).to(device)

    # --- Optimizer ---
    optimizer_name = config.get("optimizer", "adamw").lower()
    lr             = float(config.get("lr", 1e-3))
    weight_decay   = float(config.get("weight_decay", 1e-3))
    if optimizer_name == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer {optimizer_name!r}")

    use_plateau = (config.get("lr_scheduler", "plateau") == "plateau")
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10)
        if use_plateau else None
    )

    # --- Training loop ---
    batch_size = int(config.get("batch_size", 64))
    n_epochs   = int(config.get("n_epochs", 200))
    dur_w      = float(config.get("duration_loss_weight", 1.0))

    train_loader = _make_loader(X_train, y_latent_train, y_dur_train, batch_size, True)
    val_loader   = _make_loader(X_val,   y_latent_val,   y_dur_val,   batch_size, False)

    mse = nn.MSELoss()

    train_hist = {"total": [], "latent": [], "duration": []}
    val_hist   = {"total": [], "latent": [], "duration": []}

    best_val   = float("inf")
    best_state = None
    best_epoch = -1

    for epoch in range(1, n_epochs + 1):
        model.train()
        sums = {"total": 0.0, "latent": 0.0, "duration": 0.0, "n": 0}
        for xb, yl, yd in train_loader:
            xb, yl, yd = xb.to(device), yl.to(device), yd.to(device)
            pred_l, pred_d = model(xb)
            loss_l = mse(pred_l, yl)
            loss_d = mse(pred_d, yd)
            loss = loss_l + dur_w * loss_d
            opt.zero_grad()
            loss.backward()
            opt.step()
            bs = xb.size(0)
            sums["total"]    += loss.item()   * bs
            sums["latent"]   += loss_l.item() * bs
            sums["duration"] += loss_d.item() * bs
            sums["n"]        += bs
        for k in ("total", "latent", "duration"):
            train_hist[k].append(sums[k] / sums["n"])

        model.eval()
        sums = {"total": 0.0, "latent": 0.0, "duration": 0.0, "n": 0}
        with torch.no_grad():
            for xb, yl, yd in val_loader:
                xb, yl, yd = xb.to(device), yl.to(device), yd.to(device)
                pred_l, pred_d = model(xb)
                loss_l = mse(pred_l, yl)
                loss_d = mse(pred_d, yd)
                loss = loss_l + dur_w * loss_d
                bs = xb.size(0)
                sums["total"]    += loss.item()   * bs
                sums["latent"]   += loss_l.item() * bs
                sums["duration"] += loss_d.item() * bs
                sums["n"]        += bs
        for k in ("total", "latent", "duration"):
            val_hist[k].append(sums[k] / sums["n"])

        if scheduler is not None:
            scheduler.step(val_hist["total"][-1])

        if val_hist["total"][-1] < best_val:
            best_val   = val_hist["total"][-1]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

        if epoch == 1 or epoch % 10 == 0 or epoch == n_epochs:
            print(
                f"  Epoch {epoch:4d}/{n_epochs}  "
                f"train(tot/lat/dur)={train_hist['total'][-1]:.5f}/"
                f"{train_hist['latent'][-1]:.5f}/{train_hist['duration'][-1]:.5f}  "
                f"val={val_hist['total'][-1]:.5f}/"
                f"{val_hist['latent'][-1]:.5f}/{val_hist['duration'][-1]:.5f}",
                flush=True,
            )

    # --- Loss plot (one per run) ---
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(train_hist["latent"],   label="train")
    ax[0].plot(val_hist["latent"],     label="val")
    ax[0].set_title("Latent MSE"); ax[0].set_xlabel("epoch")
    ax[0].legend(); ax[0].grid(True, alpha=0.4)
    ax[1].plot(train_hist["duration"], label="train")
    ax[1].plot(val_hist["duration"],   label="val")
    ax[1].set_title("Duration MSE (standardized log-space)"); ax[1].set_xlabel("epoch")
    ax[1].legend(); ax[1].grid(True, alpha=0.4)
    fig.suptitle(f"run{run_idx + 1} {run_label}  best_val={best_val:.5f}@ep{best_epoch}")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"losses_run{run_idx + 1}_{run_label}.png"), dpi=120)
    plt.close(fig)

    summary = {
        "best_val_loss":         float(best_val),
        "best_val_latent_mse":   float(val_hist["latent"][best_epoch - 1]),
        "best_val_duration_mse": float(val_hist["duration"][best_epoch - 1]),
        "best_epoch":            int(best_epoch),
        "autoencoder_model_dir": ae_model_dir,
        "feature_set":          config.get("feature_set") or ("full" if list(indices) == list(range(12)) else "custom"),
        "body_feature_indices": [int(i) for i in indices],
        "body_feature_names":   list(feat_names),
        "body_scaler":          body_scaler,    # vector_norm or standardize; carries its own indices
        "duration_standardizer": {"mu": dur_mu, "sigma": dur_sigma, "space": "log"},
        "inferred_latent_dim":   int(inferred_latent_dim),
        "n_wings":               int(n_wings),
        "in_dim":                int(X_train.shape[1]),
    }
    return summary, best_state


def main():
    parser = argparse.ArgumentParser(description="Body→latent regressor grid search.")
    parser.add_argument("--config", default="code/body_latent_regressor_config.json",
                        help="Path to JSON config.")
    parser.add_argument("--job_name", default=None,
                        help="Run-directory prefix; defaults to 'run'.")
    args = parser.parse_args()

    with open(args.config) as f:
        raw_config = json.load(f)

    fixed: dict = {}
    grid:  dict = {}
    for k, v in raw_config.items():
        if k in _COMPLEX_VALUE_KEYS:
            is_grid = isinstance(v, list) and len(v) > 0 and isinstance(v[0], (list, tuple))
        else:
            is_grid = isinstance(v, list)
        (grid if is_grid else fixed)[k] = v

    grid_keys = list(grid.keys())
    combos    = list(itertools.product(*grid.values()))
    n_runs    = len(combos)
    print(f"Grid search: {n_runs} run(s)"
          + (f" over {grid_keys}" if grid_keys else " (no grid params)"))

    device = fixed.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Organize runs by feature set so pitch-only and full-input experiments live in
    # separate trees: data/models/body_latent_regressor/<feature_set>/<prefix>_<ts>/.
    if fixed.get("feature_set"):
        feature_set_name = str(fixed["feature_set"])
    elif fixed.get("body_feature_indices"):
        feature_set_name = "custom"
    else:
        feature_set_name = "full"
    job_name = (args.job_name or "").strip()
    prefix   = job_name if job_name else feature_set_name
    base_save_dir = fixed.get("save_dir", "data/models/body_latent_regressor")
    save_dir = os.path.join(base_save_dir, feature_set_name, f"{prefix}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Models → {save_dir}", flush=True)

    # --- One-time diagnostics: the train/val split is identical across grid runs,
    # so this only needs to run once before the loop ---
    ae_model_dir_diag = _resolve_ae_model_dir(fixed["autoencoder_model_dir"])
    val_indices_path  = os.path.join(ae_model_dir_diag, "val_indices.json")
    # Resolve (and auto-build if needed) the dataset matching this AE's (latent_dim, L),
    # then pin it into `fixed` so train_one and the post-training eval all use it.
    fixed["dataset_path"] = _resolve_dataset_path(fixed, ae_model_dir_diag, device)
    splits_for_diag   = _load_split(fixed["dataset_path"], val_indices_path)
    _print_diagnostics(splits_for_diag, fixed["dataset_path"], ae_model_dir_diag)

    summary = []
    best_overall_loss   = float("inf")
    best_overall_state  = None
    best_overall_extras = None
    best_overall_config = None

    for run_idx, combo in enumerate(combos):
        run_config = {**fixed, **dict(zip(grid_keys, combo))}
        run_label  = "_".join(f"{k}{v}" for k, v in zip(grid_keys, combo)) if grid_keys else "single"
        print(f"\n--- Run {run_idx + 1}/{n_runs} | {dict(zip(grid_keys, combo))} ---", flush=True)

        run_summary, best_state = train_one(run_config, save_dir, device, run_label, run_idx)
        run_summary_for_grid = {
            **dict(zip(grid_keys, combo)),
            "best_val_loss":         run_summary["best_val_loss"],
            "best_val_latent_mse":   run_summary["best_val_latent_mse"],
            "best_val_duration_mse": run_summary["best_val_duration_mse"],
            "best_epoch":            run_summary["best_epoch"],
        }
        summary.append(run_summary_for_grid)
        print(f"  → best_val={run_summary['best_val_loss']:.5f} @ ep {run_summary['best_epoch']}")

        if run_summary["best_val_loss"] < best_overall_loss:
            best_overall_loss   = run_summary["best_val_loss"]
            best_overall_state  = copy.deepcopy(best_state)
            best_overall_extras = run_summary
            best_overall_config = run_config

    # --- Persist the overall best checkpoint at the run dir root ---
    cfg = best_overall_config
    n_wings_best   = int(best_overall_extras.get("n_wings", 1))
    representation = "single_wing" if n_wings_best == 2 else "sa"
    ckpt = {
        "state_dict":            best_overall_state,
        "in_dim":                best_overall_extras["in_dim"],
        "latent_dim":            best_overall_extras["inferred_latent_dim"],
        "n_wings":               n_wings_best,
        "representation":        representation,
        "hidden_dims":           cfg.get("hidden_dims", [128, 128]),
        "activation":            cfg.get("activation", "gelu"),
        "dropout":               float(cfg.get("dropout", 0.1)),
        "feature_set":           best_overall_extras["feature_set"],
        "body_feature_indices":  best_overall_extras["body_feature_indices"],
        "body_feature_names":    best_overall_extras["body_feature_names"],
        "body_scaler":           best_overall_extras["body_scaler"],
        "duration_standardizer": best_overall_extras["duration_standardizer"],
        "best_val_loss":         best_overall_extras["best_val_loss"],
        "best_val_latent_mse":   best_overall_extras["best_val_latent_mse"],
        "best_val_duration_mse": best_overall_extras["best_val_duration_mse"],
        "best_epoch":            best_overall_extras["best_epoch"],
        "autoencoder_model_dir": best_overall_extras["autoencoder_model_dir"],
    }
    torch.save(ckpt, os.path.join(save_dir, "best_body_latent_regressor.pt"))

    with open(os.path.join(save_dir, "best_config.json"), "w") as f:
        json.dump(best_overall_config, f, indent=2)
    with open(os.path.join(save_dir, "grid_search_summary.json"), "w") as f:
        json.dump(sorted(summary, key=lambda r: r["best_val_loss"]), f, indent=2)

    print()
    print(f"Done. Best val_loss={best_overall_loss:.5f}")
    print(f"Saved → {save_dir}")

    # --- Auto-trigger end-to-end evaluation on the overall-best checkpoint.
    # Same suite as `python code/evaluate_body_to_wingbeat.py`. Wrapped in
    # try/except so a plotting hiccup doesn't tank a multi-hour training run.
    try:
        from evaluate_body_to_wingbeat import run_evaluation
        print("\n--- Running post-training evaluation ---", flush=True)
        run_evaluation(
            regressor_dir   = save_dir,
            autoencoder_dir = best_overall_extras["autoencoder_model_dir"],
            dataset_path    = fixed["dataset_path"],
            device          = device,
        )
    except Exception as exc:
        print(f"  Skipping post-training evaluation: {exc}", flush=True)


if __name__ == "__main__":
    main()
