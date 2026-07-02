"""
Information-ceiling diagnostic for the body→latent regressor.

Answers one question: is the regressor's error a *data/feature* problem (the
24-d current+next mean body-kinematics simply don't determine the wingbeat
latent) or an *architecture/training* problem (the MLP is leaving signal on the
table)?

It does NOT load the trained regressor. It operates directly on the dataset
(body features → encoder latents) so its conclusion is independent of any
particular model. On the regressor's validation split it reports latent MSE,
mean per-dim R², and retrieval median-percentile for a ladder of predictors:

  1. predict train-mean latent        — the R²≈0 reference (trivial baseline)
  2. ridge / linear regression        — is the usable relationship even nonlinear?
  3. kNN-in-scaled-input-space (k...)  — model-agnostic upper bound on what these
                                         features support
  4. information ceiling (one-to-many) — for near-identical inputs, how much do
                                         their latents still disagree? That
                                         disagreement is irreducible noise no
                                         architecture can remove, and it caps the
                                         achievable R².

Inputs are scaled exactly as in training (VectorNormScaler fit on the train
current-wingbeat body_means, the 12-d factor reused for both halves of the
24-d input).

Run from project root:
    python code/diagnose_regressor_ceiling.py
    python code/diagnose_regressor_ceiling.py \
        --dataset_path   data/regressor_dataset/wingbeat_regressor_dataset.npz \
        --autoencoder_dir data/models/autoencoder/autoencoder_20260615_173008
"""

import argparse
import os

import numpy as np

from body_latent_regressor import (
    _load_split,
    _fit_body_scaler,
    _resolve_ae_model_dir,
)
from data_handling.body_features import (
    resolve_feature_set,
    default_scaler_type,
    apply_body_scaler_np,
)
from evaluate_body_to_wingbeat import compute_per_dim_r2, compute_retrieval_metrics


def _scaled_inputs(
    splits: dict,
    feature_set: str | None = None,
    body_feature_indices: list[int] | None = None,
    scaler_type: str | None = None,
    include_next: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_train, X_val, y_train, y_val) with inputs selected + scaled exactly
    as in training: the chosen feature set is selected from the current (and, when
    include_next, next) body_means, with the scaler fit on the train current-wingbeat
    selected channels."""
    tr, vl = splits["train_idx"], splits["val_idx"]
    indices, _ = resolve_feature_set(feature_set, body_feature_indices)
    scaler_type = scaler_type or default_scaler_type(indices)
    body_scaler = _fit_body_scaler(splits["body_means"][tr], indices, scaler_type)
    X = apply_body_scaler_np(splits["body_means"], splits["next_body_means"], body_scaler,
                             include_next=include_next)
    y = splits["target_latents"].astype(np.float32)
    # single_wing carries a per-wingbeat (N, 2, D) latent (left, right); flatten to
    # (N, 2D) — the exact target the regressor's MSE optimizes — so the 2-D ridge/kNN/
    # R² machinery below is representation-agnostic. 'sa' stays (N, D) untouched.
    if y.ndim > 2:
        y = y.reshape(y.shape[0], -1)
    return X[tr], X[vl], y[tr], y[vl]


def _report(name: str, pred: np.ndarray, true: np.ndarray) -> dict:
    """Print + return latent MSE, mean per-dim R², retrieval median percentile."""
    mse = float(((pred - true) ** 2).mean())
    r2  = compute_per_dim_r2(pred, true)
    ret = compute_retrieval_metrics(pred, true)
    mean_r2 = float(r2.mean())
    print(f"  {name:<34s}  MSE={mse:7.4f}  meanR²={mean_r2:+.3f}  "
          f"med%ile={ret['median_percentile']:.3f}  top10={ret.get('top10_accuracy', float('nan')):.3f}")
    return {"name": name, "mse": mse, "mean_r2": mean_r2,
            "median_percentile": ret["median_percentile"], "r2_per_dim": r2}


def _ridge_fit_predict(Xtr, ytr, Xval, alpha: float):
    """Closed-form multi-output ridge with an intercept (bias not penalized)."""
    mu_x = Xtr.mean(0, keepdims=True)
    mu_y = ytr.mean(0, keepdims=True)
    Xc, Yc = Xtr - mu_x, ytr - mu_y
    d = Xc.shape[1]
    W = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(d), Xc.T @ Yc)         # (d, D)
    return (Xval - mu_x) @ W + mu_y


def _knn_predict(Xtr, ytr, Xval, k: int, batch: int = 256) -> np.ndarray:
    """Predict each val latent as the mean of its k nearest train latents
    (Euclidean in scaled-input space). Batched to bound memory."""
    out = np.empty((Xval.shape[0], ytr.shape[1]), dtype=np.float64)
    tr_sq = (Xtr ** 2).sum(1)                                             # (Ntr,)
    for s in range(0, Xval.shape[0], batch):
        xb = Xval[s:s + batch]
        d2 = tr_sq[None, :] + (xb ** 2).sum(1)[:, None] - 2.0 * xb @ Xtr.T  # (b, Ntr)
        nn = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]               # (b, k)
        out[s:s + xb.shape[0]] = ytr[nn].mean(axis=1)
    return out.astype(np.float32)


def _information_ceiling(X, y, traj_ids, n_pairs_max: int = 4000) -> dict:
    """Estimate the achievable-R² ceiling from one-to-many structure.

    For each point, find its nearest neighbour in scaled-input space — excluding
    itself AND any wingbeat from the same trajectory (consecutive wingbeats are
    near-identical in both body kinematics and latent, so including them would
    leak temporal autocorrelation and grossly over-state the ceiling, exactly
    what the by-trajectory train/val split exists to prevent).

    If two points (from different trajectories) have ~identical inputs they share
    the same true conditional-mean latent, so E‖y_i − y_nn‖² ≈ 2·σ²_noise:
        R²_max ≈ 1 − σ²_noise / Var(y) = 1 − 0.5·meanSqDiff / totalVar.
    Also reports how close those NN inputs actually are, so the estimate's
    validity is legible (close NNs → trustworthy ceiling).
    """
    N = X.shape[0]
    rng = np.random.default_rng(0)
    idx = rng.choice(N, size=min(N, n_pairs_max), replace=False)
    x_sq = (X ** 2).sum(1)
    nn_input_d  = np.empty(len(idx))
    sq_lat_diff = np.empty(len(idx))
    for j, i in enumerate(idx):
        d2 = x_sq + x_sq[i] - 2.0 * X @ X[i]
        d2[traj_ids == traj_ids[i]] = np.inf      # exclude self + same-trajectory
        nn = int(np.argmin(d2))
        nn_input_d[j]  = np.sqrt(max(d2[nn], 0.0))
        sq_lat_diff[j] = ((y[i] - y[nn]) ** 2).sum()
    total_var = float(((y - y.mean(0)) ** 2).sum(axis=1).mean())          # E‖y-ȳ‖²
    sigma2_noise = 0.5 * float(sq_lat_diff.mean())
    r2_max = 1.0 - sigma2_noise / max(total_var, 1e-12)
    typ_input_d = float(np.median(np.sqrt(x_sq)))                         # scale ref
    return {
        "r2_max_estimate":     r2_max,
        "sigma2_noise":        sigma2_noise,
        "total_latent_var":    total_var,
        "median_nn_input_dist": float(np.median(nn_input_d)),
        "median_point_norm":   typ_input_d,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.strip(),
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset_path",    default="data/regressor_dataset/wingbeat_regressor_dataset.npz")
    p.add_argument("--autoencoder_dir", default="data/models/autoencoder",
                   help="Dir with val_indices.json (defines the regressor's val split).")
    p.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10, 25, 50])
    p.add_argument("--ridge_alphas", type=float, nargs="+", default=[0.0, 1.0, 10.0, 100.0])
    p.add_argument("--feature_set", default="full",
                   help="Named body feature set to diagnose (full, pitch, pitch_accel, ...). "
                        "Restrict inputs to compare the achievable ceiling across feature views.")
    p.add_argument("--body_feature_indices", type=int, nargs="+", default=None,
                   help="Explicit body-channel indices (overrides --feature_set).")
    p.add_argument("--current_only", action="store_true",
                   help="Use only the current wingbeat's body kinematics (drop the "
                        "next-wingbeat half), matching a use_next_wingbeat=false regressor.")
    args = p.parse_args()

    ae_dir = _resolve_ae_model_dir(args.autoencoder_dir)
    splits = _load_split(args.dataset_path, os.path.join(ae_dir, "val_indices.json"))
    indices, feat_names = resolve_feature_set(args.feature_set, args.body_feature_indices)
    include_next = not args.current_only
    Xtr, Xval, ytr, yval = _scaled_inputs(splits, args.feature_set, args.body_feature_indices,
                                          include_next=include_next)
    print(f"Dataset:     {args.dataset_path}")
    print(f"AE split:    {ae_dir}")
    print(f"Feature set: {feat_names} (indices {indices})")
    print(f"Window:      {'current+next' if include_next else 'current-only'} wingbeat(s)")
    print(f"Train/val:   {len(Xtr)} / {len(Xval)}   input_dim={Xtr.shape[1]}  latent_dim={ytr.shape[1]}")
    print()

    print("Predictors on the val set  (R²=1 perfect, 0 = predict-the-mean):")
    rows = []
    # 1. predict train mean
    pred_mean = np.repeat(ytr.mean(0, keepdims=True), len(yval), axis=0)
    rows.append(_report("predict train-mean (baseline)", pred_mean, yval))
    # 2. ridge / linear
    for a in args.ridge_alphas:
        rows.append(_report(f"ridge (alpha={a:g})", _ridge_fit_predict(Xtr, ytr, Xval, a), yval))
    # 3. kNN upper bound
    for k in args.ks:
        rows.append(_report(f"kNN (k={k})", _knn_predict(Xtr, ytr, Xval, k), yval))

    # 4. information ceiling
    print()
    all_tids = splits["trajectory_ids"][np.concatenate([splits["train_idx"], splits["val_idx"]])]
    ceil = _information_ceiling(np.concatenate([Xtr, Xval], 0),
                               np.concatenate([ytr, yval], 0),
                               all_tids)
    print("Information ceiling (one-to-many structure of the body→latent map):")
    print(f"  estimated max achievable mean R²  : {ceil['r2_max_estimate']:+.3f}")
    print(f"  irreducible noise var σ²          : {ceil['sigma2_noise']:.4f}  "
          f"of total latent var {ceil['total_latent_var']:.4f}")
    print(f"  median nearest-neighbour input dist: {ceil['median_nn_input_dist']:.4f}  "
          f"(median point norm {ceil['median_point_norm']:.4f})")

    # --- Verdict ---
    best_knn = max(r["mean_r2"] for r in rows if r["name"].startswith("kNN"))
    best_lin = max(r["mean_r2"] for r in rows if r["name"].startswith("ridge"))
    print()
    print("Verdict guide:")
    print(f"  best linear R² = {best_lin:+.3f} | best kNN R² = {best_knn:+.3f} | "
          f"ceiling ≈ {ceil['r2_max_estimate']:+.3f}")
    print("  - If kNN ≈ ceiling and both are low → features are the limit "
          "(data problem). Enrich inputs before tuning the model.")
    print("  - If kNN ≫ a trained MLP's R² → the model is underfitting "
          "(architecture/training problem).")


if __name__ == "__main__":
    main()
