"""
Memory-efficient sparse Diffusion Map of per-wingbeat wing kinematics in PyTorch.

Computes a diffusion-map embedding from the wing angles alone (stroke phi,
deviation theta, rotation psi over a wingbeat). Body kinematics are NOT part of
the metric: the embedding is built purely from wing motion, so two wingbeats are
"near" only when their wing kinematics agree. Body angular accelerations (yaw,
pitch, roll) are still carried alongside as *colour-only* diagnostics — they let
plot_ddm_embedding.py ask post-hoc "does the wing-only embedding organise by body
response?" without ever influencing the embedding (a non-circular comparison).

(The module/function names keep the historical "directed diffusion map" wording;
the map is now undirected — wing-only — and the body label F is no longer folded
into the feature vector.)

Pure-PyTorch, device-agnostic by design: there is no FAISS / CUDA-specific
dependency, so the exact same code path runs on a CPU dev node and a SLURM GPU
node — only the `device` differs.

Pipeline (all device-resident where possible):
    1. build_augmented_features : flatten + weight kinematics, derivative penalty
       and cycle penalty into one 2D matrix whose plain squared-L2 distance equals
       the custom diffusion metric. (No epsilon here — see step 3.)
    2. knn                      : exact k-NN via a chunked brute-force matmul that
       never materialises the full N*N distance matrix (peak memory O(chunk*N)).
    3. build_sparse_operator    : pick the kernel bandwidth epsilon (auto = median
       squared k-NN distance), form affinities exp(-d^2 / epsilon) -> sparse W ->
       symmetrise -> symmetric-normalised P_sym = D^{-1/2} W D^{-1/2}.
    4. diffusion_eigendecomp    : top-(k+1) eigenpairs of P_sym (torch.lobpcg on
       GPU, scipy.sparse.linalg.eigsh fallback), drop the trivial constant mode,
       return the diffusion coordinates lambda_i * psi_i for the next k modes.

Why squared-L2 == diffusion metric: every additive penalty in step 1 is folded
into the feature vector as sqrt(weight) * component, so ||a-b||^2 reproduces the
weighted sum of squared component differences. Epsilon is applied later (step 3)
as a single global scalar; since it is monotonic it never changes k-NN ranking,
which is what lets the bandwidth be chosen *from* the k-NN distances.

Auto-epsilon: in the high-dimensional augmented space the squared distances are
large, so a fixed epsilon=1 makes exp(-d^2) underflow to 0 and the graph
disconnects. Setting epsilon=None (the default) picks epsilon = median of the
non-self k-NN squared distances, so the typical neighbour affinity lands near
exp(-1) ~ 0.37.

Run a quick CPU smoke test:
    python code/directed_diffusion_map.py --n 1500 --device cpu --k 50

Full-scale run on a SLURM GPU node:
    srun --gres=gpu:1 --mem=64g python code/directed_diffusion_map.py
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Hyper-parameters                                                            #
# --------------------------------------------------------------------------- #
@dataclass
class DDMConfig:
    """All tunables for the wing-only diffusion map.

    The penalty weights are *variances* (they multiply squared differences); the
    builder folds sqrt(weight) into the feature matrix so plain L2 reproduces them.
    """
    # --- feature-construction weights ---
    channel_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)  # per (phi,theta,psi); e.g. (3,3,10)
    equalize_channels: bool = False    # if True, divide each channel by its own std first so the
                                       # three angles contribute equally (channel_weights then bias on top)
    endpoint_steps: int = 5            # first/last N time steps get extra weight
    endpoint_weight: float = 5.0       # variance multiplier on those steps
    derivative_weight: float = 0.1     # variance multiplier on the finite-diff derivative
    cycle_weight: float = 0.02         # variance multiplier on the start-vs-end gap

    # --- kernel bandwidth ---
    epsilon: float | None = None       # None -> auto (median sq. k-NN distance); else a fixed value
    epsilon_scale: float = 1.0         # multiplier applied to the auto-epsilon (tuning knob)

    # --- degeneracy guard (detect an epsilon too small -> shattered graph) ---
    degenerate_eig_tol: float = 1e-3   # a non-trivial eigenvalue within this of 1.0 = a separate component
    fail_on_degenerate: bool = True    # raise on a shattered spectrum (set False to only warn)

    # --- graph / spectral ---
    k: int = 100                       # nearest neighbours per node
    n_components: int = 16             # non-trivial diffusion coordinates returned
    knn_chunk: int = 2048              # query chunk size for the chunked k-NN
    lobpcg_niter: int = 200            # max iterations for torch.lobpcg on GPU
    lobpcg_tol: float = 1e-5

    device: str = "cuda"

    def channel_weight_tensor(self) -> torch.Tensor:
        return torch.tensor(self.channel_weights, dtype=torch.float32)


@dataclass
class DDMResult:
    eigenvalues: np.ndarray            # (n_components,) non-trivial eigenvalues, descending
    coordinates: np.ndarray           # (N, n_components) diffusion coords  lambda_i * psi_i
    eigenvectors_sym: np.ndarray      # (N, n_components) raw P_sym eigenvectors (orthonormal)
    trivial_eigenvalue: float         # the dropped constant-mode eigenvalue (should be ~1.0)
    epsilon: float = 0.0              # bandwidth actually used
    backend_eig: str = ""
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Step 1 — Augmented feature matrix                                           #
# --------------------------------------------------------------------------- #
def build_augmented_features(
    X: torch.Tensor,
    cfg: DDMConfig,
) -> torch.Tensor:
    """Build the (N, D) augmented matrix whose squared-L2 == the diffusion metric.

    X : (N, C, T) wing kinematics (C channels, T time steps).

    The metric is built from the wing angles alone — body kinematics never enter
    it. If cfg.equalize_channels, each channel is first divided by its own dataset
    std so the three angles contribute equally (otherwise the widest-swing channel,
    here θ, dominates the L2 distance); channel_weights then bias on top. Components
    concatenated along the feature axis:
      * Base kinematics  : X, per-channel weighted by sqrt(channel_weights) and with
                           the first/last `endpoint_steps` time steps weighted by
                           sqrt(endpoint_weight). -> C*T dims.
      * Derivative       : finite-diff dX/dt, * sqrt(derivative_weight). -> C*(T-1) dims.
      * Cycle gap        : X[...,0] - X[...,-1], * sqrt(cycle_weight). -> C dims.

    Epsilon is NOT applied here; it is a single global scalar folded in at the
    affinity step (step 3), which keeps k-NN ranking independent of bandwidth.
    """
    if X.dim() != 3:
        raise ValueError(f"X must be (N, C, T); got {tuple(X.shape)}")

    device, dtype = X.device, torch.float32
    X = X.to(dtype)
    N, C, T = X.shape

    # --- optional per-channel equalisation (unit-variance each angle) ---
    if cfg.equalize_channels:
        chan_std = X.std(dim=(0, 2), keepdim=True).clamp(min=1e-8)  # (1, C, 1)
        X = X / chan_std

    # --- per-channel weighting (sqrt so squared-L2 yields the variance weight) ---
    chan_w = cfg.channel_weight_tensor().to(device)            # (C,)
    if chan_w.numel() != C:
        raise ValueError(f"channel_weights has {chan_w.numel()} entries but X has {C} channels")
    chan_scale = chan_w.sqrt().view(1, C, 1)                   # (1, C, 1)

    # --- temporal endpoint weighting ---
    time_scale = torch.ones(T, dtype=dtype, device=device)     # (T,)
    e = min(cfg.endpoint_steps, T)
    if e > 0 and cfg.endpoint_weight != 1.0:
        w = float(cfg.endpoint_weight) ** 0.5
        time_scale[:e] = w
        time_scale[T - e:] = w
    time_scale = time_scale.view(1, 1, T)                      # (1, 1, T)

    base = (X * chan_scale * time_scale).reshape(N, C * T)     # (N, C*T)

    # --- derivative penalty (finite difference of raw X) ---
    dX = (X[:, :, 1:] - X[:, :, :-1]) * (cfg.derivative_weight ** 0.5)
    deriv = dX.reshape(N, C * (T - 1))                         # (N, C*(T-1))

    # --- cycle penalty (start vs end of the wingbeat) ---
    cycle = (X[:, :, 0] - X[:, :, -1]) * (cfg.cycle_weight ** 0.5)  # (N, C)

    aug = torch.cat([base, deriv, cycle], dim=1)               # (N, D)
    return aug.contiguous()


# --------------------------------------------------------------------------- #
# Step 2 — Sparse k-NN graph (chunked, pure torch, device-agnostic)          #
# --------------------------------------------------------------------------- #
def knn(aug: torch.Tensor, cfg: DDMConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact k-NN, fully on aug.device, via chunked brute force.

    Uses ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b so the heavy step is one matmul per
    query chunk; peak extra memory is O(chunk * N), never the full N*N matrix.

    Returns (dist2 (N,k), idx (N,k)) sorted nearest-first. Each point's own nearest
    neighbour is itself with distance 0 (kept; masked out where it matters).
    """
    n = aug.shape[0]
    k = min(cfg.k, n)
    sq = (aug * aug).sum(dim=1)                                # (N,)
    dist2 = aug.new_empty((n, k))
    idx = torch.empty((n, k), dtype=torch.long, device=aug.device)

    for s in range(0, n, cfg.knn_chunk):
        e = min(s + cfg.knn_chunk, n)
        q = aug[s:e]                                           # (c, D)
        # (c, N): squared distances from each query to all points.
        d2 = sq.unsqueeze(0) + (q * q).sum(1, keepdim=True) - 2.0 * (q @ aug.t())
        d2.clamp_(min=0.0)
        vals, ids = torch.topk(d2, k, dim=1, largest=False, sorted=True)
        dist2[s:e] = vals
        idx[s:e] = ids
    return dist2, idx


# --------------------------------------------------------------------------- #
# Step 3 — Sparse symmetric-normalised diffusion operator                     #
# --------------------------------------------------------------------------- #
def resolve_epsilon(dist2: torch.Tensor, idx: torch.Tensor, cfg: DDMConfig) -> float:
    """Pick the kernel bandwidth.

    If cfg.epsilon is set, return it (times epsilon_scale). Otherwise auto-scale to
    the median of the *non-self* k-NN squared distances, so the typical neighbour
    affinity is exp(-1) ~ 0.37 rather than underflowing to 0 in high dimension.
    """
    if cfg.epsilon is not None:
        return float(cfg.epsilon) * float(cfg.epsilon_scale)

    n = dist2.shape[0]
    self_mask = idx == torch.arange(n, device=idx.device).unsqueeze(1)  # exclude i==j (dist 0)
    nonself = dist2[~self_mask]
    if nonself.numel() == 0:
        raise RuntimeError("No non-self neighbours found; cannot auto-scale epsilon.")
    eps = float(nonself.median().item()) * float(cfg.epsilon_scale)
    return max(eps, 1e-12)


def build_sparse_operator(
    dist2: torch.Tensor,
    idx: torch.Tensor,
    n: int,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """From k-NN (dist2, idx) and bandwidth epsilon build the operator P_sym.

    affinity = exp(-dist2 / epsilon)
    W        = sparse k-NN affinity, symmetrised W <- (W + W^T)/2
    P_sym    = D^{-1/2} W D^{-1/2},  D = diag(row sums of the symmetrised W)

    Returns (P_sym sparse COO coalesced, dinv_sqrt (N,)). dinv_sqrt is needed to
    map P_sym eigenvectors back to the diffusion (random-walk) eigenvectors.
    """
    device = dist2.device
    k = dist2.shape[1]

    rows = torch.arange(n, device=device).repeat_interleave(k)  # (N*k,)
    cols = idx.reshape(-1)                                       # (N*k,)
    aff = torch.exp(-dist2.reshape(-1) / epsilon)               # (N*k,)

    indices = torch.stack([rows, cols], dim=0)
    W = torch.sparse_coo_tensor(indices, aff, (n, n)).coalesce()

    # Symmetrise: (W + W^T) / 2.
    Wt = W.transpose(0, 1).coalesce()
    W_sym = ((W + Wt) * 0.5).coalesce()

    # Degree (row sums) and D^{-1/2}.
    deg = torch.sparse.sum(W_sym, dim=1).to_dense()            # (N,)
    dinv_sqrt = deg.clamp(min=1e-12).rsqrt()                   # (N,)

    # P_sym values: scale each entry v_ij by dinv_sqrt[i] * dinv_sqrt[j].
    si = W_sym.indices()
    sv = W_sym.values()
    pv = sv * dinv_sqrt[si[0]] * dinv_sqrt[si[1]]
    P_sym = torch.sparse_coo_tensor(si, pv, (n, n)).coalesce()

    return P_sym, dinv_sqrt


# --------------------------------------------------------------------------- #
# Step 4 — Sparse eigendecomposition                                          #
# --------------------------------------------------------------------------- #
def _eig_lobpcg(P_sym: torch.Tensor, n_eig: int, cfg: DDMConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-n_eig eigenpairs of the sparse symmetric P_sym via torch.lobpcg (GPU)."""
    n = P_sym.shape[0]
    X0 = torch.randn(n, n_eig, device=P_sym.device, dtype=P_sym.dtype)
    eigvals, eigvecs = torch.lobpcg(
        P_sym, k=n_eig, X=X0, largest=True, niter=cfg.lobpcg_niter, tol=cfg.lobpcg_tol,
    )
    return eigvals, eigvecs


def _eig_scipy(P_sym: torch.Tensor, n_eig: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-n_eig eigenpairs via scipy.sparse.linalg.eigsh on CPU (robust fallback)."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh

    P = P_sym.cpu().coalesce()
    si = P.indices().numpy()
    sv = P.values().numpy()
    n = P.shape[0]
    P_csr = sp.csr_matrix((sv, (si[0], si[1])), shape=(n, n))
    # Symmetric in exact arithmetic; tiny COO round-off can break that — re-symmetrise.
    P_csr = (P_csr + P_csr.T) * 0.5

    vals, vecs = eigsh(P_csr, k=n_eig, which="LA")            # ascending
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    return torch.from_numpy(vals).float(), torch.from_numpy(vecs).float()


def diffusion_eigendecomp(
    P_sym: torch.Tensor,
    dinv_sqrt: torch.Tensor,
    cfg: DDMConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, str]:
    """Eigendecompose P_sym, drop the trivial constant mode, return diffusion coords.

    Requests n_components + 1 eigenpairs (the leading one is the trivial ~1 constant
    mode of a Markov operator). The diffusion coordinate for non-trivial mode i is
        Psi_i = lambda_i * (D^{-1/2} v_i)
    where v_i is the P_sym eigenvector (D^{-1/2} v_i is the right eigenvector of the
    random-walk matrix P = D^{-1} W).

    Returns (eigvals (k,), coords (N,k), eigvecs_sym (N,k), trivial_eigval, backend).
    """
    n_eig = cfg.n_components + 1
    backend = "scipy"
    if P_sym.is_cuda:
        try:
            eigvals_t, eigvecs_t = _eig_lobpcg(P_sym, n_eig, cfg)
            backend = "lobpcg"
        except Exception as exc:
            print(f"[ddm] torch.lobpcg failed ({exc}); falling back to scipy.eigsh on CPU.", flush=True)
            eigvals_t, eigvecs_t = _eig_scipy(P_sym, n_eig)
    else:
        eigvals_t, eigvecs_t = _eig_scipy(P_sym, n_eig)

    # Order descending and split off the trivial leading mode.
    order = torch.argsort(eigvals_t, descending=True)
    eigvals_t = eigvals_t[order]
    eigvecs_t = eigvecs_t[:, order]

    trivial_eigval = float(eigvals_t[0].item())
    eigvals = eigvals_t[1:n_eig]                               # (k,)
    eigvecs = eigvecs_t[:, 1:n_eig]                            # (N, k)

    # Map P_sym eigenvectors -> random-walk eigenvectors, then scale by eigenvalue.
    dinv_sqrt_cpu = dinv_sqrt.to(eigvecs.device).float()
    psi = dinv_sqrt_cpu.unsqueeze(1) * eigvecs                 # (N, k)
    coords = psi * eigvals.unsqueeze(0)                        # (N, k)

    return (
        eigvals.detach().cpu().numpy(),
        coords.detach().cpu().numpy(),
        eigvecs.detach().cpu().numpy(),
        trivial_eigval,
        backend,
    )


# --------------------------------------------------------------------------- #
# Degeneracy guard                                                             #
# --------------------------------------------------------------------------- #
def assess_spectrum_health(
    eigvals: np.ndarray,
    eig_tol: float = 1e-3,
) -> tuple[int, str]:
    """Flag a shattered graph (an epsilon chosen too small) from the spectrum.

    For the symmetric-normalised diffusion operator the eigenvalue 1 has
    multiplicity equal to the number of connected components. The trivial constant
    mode (one eigenvalue == 1) is already split off upstream, so ANY *non-trivial*
    eigenvalue still within `eig_tol` of 1 means the weighted graph has broken into
    >1 effectively-isolated component — the signature of a bandwidth so small that
    the bridging edges' exp(-d^2/eps) affinity underflowed to ~0.

    Returns (n_degenerate, message). n_degenerate == 0 → healthy (empty message);
    otherwise `message` is a ready-to-print multi-line WARNING.
    """
    ev = np.asarray(eigvals, dtype=np.float64)
    n_degenerate = int((ev > 1.0 - eig_tol).sum())
    if n_degenerate == 0:
        return 0, ""
    lam1 = float(ev[0])
    msg = (
        f"[ddm] WARNING: DEGENERATE spectrum — {n_degenerate} non-trivial eigenvalue(s) "
        f"within {eig_tol:g} of 1.0 (lambda_1={lam1:.6f}).\n"
        f"       The diffusion operator has eigenvalue 1 once per connected component, so the "
        f"graph has shattered into ~{n_degenerate + 1} effectively-isolated pieces.\n"
        f"       This almost always means epsilon is too small (bridging affinities underflowed "
        f"to ~0). Re-run with a larger --epsilon_scale (e.g. 20) or a fixed --epsilon.\n"
        f"       The returned coordinates are near-constant component indicators, NOT smooth "
        f"diffusion axes — do not trust them."
    )
    return n_degenerate, msg


# --------------------------------------------------------------------------- #
# Orchestrator                                                                 #
# --------------------------------------------------------------------------- #
def directed_diffusion_map(X: torch.Tensor, cfg: DDMConfig) -> DDMResult:
    """Run the full wing-only diffusion-map pipeline on X and return a DDMResult."""
    n = X.shape[0]

    aug = build_augmented_features(X, cfg)
    print(f"[ddm] augmented features: {tuple(aug.shape)} on {aug.device}", flush=True)

    dist2, idx = knn(aug, cfg)
    print(f"[ddm] kNN: k={dist2.shape[1]}  dist2 range "
          f"[{dist2.min().item():.4g}, {dist2.max().item():.4g}]", flush=True)

    epsilon = resolve_epsilon(dist2, idx, cfg)
    mode = "auto" if cfg.epsilon is None else "fixed"
    print(f"[ddm] epsilon ({mode}): {epsilon:.6g}", flush=True)

    P_sym, dinv_sqrt = build_sparse_operator(dist2, idx, n, epsilon)
    print(f"[ddm] P_sym: {tuple(P_sym.shape)}  nnz={P_sym._nnz()}", flush=True)

    eigvals, coords, eigvecs, trivial, eig_backend = diffusion_eigendecomp(P_sym, dinv_sqrt, cfg)
    print(f"[ddm] eig ({eig_backend}): trivial lambda_0={trivial:.4f}  "
          f"top non-trivial lambda_1={eigvals[0]:.4f}  lambda_k={eigvals[-1]:.4f}", flush=True)

    # Guard: an epsilon too small shatters the graph -> many non-trivial eigenvalues ~1.0.
    n_degenerate, health_msg = assess_spectrum_health(eigvals, cfg.degenerate_eig_tol)
    if n_degenerate:
        print(health_msg, flush=True)
        if cfg.fail_on_degenerate:
            raise RuntimeError(
                f"Degenerate diffusion spectrum: {n_degenerate} non-trivial eigenvalue(s) ~1.0 "
                f"(shattered graph from too-small epsilon). Raise --epsilon_scale or set a fixed "
                f"--epsilon. Pass --no_fail_on_degenerate (fail_on_degenerate=False) to warn only."
            )

    return DDMResult(
        eigenvalues=eigvals,
        coordinates=coords,
        eigenvectors_sym=eigvecs,
        trivial_eigenvalue=trivial,
        epsilon=epsilon,
        backend_eig=eig_backend,
        extra={"n": n, "n_features": aug.shape[1]},
    )


# --------------------------------------------------------------------------- #
# Persistence                                                                  #
# --------------------------------------------------------------------------- #
def save_result(
    result: DDMResult,
    path: str,
    cfg: DDMConfig | None = None,
    color_features: dict[str, np.ndarray] | None = None,
) -> None:
    """Persist the embedding to .npz (NumPy) or .pt (PyTorch); format from extension.

    Saves `coordinates` and `eigenvalues` (plus the raw P_sym eigenvectors) together
    with provenance — the bandwidth used, the eigensolver backend, and the config
    that produced them — so a loaded embedding is self-describing.

    `color_features` is an optional mapping name -> per-wingbeat scalar array (N,),
    aligned row-for-row to `coordinates` (the pipeline never reorders rows). These are
    the physical quantities the embedding can be coloured by downstream
    (plot_ddm_embedding.py); stored as `color_<name>` plus a `color_feature_names` list.
    """
    out_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(out_dir, exist_ok=True)

    n_rows = result.coordinates.shape[0]
    color_features = color_features or {}
    color_arrays: dict[str, np.ndarray] = {}
    for name, arr in color_features.items():
        a = np.asarray(arr, dtype=np.float32).reshape(-1)
        if a.shape[0] != n_rows:
            raise ValueError(
                f"color feature {name!r} has {a.shape[0]} rows but embedding has {n_rows}"
            )
        color_arrays[name] = a
    color_names = list(color_arrays.keys())

    meta = {
        "trivial_eigenvalue": float(result.trivial_eigenvalue),
        "epsilon":            float(result.epsilon),
        "backend_eig":        result.backend_eig,
        "n":                  int(result.extra.get("n", n_rows)),
        "n_features":         int(result.extra.get("n_features", 0)),
        "n_components":       int(result.coordinates.shape[1]),
    }
    if cfg is not None:
        meta.update({
            "k":                 int(cfg.k),
            "channel_weights":   list(cfg.channel_weights),
            "equalize_channels": bool(cfg.equalize_channels),
            "endpoint_steps":    int(cfg.endpoint_steps),
            "endpoint_weight":   float(cfg.endpoint_weight),
            "derivative_weight": float(cfg.derivative_weight),
            "cycle_weight":      float(cfg.cycle_weight),
            "epsilon_scale":     float(cfg.epsilon_scale),
        })

    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        np.savez_compressed(
            path,
            coordinates=result.coordinates,
            eigenvalues=result.eigenvalues,
            eigenvectors_sym=result.eigenvectors_sym,
            color_feature_names=np.array(color_names),
            **{f"color_{k}": v for k, v in color_arrays.items()},
            **meta,
        )
    elif ext == ".pt":
        torch.save(
            {
                "coordinates":      torch.from_numpy(result.coordinates),
                "eigenvalues":      torch.from_numpy(result.eigenvalues),
                "eigenvectors_sym": torch.from_numpy(result.eigenvectors_sym),
                "color_features":   {k: torch.from_numpy(v) for k, v in color_arrays.items()},
                **meta,
            },
            path,
        )
    else:
        raise ValueError(f"Unsupported output extension {ext!r}; use .npz or .pt")
    print(f"[ddm] saved embedding → {path}"
          + (f"  (+{len(color_names)} colour features)" if color_names else ""), flush=True)


def default_color_features(X: torch.Tensor, F: torch.Tensor) -> dict[str, np.ndarray]:
    """Per-wingbeat physical scalars to colour the embedding by (diagnostics only).

    None of these enter the embedding — they are post-hoc overlays. From the body
    labels F: one channel per angular-acceleration component (named yaw/pitch/roll
    when C==3, generic otherwise); these let you check whether the wing-only
    embedding organises by body response. From the kinematics X: the peak-to-peak
    amplitude of each wing-angle channel (stroke φ, deviation θ, rotation ψ when
    C==3). All returned as (N,) numpy arrays aligned to the rows.
    """
    feats: dict[str, np.ndarray] = {}
    C = F.shape[1]
    label_names = ["yaw_accel", "pitch_accel", "roll_accel"] if C == 3 else [f"F{i}_accel" for i in range(C)]
    for i, name in enumerate(label_names):
        feats[name] = F[:, i].detach().cpu().numpy()

    amp_names = ["phi_amplitude", "theta_amplitude", "psi_amplitude"] if X.shape[1] == 3 \
        else [f"ch{i}_amplitude" for i in range(X.shape[1])]
    amplitude = (X.amax(dim=2) - X.amin(dim=2))                 # (N, C): peak-to-peak per channel
    for i, name in enumerate(amp_names):
        feats[name] = amplitude[:, i].detach().cpu().numpy()
    return feats


def load_single_wing_dataset(path: str, device: str) -> tuple[torch.Tensor, dict[str, np.ndarray]]:
    """Load real single-wing wingbeats to embed with the wing-only diffusion map.

    Reads the autoencoder dataset's `single_wing_wingbeats` (N, 3, T) — stroke φ,
    deviation θ, rotation ψ — which is the *only* thing that builds the embedding.

    Returns (X, color_features). The colour features are post-hoc diagnostics that
    never enter the metric: the per-wingbeat (yaw, pitch, roll) maneuver scores in
    [0,1] (body-response proxies, to ask "does the wing-only embedding organise by
    body response?") plus the peak-to-peak amplitude of each wing angle.
    """
    z = np.load(path, allow_pickle=True)
    X = torch.from_numpy(np.asarray(z["single_wing_wingbeats"], dtype=np.float32)).to(device)
    scores = np.asarray(z["maneuver_scores"], dtype=np.float32)        # (N, 3): yaw, pitch, roll
    feats: dict[str, np.ndarray] = {
        "yaw_maneuver":   scores[:, 0],
        "pitch_maneuver": scores[:, 1],
        "roll_maneuver":  scores[:, 2],
    }
    amp = (X.amax(dim=2) - X.amin(dim=2)).detach().cpu().numpy()       # (N, 3): peak-to-peak
    for i, nm in enumerate(["phi_amplitude", "theta_amplitude", "psi_amplitude"]):
        feats[nm] = amp[:, i]
    return X, feats


# --------------------------------------------------------------------------- #
# Main / smoke test                                                           #
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description="Sparse wing-only Diffusion Map (PyTorch).")
    p.add_argument("--data", default="data/autoencoder_dataset/wingbeats_single_wing_L69.npz",
                   help="single-wing wingbeat .npz to embed; 'dummy' for random smoke-test data")
    p.add_argument("--n", type=int, default=56624, help="number of dummy samples (--data dummy only)")
    p.add_argument("--channels", type=int, default=3)
    p.add_argument("--timesteps", type=int, default=69)
    p.add_argument("--k", type=int, default=100, help="nearest neighbours per node")
    p.add_argument("--n_components", type=int, default=16)
    p.add_argument("--equalize_channels", action="store_true",
                   help="divide each wing angle by its std so φ/θ/ψ contribute equally (else θ dominates)")
    p.add_argument("--epsilon", default="auto",
                   help="'auto' (median sq. k-NN distance) or a fixed float bandwidth")
    p.add_argument("--epsilon_scale", type=float, default=1.0,
                   help="multiplier on the (auto or fixed) epsilon")
    p.add_argument("--no_fail_on_degenerate", dest="fail_on_degenerate", action="store_false",
                   help="downgrade the degenerate-spectrum guard to a warning; default is to EXIT "
                        "non-zero on a shattered graph (multiple eigenvalues ~1.0 from too-small epsilon)")
    p.set_defaults(fail_on_degenerate=True)
    p.add_argument("--degenerate_eig_tol", type=float, default=1e-3,
                   help="a non-trivial eigenvalue within this of 1.0 counts as a separate component")
    p.add_argument("--out", default="data/analysis/ddm_embedding.npz",
                   help="where to save coordinates + eigenvalues (.npz or .pt); '' to skip")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available (login node?). Use --device cpu or an srun GPU node.")
    print(f"[ddm] device={device}", flush=True)

    epsilon = None if str(args.epsilon).lower() == "auto" else float(args.epsilon)

    torch.manual_seed(args.seed)

    cfg = DDMConfig(
        k=args.k,
        n_components=args.n_components,
        equalize_channels=args.equalize_channels,
        epsilon=epsilon,
        epsilon_scale=args.epsilon_scale,
        fail_on_degenerate=args.fail_on_degenerate,
        degenerate_eig_tol=args.degenerate_eig_tol,
        device=device,
    )

    # --- Data: real single-wing wingbeats, or random smoke-test data ---
    if str(args.data).lower() in ("", "dummy", "none"):
        # X builds the embedding; F (body angular accels) only exercises the
        # colour-only diagnostic path so a dummy run mirrors the real one.
        X = torch.randn(args.n, args.channels, args.timesteps, device=device)
        F = torch.randn(args.n, args.channels, device=device)
        color_features = default_color_features(X, F)
        print(f"[ddm] dummy data: X={tuple(X.shape)} (colour features: randn)", flush=True)
    else:
        X, color_features = load_single_wing_dataset(args.data, device)
        print(f"[ddm] loaded {args.data}: X={tuple(X.shape)}  "
              f"colour features={list(color_features)}", flush=True)

    result = directed_diffusion_map(X, cfg)

    print()
    print("=== Wing-only Diffusion Map result ===")
    print(f"  samples            : {result.extra['n']}")
    print(f"  augmented features : {result.extra['n_features']}")
    print(f"  epsilon            : {result.epsilon:.6g}")
    print(f"  eig backend        : {result.backend_eig}")
    print(f"  trivial eigenvalue : {result.trivial_eigenvalue:.6f}  (expect ~1.0)")
    print(f"  eigenvalues        : {np.array2string(result.eigenvalues, precision=4)}")
    print(f"  coordinates shape  : {result.coordinates.shape}")

    if args.out:
        save_result(result, args.out, cfg, color_features=color_features)


if __name__ == "__main__":
    main()
