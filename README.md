# headon-flight

Trying out a multi-head approach to solve the inverse flight-dynamics problem for
fruit-fly wingbeats: a wingbeat **autoencoder** (AE) learns a compact latent code
for wing motion, and a downstream **body→latent regressor** predicts that code
from body kinematics.

This README is the operating guide for the full data + training pipeline.

---

## 1. The pipeline at a glance

```
data/unprocessed_data/<experiment>/      raw  *_analysis_smoothed.h5   (one folder per experiment)
        │   process_data.py        — condense + L/R-mirror augment; output auto-named from the input folder
        ▼
data/processed/<experiment>/             condensed *_condensed_data.h5  (one folder per experiment)
        │   transform_data.py      — gather ALL experiment folders, build the training arrays
        ▼
data/autoencoder_dataset/                   trajectories.npy · body_kinematics.npy · golden_template.npy · wingbeats_L69.npz
        │   autoencoder.py         — train the AE (GPU, via sbatch)
        ▼
data/models/autoencoder[/_single_wing]   best_autoencoder.pt · best_config.json · grid_search_summary.json · val_indices.json · eval/
        │   body_latent_regressor.py  — (downstream) train body→latent regressor on top of a trained AE
        ▼
data/models/body_latent_regressor/
```

There are **no symlinks** and **no manual steps** between stages. Provenance of
every wingbeat is its experiment **subfolder name** under `data/processed/`.

---

## 2. Environment

- Python venv: **`.env`** — `source .env/bin/activate` (or call `.env/bin/python`).
- The login node has **no GPU**; training runs on a GPU node via **`sbatch`**.
  Data condensing / dataset building / evaluation run fine on CPU.
- Run every command **from the project root** (`/cs/labs/tsevi/lior.kotlar/headon-flight`).

---

## 3. Quick start — train an autoencoder end to end

```bash
# 1. condense every experiment under data/unprocessed_data/  (auto-detects single vs many)
python code/data_handling/process_data.py

# 2. build the training dataset from all condensed experiments
python code/transform_data.py --config code/autoencoder_config.json --fixed_len 69

# 3. train on a GPU node  (single-wing: append  --representation single_wing)
sbatch -J autoencoder sbatch_train_autoencoder.sh code/autoencoder_config.json

# 4. evaluate the trained model
python code/evaluate_autoencoder.py --model_dir data/models/autoencoder
```

Watch a job: `squeue -u $USER` · live log: `tail -f logs/autoencoder_<jobid>.out`.

---

## 4. Stage-by-stage

### Stage 0 (optional) — vet a new batch before adding it
Always sanity-check a freshly recorded experiment against the established data:
```bash
python code/compare_new_data.py \
  --new_dir data/unprocessed_data/<new_experiment> \
  --old_dir data/unprocessed_data/old_data \
  --out_dir data/analysis/compare_<new_experiment>
```
Reads `--old_dir`/`--new_dir` non-recursively (point them at specific experiment
folders). Writes 7 figures + `comparison_summary.json` + `GUIDE.txt`. The verdict
flags *distributional difference* (KS effect size), not necessarily "bad" — read
the GUIDE. Use it to catch contamination (high asymmetry, template blow-up) vs. a
genuinely different-but-clean flight regime.

### Stage 1 — condense raw → `data/processed/<experiment>/`
`process_data.py` recomputes body kinematics, NaN-trims wings, and writes an
L/R-mirror-augmented copy per movie. The output folder is **auto-named** from the
input folder, and only that experiment's folder is cleared (others untouched).
```bash
# all experiments (bare):
python code/data_handling/process_data.py
# one experiment:
python code/data_handling/process_data.py --unprocessed_dir data/unprocessed_data/roni_60ms
# custom output parent (default data/processed):
python code/data_handling/process_data.py --output_root data/processed
```
**Auto-detect:** if `--unprocessed_dir` holds `*_analysis_smoothed.h5` directly it
is one experiment; otherwise each immediate subdirectory is its own experiment.

### Stage 2 — build the dataset
`transform_data.py` gathers **all** condensed files under `data/processed/`
(recursively, path-sorted), applies the 10×-median L/R-asymmetry garbage filter,
builds the golden template, and writes the training arrays.
```bash
python code/transform_data.py --config code/autoencoder_config.json --fixed_len 69
```
Input dir resolution (precedence): `--processed_dir` arg ▸ the config's
`processed_dir` key (`data/processed`) ▸ the global default. Outputs go to the
config's `data_path` / `template_path` (`data/autoencoder_dataset/...`).

### Stage 3 — train the autoencoder (GPU)
One config trains either representation; `--representation` overrides it and
auto-suffixes the save dir so the two models never collide.
```bash
sbatch -J autoencoder             sbatch_train_autoencoder.sh code/autoencoder_config.json
sbatch -J autoencoder_single_wing sbatch_train_autoencoder.sh code/autoencoder_config.json --representation single_wing
# locally (CPU, slow): python code/autoencoder.py --config code/autoencoder_config.json [--representation single_wing]
```
- 6-channel **S/A** model → `data/models/autoencoder/`
- 3-channel **single-wing** model → `data/models/autoencoder_single_wing/`

`auto_build_dataset: true` rebuilds `wingbeats_L<L>.npz` automatically if the
trajectories/template changed (md5-checked), so step 2 is optional if only the
model hyperparameters changed.

### Stage 4 — evaluate
```bash
python code/evaluate_autoencoder.py --model_dir data/models/autoencoder
```
Produces reconstruction plots, per-phase error, and a per-maneuver-bucket report
(from the model's own saved val split). Useful flags: `--no_bucket_eval` (just the
reconstruction plot), `--score_axis yaw pitch roll`, `--n_beats N`, `--seed N`,
`--phase_ranges`, `--npz_path <wingbeats.npz>`.

---

## 5. The config — `code/autoencoder_config.json`

One file controls a run. Key fields:

| field | meaning |
|-------|---------|
| `data_path` / `template_path` | where the built dataset + template live (`data/autoencoder_dataset/...`) |
| `processed_dir` | where Stage 2 reads condensed files (`data/processed`) |
| `save_dir` | model output root (`data/models/autoencoder`; single-wing auto-suffixes `_single_wing`) |
| `representation` | `sa` (default) or `single_wing`; overridden by the `--representation` CLI flag |
| `fixed_len` | wingbeat resample length L (69) |
| `latent_dim`, `random_seed` | **list ⇒ grid sweep** (currently `16` × 5 seeds → best-of-5) |
| `n_epochs`, `batch_size`, `lr`, `dropout`, `base_channels`, … | training hyperparameters |

To grid-search, make a field a list (e.g. `"latent_dim": [4, 8, 12, 16]`).

---

## 6. Adding or removing an experiment

- **Add:** drop the raw folder in `data/unprocessed_data/<name>/`, (optionally vet
  with Stage 0), then `process_data.py --unprocessed_dir data/unprocessed_data/<name>`
  and re-run Stage 2. It joins the dataset automatically.
- **Remove:** delete `data/processed/<name>/` and re-run Stage 2.
- An experiment is in the dataset **iff** it has a folder under `data/processed/`.

> `amitai_dark_disturbance/` raw exists but is intentionally **not** in the dataset
> (a distinct dark-disturbance regime). Condense it in to include it.

---

## 7. Downstream — body→latent regressor

Trains a model that predicts the AE latent from body kinematics. It sits on top of
a trained AE (configs point at the AE model dir) and auto-builds its dataset
(`wingbeat_regressor_dataset*.npz`) by encoding wingbeats through that AE.
```bash
# configs: code/body_latent_regressor_config.json (full) · _pitch_config.json · _single_wing_config.json
sbatch -J regressor sbatch_train_regressor.sh code/body_latent_regressor_config.json
# local: python code/body_latent_regressor.py --config code/body_latent_regressor_config.json
# inference / decode latents back to wingbeats:
python code/body_to_wingbeat.py --regressor_dir data/models/body_latent_regressor --autoencoder_dir data/models/autoencoder
```
Each regressor config's `autoencoder_model_dir` must point at the AE run you want
it built on (currently the combined-data `data/models/autoencoder/autoencoder_<ts>`).
After retraining the AE, update those paths and rebuild.

---

## 8. Directory reference

| path | what |
|------|------|
| `data/unprocessed_data/<exp>/` | raw `*_analysis_smoothed.h5`, one folder per experiment |
| `data/processed/<exp>/` | condensed `*_condensed_data.h5` (+ augmented), one folder per experiment |
| `data/autoencoder_dataset/` | built AE dataset (`trajectories.npy`, `wingbeats_L69.npz`, templates) + `PROVENANCE.md` |
| `data/regressor_dataset/` | built regressor dataset(s) (`wingbeat_regressor_dataset_*.npz`), auto-built from a trained AE |
| `data/models/autoencoder[/_single_wing]/` | trained AEs |
| `data/models/body_latent_regressor/` | trained regressors |
| `data/analysis/` | golden template, comparison reports, eval plots |
| `code/autoencoder.py` · `evaluate_autoencoder.py` | AE train / eval |
| `code/data_handling/process_data.py` · `transform_data.py` | condense / build dataset |
| `code/data_handling/build_regressor_dataset.py` · `body_latent_regressor.py` · `body_to_wingbeat.py` | regressor build / train / inference |
| `code/compare_new_data.py` | vet a new batch vs the established data |
| `sbatch_train_autoencoder.sh` · `sbatch_train_regressor.sh` | GPU job submission |

See `data/autoencoder_dataset/PROVENANCE.md` for the exact experiments currently in
the dataset and their movie/trajectory counts.
