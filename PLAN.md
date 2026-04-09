# Nmotion Ultraplan — Ground-Up Refactor

## Context

Nmotion is being refactored from a web-based skeleton-pose clinical prototype into a
research pipeline that takes neonatal videos → extracts dense optical flow → computes
nonlinear dynamics features → produces matplotlib comparison figures across three
movement groups (normal, spasms/seizures, hypertonia).

Everything else (Streamlit, FastAPI, RL pipeline, YOLO-Pose, Gemini cache, user auth,
dashboards, pages/) is dead weight and will be stripped.

---

## Research Findings (April 2026 SOTA)

### Optical Flow

| Model | Venue | Spring EPE | vs. Prior | Speed | Notes |
|---|---|---|---|---|---|
| **RAFT** | ECCV 2020 | baseline | — | 1x | In torchvision. 6 years old. |
| **SEA-RAFT** | ECCV 2024 Oral | 3.69 | −22.9% | 2.3x faster | Princeton-vl. Best paper candidate. Mature repo. |
| **FlowSeek** | ICCV 2025 | — | −10-15% vs SEA-RAFT | trains on 1 GPU | Requires Depth Anything v2 as additional component. Newest SOTA. |

**Decision: Start with SEA-RAFT.** It's the best tradeoff of accuracy, speed, maturity,
and simplicity. FlowSeek adds a depth foundation model dependency for marginal gains on
benchmarks that don't map to our controlled-camera neonatal setting. **Recursive
checkpoint (RC-1)** after first flow extraction: visually inspect flow quality on neonatal
video. If flow fails on occluded/blanketed infants, evaluate FlowSeek.

### Feature Extraction

The proposed feature set is well-validated in current literature:

| Feature | What it measures | Why it discriminates | Library |
|---|---|---|---|
| **Sample entropy** | Regularity/predictability of motion time series | Seizures = quasi-periodic (low), normal = complex (high), hypotonic = low but different amplitude | `antropy` |
| **Multiscale entropy** | Entropy at multiple temporal coarsening scales | Seizures: low at seizure freq, high elsewhere. Hypotonic: uniformly low across all scales | `antropy` |
| **Spectral entropy** | Entropy of normalized power spectral density | Seizures concentrate power in narrow bands (1-3Hz clonic). Normal = diffuse | `scipy` + manual |
| **DFA exponent (α)** | Long-range temporal correlations | Normal: anti-persistent (α<0.5). Seizures: persistent (α>0.5). Hypotonic: ~white noise | `antropy` |
| **Approximate symmetry index** | Left/right flow magnitude comparison across midline | Focal seizures break symmetry. Generalized seizures + normal ≈ symmetric | Manual from flow field |
| **Flow magnitude distribution** | Per-pixel motion energy statistics | Hypotonic clusters near zero. Seizures show high-amplitude peaks. Normal is intermediate | `numpy` |
| **Kinetic energy** | Mean squared flow magnitude per frame | Already in existing pipeline, still valid | `numpy` |
| **Peak frequency** | Dominant frequency in motion power spectrum | Clonic seizures: 1-3Hz. Tonic-clonic: 0.5-1Hz. Normal: no dominant peak | `scipy.signal` |

**Library choice: `antropy` (v0.2.1, March 2026).** Numba-JIT'd, includes sample_entropy,
multiscale_entropy, detrended_fluctuation. Actively maintained. Replaces `nolds`.

**Dropped:** `sarnat_score` (clinical assessment, not computable from motion data).

### Visualization

Output: matplotlib figures + pandas DataFrames saved to `output/`. No dashboards.

Target figures:
1. **KDE of flow magnitude** — one curve per group, overlaid
2. **Multiscale entropy profiles** — entropy vs. scale, per group with CI bands
3. **Power spectral density** — log-log, averaged per group
4. **Phase space plots** — velocity vs. acceleration, colored by group
5. **DFA log-log plots** — showing different scaling exponents per group

---

## New Project Structure

```
Nmotion/
├── pipeline/
│   ├── __init__.py
│   ├── flow_extract.py      # SEA-RAFT inference: video → .npy flow fields
│   ├── features.py          # flow fields → feature DataFrames
│   └── visualize.py         # DataFrames → matplotlib figures
├── run.py                   # CLI: python run.py --video-dir data/videos/
├── requirements.txt         # New, minimal deps
├── trecvit/                 # Keep as-is (separate concern, JAX-based)
├── data/
│   └── videos/              # Neonatal videos organized by group
│       ├── normal/
│       ├── spasms/
│       └── hypertonia/
└── output/                  # Generated figures + dataframes
    ├── figures/
    └── dataframes/
```

### What Gets Deleted

- `server/` — entire directory (api.py, models.py, physics_engine.py, gemini_cache.py,
  storage.py, yolo_inference.py, trajectory_generator.py, visualize_motion.py,
  case_search.py, analyze_logs.py, analysis_logs/, rl_pipeline/)
- `pages/` — all Streamlit pages
- `charts.py`, `streamlit_app.py`, `app.py` — web UI
- `.streamlit/` — Streamlit config
- `misc/` — old docs
- `pyrightconfig.json` — old config
- `Nmotion/` (the empty venv) — replaced by trecvit_env
- `todo.py` — empty file

### What Stays

- `trecvit/` — TRecViT model (separate JAX concern, may be useful later)
- `trecvit_env/` — the combined venv (will add new deps to it)
- `data/` — empty dir, will hold videos
- `.git/`, `.gitignore` — version control

---

## Implementation Plan

### Phase 0: Environment Setup
- [ ] Install PyTorch + torchvision with CUDA 12.8 into `trecvit_env`
- [ ] Install `antropy>=0.2.0`, `matplotlib`, `pandas`, `scipy` into `trecvit_env`
- [ ] Clone SEA-RAFT repo or install as dependency
- [ ] Write new minimal `requirements.txt`
- [ ] Verify: `import torch; torch.cuda.is_available()` and `import jax; jax.devices()`
  both see GPU

> **RC-0 (Recursive Checkpoint):** If PyTorch CUDA 12.8 and JAX CUDA 13 conflict in the
> same venv (NCCL/CUDA runtime collisions), split into two venvs: `venv-flow/` (torch)
> and `trecvit_env/` (jax). This is unlikely since they ship bundled CUDA libs, but check.

### Phase 1: Flow Extraction (`pipeline/flow_extract.py`)
- [ ] Download SEA-RAFT pretrained weights (SEA-RAFT-S or SEA-RAFT-M)
- [ ] Write `extract_flow(video_path) → np.ndarray` — loads video, runs SEA-RAFT on
  consecutive frame pairs, returns flow fields `[N_frames-1, H, W, 2]`
- [ ] Handle video I/O with `torchvision.io` or `cv2.VideoCapture`
- [ ] Save flow fields as `.npy` files to `output/flows/`
- [ ] Write batch processing: iterate over `data/videos/{group}/*.mp4`

> **RC-1:** After running on first real neonatal videos, visually inspect flow fields
> (quiver plot or HSV colorwheel). If flow quality is poor on occluded/blanketed regions:
> - Option A: Try SEA-RAFT-L (larger backbone)
> - Option B: Evaluate FlowSeek (adds Depth Anything v2, better cross-domain)
> - Option C: Pre-process with background subtraction before flow estimation

### Phase 2: Feature Extraction (`pipeline/features.py`)
- [ ] Write `flow_to_timeseries(flow_fields) → dict[str, np.ndarray]` — converts per-frame
  flow fields into 1D time series (flow magnitude mean, max, std per frame)
- [ ] Implement feature extractors:
  - `compute_sample_entropy(ts)` — via `antropy.sample_entropy`
  - `compute_multiscale_entropy(ts, scales)` — via `antropy.multiscale_entropy` or manual
    coarsening + sample_entropy at each scale
  - `compute_spectral_entropy(ts)` — normalize PSD from `scipy.signal.welch`, compute
    Shannon entropy of normalized spectrum
  - `compute_dfa(ts)` — via `antropy.detrended_fluctuation`
  - `compute_symmetry_index(flow_fields)` — split at vertical midline, compare
    L1/L2 norms of left vs right flow magnitudes
  - `compute_peak_frequency(ts, fps)` — dominant peak from `scipy.signal.welch`
  - `compute_kinetic_energy(flow_fields)` — mean squared flow magnitude per frame
- [ ] Write `extract_all_features(video_dir, group_name) → pd.DataFrame` — one row per
  video, columns = features
- [ ] Save DataFrames as CSV to `output/dataframes/`

> **RC-2:** After computing features on all three groups, run quick Kruskal-Wallis test
> per feature. If any feature shows p > 0.1 across all groups, it has no discriminative
> power — drop it. If the feature set is too correlated, consider PCA or drop redundant
> features. If a feature the literature says should discriminate doesn't, investigate
> whether the flow extraction or time-series aggregation is the bottleneck (go back to
> RC-1).

### Phase 3: Visualization (`pipeline/visualize.py`)
- [ ] Figure 1: **KDE of flow magnitude** — `sns.kdeplot` or `scipy.stats.gaussian_kde`,
  one curve per group overlaid, with legend
- [ ] Figure 2: **Multiscale entropy profiles** — line plot, entropy (y) vs scale (x),
  one line per group, shaded CI bands (bootstrap or SEM)
- [ ] Figure 3: **Power spectral density** — `plt.loglog`, averaged PSD per group,
  annotate seizure frequency bands
- [ ] Figure 4: **Phase space plots** — scatter of velocity vs acceleration, colored by
  group. Seizures → limit cycles, normal → diffuse, hypotonic → origin cluster
- [ ] Figure 5: **DFA log-log plots** — log(fluctuation) vs log(scale), one line per
  group, annotate α exponents
- [ ] All figures saved to `output/figures/` as both PNG (300dpi) and PDF

> **RC-3:** After generating all figures, assess visual separation. If groups overlap
> heavily on all plots:
> - Re-examine flow extraction quality (RC-1)
> - Consider additional features: permutation entropy, fractal dimension,
>   recurrence quantification analysis
> - Consider temporal windowing: compute features on sliding windows rather than
>   full-video aggregates

### Phase 4: CLI Entry Point (`run.py`)
- [ ] Argparse: `--video-dir`, `--output-dir`, `--skip-flow` (use cached .npy),
  `--groups` (subset of groups to process)
- [ ] Orchestrate: flow extraction → feature computation → visualization
- [ ] Print summary table of features per group to stdout

---

## Dependencies (New `requirements.txt`)

```
# Optical flow
torch>=2.4.0
torchvision>=0.19.0
# SEA-RAFT cloned separately or vendored

# Feature extraction
antropy>=0.2.0
scipy>=1.10.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.4.0

# Visualization
matplotlib>=3.8.0

# Video I/O
opencv-python-headless>=4.8.0
```

No streamlit, no fastapi, no mlflow, no stable-baselines3, no pettingzoo.

---

## Recursive Checkpoint Summary

| Checkpoint | Trigger | Action |
|---|---|---|
| **RC-0** | PyTorch + JAX CUDA runtime conflict | Split into two venvs |
| **RC-1** | Poor flow quality on neonatal video | Try SEA-RAFT-L → FlowSeek → preprocessing |
| **RC-2** | Feature has no discriminative power (p>0.1) | Drop feature or investigate flow quality |
| **RC-3** | All figures show overlapping groups | Re-examine flow → add features → try windowing |

Each checkpoint feeds back to earlier phases. The pipeline is designed to fail fast and
surface problems early rather than building elaborate infrastructure on untested
assumptions.

---

## Estimated New File Count

| File | Lines (est.) | Purpose |
|---|---|---|
| `pipeline/__init__.py` | 5 | Package marker |
| `pipeline/flow_extract.py` | ~120 | SEA-RAFT inference |
| `pipeline/features.py` | ~200 | All feature extractors |
| `pipeline/visualize.py` | ~250 | All 5 figure generators |
| `run.py` | ~80 | CLI entry point |
| `requirements.txt` | ~15 | Dependencies |

**Total: ~670 lines of new code.** That's the whole pipeline.
