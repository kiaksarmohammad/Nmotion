# Nmotion

Neonatal movement analysis research tool. No web UI, no production infra — output is matplotlib figures + pandas DataFrames.

---

## Pipeline

```
video files → [RAFT optical flow] → .npy flow fields → [feature extraction] → CSV + figures
```

### Stage 1: Flow Extraction (`pipeline/flow_extract.py`)
- torchvision **RAFT-Large** extracts dense optical flow from video frame pairs
- Input: `{video_dir}/{group}/*.mp4` (BGR, any resolution)
- Output: `output/flows/{group}/{stem}.npy` — float32 `[N-1, H, W, 2]` (u, v displacement)
- Caches results; skips if `.npy` exists

### Stage 2: Feature Extraction (`pipeline/features.py`)
- Converts flow fields → 1D time series (mean/max/std of per-pixel magnitude)
- Computes nonlinear dynamics features via `antropy`:
  - Sample entropy, multiscale entropy (20 scales), spectral entropy
  - DFA exponent, peak frequency
  - Kinetic energy (mean/std), flow magnitude stats (mean, std, skew, kurtosis)
  - Symmetry index (left/right energy ratio)
- Output: `output/dataframes/{group}_features.csv` + `all_features.csv`

### Stage 3: Visualization (`pipeline/visualize.py`)
- 5 figures (PNG 300dpi + PDF): flow magnitude KDE, multiscale entropy profiles, PSD, phase space, DFA exponents
- Group colors: normal=#2ecc71, spasms=#e74c3c, hypertonia=#3498db

### Entry point
```bash
python run.py --video-dir data/videos                # full pipeline
python run.py --video-dir data/videos --skip-flow    # cached flows only
```

---

## Current State

- **Early research stage** — limited test data so far (seizure videos only)
- Target taxonomy: seizure, hypotonic, normal (+ subtypes like clonic, myoclonic)
- **Next milestone**: statistical analysis to identify which features show significant group separation, before building any classifier

---

## Dependencies of Note

- **trecvit** (in-repo, Google-licensed): video transformer model. Not yet integrated into the pipeline — intended for future classification stage once discriminating features are identified.
- **antropy**: all entropy/DFA computations
- **torchvision RAFT**: optical flow backbone (may swap to SEA-RAFT or FlowSeek if quality is insufficient on neonatal video)

---

## Conventions

- Flow fields are always `[N, H, W, 2]` with channels = (u, v) displacement
- FPS stored as separate `{stem}_fps.npy` alongside flow files
- Groups are directory names under the video/flow dirs
