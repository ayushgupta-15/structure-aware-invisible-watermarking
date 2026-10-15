# Structure-Aware Invisible Image Watermarking

B.Tech major project (ECE, DTU) — investigating whether structure-aware watermark
embedding changes watermark detectability under diffusion-based image regeneration,
compared to uniform-strength embedding.

## Project layout

```
code/
  venv/          - Python virtual environment (not tracked in git)
  src/           - all source code
  data/          - test / dataset images (DIV2K subset goes here)
  results/       - watermarked images, logs, output metrics
  requirements.txt
```

## Setup

```bash
cd code
python3 -m venv venv
source venv/bin/activate
pip install numpy opencv-python-headless scikit-image PyWavelets Pillow
pip install invisible-watermark --no-deps
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Why not just `pip install -r requirements.txt`?** The `invisible-watermark` PyPI
package unconditionally imports `torch` at the top level (for its optional RivaGAN
deep-learning mode, which we don't use — we only use its `dwtDctSvd` method). A
plain `pip install invisible-watermark` pulls the full GPU/CUDA build of torch,
which is several hundred MB to multi-GB and can blow past disk quotas on a shared
machine. Installing it `--no-deps` and adding a small **CPU-only** torch build
separately avoids that entirely — this machine has no GPU anyway, so CPU-only torch
is all we need until the Phase 4 diffusion attack step, which will likely run on a
cloud GPU (e.g. Colab) rather than locally.

## Phase progress

- [x] Phase 1 — literature review and scope
- [x] Phase 2 — baseline implementation (DwtDctSvd embed/extract, sanity check) — verified working:
      37.56 dB PSNR, 0.9717 SSIM, exact payload recovery on a real photographic test image
      (no attack applied yet — this is the pre-attack sanity check only)
- [x] Phase 3 — structure-aware variants (local variance, edge density) — verified working:
      all three experiments (uniform / variance-driven / edge-driven) give exact payload
      recovery at ~54 dB PSNR, 0.9987 SSIM, no attack applied yet (see below)
- [ ] Phase 4 — attack pipeline (JPEG/noise/blur + diffusion regeneration)
- [ ] Phase 5 — evaluation (PSNR/SSIM/BER/NC across all conditions)
- [ ] Phase 6 — analysis and writeup

## Baseline method (Phase 2)

DwtDctSvd (from the open-source `invisible-watermark` library) — a transform-domain
watermarking scheme combining Discrete Wavelet Transform, Discrete Cosine Transform,
and Singular Value Decomposition. Used as-is as the reference baseline.

## Structure-aware method (Phase 3)

The `invisible-watermark` library embeds with one fixed strength for the whole
image and exposes no way to vary that per block, so Phase 3 is our own from-scratch
DWT → block-DCT → SVD embedder (`src/svd_watermark.py`), small enough to reason
about, with an explicit per-block strength that a structure map can drive.

- **Embedding domain:** one-level Haar DWT on the blue channel, blocks of the LL
  subband (8×8), one payload bit per block via quantization index modulation (QIM)
  on the gap between the top two singular values of each block's DCT.
- **Structure maps** (`src/structure_maps.py`): local pixel variance and Canny
  edge density, each computed per 16×16 block on the full-resolution image (2×
  the LL block size, since one level of Haar DWT halves resolution — this keeps
  the two grids aligned block-for-block) and normalized to [0, 1].
- **Minimum viable scope:** Experiment 1 (uniform), Experiment 2 (variance-driven
  strength), Experiment 3 (edge-driven strength). Each scales the QIM step size
  per block as `base_strength * (1 + alpha * structure_score)`.
- **A real bug worth knowing about:** naively setting the two target singular
  values from a QIM step can push the second one below the block's *third*
  singular value, silently breaking the ordering SVD relies on — both embedding
  and extraction go quietly wrong for that block. Fixed by capping the achievable
  gap per block so the chosen singular values never cross that floor.
- **Minimum reliable strength:** swept `base_strength` from 5 to 200 — below 40,
  basic 8-bit pixel quantization (just saving the PNG, not an attack) already
  corrupts a meaningful fraction of bits. 40+ gives exact recovery; we use 50 for
  margin. Run with `python src/phase3_experiment.py` from `code/`.
