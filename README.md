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
- [ ] Phase 3 — structure-aware variants (local variance, edge density)
- [ ] Phase 4 — attack pipeline (JPEG/noise/blur + diffusion regeneration)
- [ ] Phase 5 — evaluation (PSNR/SSIM/BER/NC across all conditions)
- [ ] Phase 6 — analysis and writeup

## Baseline method

DwtDctSvd (from the open-source `invisible-watermark` library) — a transform-domain
watermarking scheme combining Discrete Wavelet Transform, Discrete Cosine Transform,
and Singular Value Decomposition. Used as-is here as the reference baseline; the
structure-aware variant (Phase 3) modifies embedding strength per block without
changing this core mechanism.
