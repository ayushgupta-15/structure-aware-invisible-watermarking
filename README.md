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
- [x] Phase 4a — conventional attacks (JPEG, Gaussian noise, Gaussian blur) — verified
      working, real comparison table across all three embedding variants (see below)
- [ ] Phase 4b — diffusion regeneration attack — code complete (`src/phase4_diffusion_attack.py`,
      `src/phase4_diffusion_experiment.py`), **not run yet** — needs a GPU and ~6 GB free disk
      this machine doesn't have (6.5 GB free total; already hit a quota crisis once from a much
      smaller download in Phase 2). Needs to run on Colab — see below.
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

## Conventional attacks (Phase 4a)

`src/attacks.py` — JPEG recompression, Gaussian noise, Gaussian blur. Run the full
comparison (all three embedding variants × all attack conditions, with PSNR/SSIM/BER)
with `python src/phase4_conventional_experiment.py` from `code/`.

Real result, one run: JPEG compression is the one conventional attack where the
embedding variants actually differ — variance-driven embedding recovered more bits
correctly than uniform or edge-driven at both JPEG quality levels tested (q50: BER
0.263 vs 0.338/0.362; q30: BER 0.475 vs 0.575/0.487). Under noise, all three variants
converge toward ~0.5 BER (random guessing) at higher noise levels — no variant
survives that well. Under blur, all three variants perform identically, no
differentiation at all. This is consistent with the literature we cited
(content-adaptive embedding helps against conventional attacks like JPEG, but the
effect isn't guaranteed to show up under every kind of conventional distortion) —
and it's the sanity check that justifies trusting the more important, still-unrun
diffusion-regeneration comparison below.

## Diffusion regeneration attack (Phase 4b) — run on Colab, not here

`src/phase4_diffusion_attack.py` implements the actual attack (SDEdit-style: add
partial noise, run the diffusion model's denoiser, matching how Zhao et al. 2023
and the WAVES benchmark describe this attack class). `src/phase4_diffusion_experiment.py`
runs it against all three Phase 3 embedding variants and reports BER — this is the
experiment the whole project's research question depends on, and it hasn't been run
yet because this machine doesn't have the disk or GPU for it.

To run on Colab:

```python
!pip install diffusers transformers accelerate torch scikit-image opencv-python-headless PyWavelets
# upload this repo's code/ directory (or git clone it), then:
%cd code
!python src/phase4_diffusion_experiment.py
```

Uses a free Colab GPU runtime (T4 is enough) and Colab's own disk, not this
machine's — no training involved, inference only, with an existing pretrained
Stable Diffusion checkpoint (`runwayml/stable-diffusion-v1-5` by default).
