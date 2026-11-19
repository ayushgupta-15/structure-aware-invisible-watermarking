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

## Status

- [x] Phase 1 — literature review and scope
- [x] Phase 2 — baseline DwtDctSvd sanity check (37.56 dB PSNR, exact recovery)
- [x] Phase 3 — structure-aware variants (variance, edge density); ~53 dB PSNR, exact recovery without attack
- [x] Phase 4a — conventional attacks (JPEG, noise, blur)
- [x] Phase 5 — evaluation on 20 DIV2K validation images (`src/run_experiments.py`, `results/conventional_results.csv`, `results/ber_comparison.png`)
- [x] Phase 4b — diffusion regeneration attack run on a Colab T4 (10 images, SD v1.5, strengths 0.2/0.4/0.6): `results/diffusion_results.csv`
- [x] Phase 6 — report: `docs/final_project_report.pdf`

## Reproduce

```bash
python3 -m venv venv && source venv/bin/activate   # see Setup below for install order
python src/prepare_dataset.py --n 20               # downloads DIV2K valid (450 MB), crops 20 images
python src/run_experiments.py                      # conventional attacks -> results/conventional_results.csv
python src/make_figures.py                         # needs matplotlib
python src/run_experiments.py --diffusion --n 10   # GPU required; or use colab_diffusion.ipynb
```

## Results (20 DIV2K images, 80-bit payload, mean BER, lower is better)

| Attack | Uniform | Variance | Edge |
|---|---|---|---|
| None | 0.000 | 0.000 | 0.000 |
| JPEG q50 | 0.413 | 0.401 | 0.353 |
| JPEG q30 | 0.499 | 0.505 | 0.471 |
| Noise s=10 | 0.553 | 0.517 | 0.471 |
| Noise s=20 | 0.519 | 0.533 | 0.489 |
| Blur k=5 | 0.193 | 0.189 | 0.149 |
| Blur k=7 | 0.296 | 0.282 | 0.274 |

Edge-driven embedding has the lowest mean BER everywhere, but gains are modest and only
statistically clear for some attacks; all variants are near chance at strong JPEG/noise. The

**Diffusion regeneration (10 images, mean BER):** every variant is at chance level at every
strength (Uniform 0.524/0.503/0.499, Variance 0.511/0.491/0.496, Edge 0.500/0.476/0.510 for
strength 0.2/0.4/0.6), differences are well within per-image noise. Structure-aware embedding gave
no measurable benefit against diffusion regeneration for this classical DWT-DCT-SVD scheme.

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

**Alternative: running it on a Mac instead of Colab.** `regenerate()` auto-detects
the best available device (`cuda` > `mps` > `cpu`), so this also works locally on
a Mac — no flags needed:

```bash
pip install diffusers transformers accelerate torch scikit-image opencv-python-headless PyWavelets
python src/phase4_diffusion_experiment.py
```

- **Apple Silicon (M1/M2/M3/M4):** uses the `mps` (Metal) backend — noticeably
  faster than CPU, though not as fast as a real CUDA GPU.
- **Intel Mac:** no GPU backend available, falls back to CPU — works, but slow
  (minutes per image).
- Needs roughly 6-10 GB free disk for the model + libraries, and 16 GB+ RAM is
  recommended (8 GB will likely swap heavily).
- Safe to run either way — worst case is slow, not harmful to the machine.
