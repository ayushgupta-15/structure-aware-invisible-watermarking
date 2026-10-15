"""
Phase 4b evaluation — run on Colab (GPU runtime), not on the dev machine.

This is the actual research-question experiment: does structure-aware
embedding change watermark survival under diffusion-based regeneration,
compared to uniform embedding? Everything up to this point (Phase 2-4a) was
building and sanity-checking the pieces; this script is where those pieces
answer the question the whole project is about.

Setup on Colab:
    !pip install diffusers transformers accelerate torch scikit-image opencv-python-headless PyWavelets
    # upload or git-clone this repo's code/src/ directory, then:
    %cd code
    !python src/phase4_diffusion_experiment.py

Uses the Phase 3 watermarked outputs (results/exp1_uniform.png,
exp2_variance.png, exp3_edge.png) as input, so Phase 3 must have been run
first (it has — those files are already committed).
"""

import sys

import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, "src")
from svd_watermark import extract_bits, text_to_bits, bit_error_rate, BLOCK_SIZE
from structure_maps import local_variance_map, edge_density_map
from phase4_diffusion_attack import regenerate

PAYLOAD_TEXT = "DTU-ECE-25"
BASE_STRENGTH = 50.0
ALPHA = 1.0
STRUCTURE_BLOCK_SIZE = BLOCK_SIZE * 2
COVER_IMAGE_PATH = "data/test_image.png"

VARIANTS = {
    "Uniform": ("results/exp1_uniform.png", None),
    "Variance": ("results/exp2_variance.png", "variance"),
    "Edge": ("results/exp3_edge.png", "edge"),
}

REGENERATION_STRENGTHS = [0.2, 0.4, 0.6]


def build_strength_map(kind: str | None):
    if kind is None:
        return None
    gray = cv2.cvtColor(cv2.imread(COVER_IMAGE_PATH), cv2.COLOR_BGR2GRAY)
    if kind == "variance":
        return 1.0 + ALPHA * local_variance_map(gray, STRUCTURE_BLOCK_SIZE)
    if kind == "edge":
        return 1.0 + ALPHA * edge_density_map(gray, STRUCTURE_BLOCK_SIZE)
    raise ValueError(kind)


def main() -> None:
    original_bits = text_to_bits(PAYLOAD_TEXT)
    num_bits = len(original_bits)

    header = f"{'Variant':<10s} {'Regen. strength':<16s} {'PSNR (dB)':<11s} {'SSIM':<8s} {'BER':<7s}"
    print(header)
    print("-" * len(header))

    for variant_name, (path, map_kind) in VARIANTS.items():
        strength_map = build_strength_map(map_kind)
        watermarked = cv2.imread(path)
        if watermarked is None:
            raise FileNotFoundError(f"{path} — run phase3_experiment.py first")

        for regen_strength in REGENERATION_STRENGTHS:
            regenerated = regenerate(watermarked, strength=regen_strength, device="cuda")
            recovered_bits = extract_bits(regenerated, num_bits, BASE_STRENGTH, strength_map)
            ber = bit_error_rate(original_bits, recovered_bits)
            p = psnr(watermarked, regenerated)
            s = ssim(watermarked, regenerated, channel_axis=2)
            print(f"{variant_name:<10s} {regen_strength:<16.1f} {p:<11.2f} {s:<8.4f} {ber:<7.3f}")

            out_path = f"results/diffusion_{variant_name.lower()}_s{regen_strength}.png"
            cv2.imwrite(out_path, regenerated)

    print("\nBER: 0.0 = perfect recovery, 0.5 = watermark effectively destroyed (random guessing).")
    print("This table is the core result for the research question — compare BER across")
    print("variants at each regeneration strength to see whether structure-awareness helped,")
    print("hurt, or made no difference.")


if __name__ == "__main__":
    main()
