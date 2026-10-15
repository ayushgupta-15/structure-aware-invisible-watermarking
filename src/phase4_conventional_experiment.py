"""
Phase 4a — conventional attacks vs. the three embedding variants from Phase 3.

This is the "sanity check against classical attacks" half of Phase 4 — the
comparison matrix from the project report, minus the diffusion-regeneration
row (see phase4_diffusion_attack.py / README for why that row needs a cloud
GPU rather than this machine).

For every embedding variant (uniform, variance-driven, edge-driven) and
every attack condition (none, JPEG, noise, blur), we measure:
  - PSNR/SSIM of the attacked image vs. the original watermarked image
    (how much the attack itself visibly changed the image)
  - Bit Error Rate of the recovered payload vs. the original payload bits
    (0.0 = perfect recovery, 0.5 = watermark effectively destroyed)
"""

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from structure_maps import local_variance_map, edge_density_map
from svd_watermark import embed, extract_bits, text_to_bits, bit_error_rate, BLOCK_SIZE
from attacks import jpeg_compress, gaussian_noise, gaussian_blur

COVER_IMAGE_PATH = "data/test_image.png"
PAYLOAD_TEXT = "DTU-ECE-25"
BASE_STRENGTH = 50.0
ALPHA = 1.0
STRUCTURE_BLOCK_SIZE = BLOCK_SIZE * 2

ATTACKS = {
    "None": lambda img: img,
    "JPEG q50": lambda img: jpeg_compress(img, quality=50),
    "JPEG q30": lambda img: jpeg_compress(img, quality=30),
    "Noise sigma=10": lambda img: gaussian_noise(img, sigma=10),
    "Noise sigma=20": lambda img: gaussian_noise(img, sigma=20),
    "Blur k=5": lambda img: gaussian_blur(img, ksize=5),
    "Blur k=7": lambda img: gaussian_blur(img, ksize=7),
}


def main() -> None:
    img = cv2.imread(COVER_IMAGE_PATH)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    original_bits = text_to_bits(PAYLOAD_TEXT)
    num_bits = len(original_bits)

    variance_map = 1.0 + ALPHA * local_variance_map(gray, STRUCTURE_BLOCK_SIZE)
    edge_map = 1.0 + ALPHA * edge_density_map(gray, STRUCTURE_BLOCK_SIZE)

    variants = {
        "Uniform": None,
        "Variance": variance_map,
        "Edge": edge_map,
    }

    watermarked = {name: embed(img, PAYLOAD_TEXT, BASE_STRENGTH, smap) for name, smap in variants.items()}

    header = f"{'Variant':<10s} {'Attack':<16s} {'PSNR (dB)':<11s} {'SSIM':<8s} {'BER':<7s} {'Exact':<6s}"
    print(header)
    print("-" * len(header))

    rows = []
    for variant_name, wm_img in watermarked.items():
        strength_map = variants[variant_name]
        for attack_name, attack_fn in ATTACKS.items():
            attacked = attack_fn(wm_img)
            recovered_bits = extract_bits(attacked, num_bits, BASE_STRENGTH, strength_map)
            ber = bit_error_rate(original_bits, recovered_bits)
            exact = ber == 0.0
            if attack_name == "None":
                p_str, s_str = "n/a", "1.0000"
            else:
                p_str = f"{psnr(wm_img, attacked):.2f}"
                s_str = f"{ssim(wm_img, attacked, channel_axis=2):.4f}"
            rows.append((variant_name, attack_name, p_str, s_str, ber, exact))
            print(f"{variant_name:<10s} {attack_name:<16s} {p_str:<11s} {s_str:<8s} {ber:<7.3f} {str(exact):<6s}")

    print("\nBER: 0.0 = perfect recovery, 0.5 = watermark effectively destroyed (random guessing).")


if __name__ == "__main__":
    main()
