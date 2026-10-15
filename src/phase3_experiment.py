"""
Phase 3 — Experiments 1-3: uniform vs. structure-aware embedding, no attack yet.

This is the pre-attack sanity check for the structure-aware variants, same
spirit as Phase 2's baseline_test.py: confirm each variant embeds invisibly
and extracts exactly, before Phase 4 introduces any attack. The real
research question (does structure-awareness change survival *under
diffusion regeneration*) can only be answered once Phase 4 exists — this
script only establishes that all three variants are correctly implemented
and usable as a fair starting point for that comparison.

Experiment 1: uniform embedding (strength_map = None)
Experiment 2: local-variance-driven embedding strength
Experiment 3: edge-density-driven embedding strength
"""

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from structure_maps import local_variance_map, edge_density_map
from svd_watermark import embed, extract, text_to_bits, BLOCK_SIZE

COVER_IMAGE_PATH = "data/test_image.png"
PAYLOAD_TEXT = "DTU-ECE-25"
BASE_STRENGTH = 50.0
ALPHA = 1.0  # multiplier range becomes [1, 1+ALPHA] across the structure map

# Structure maps are computed on the full-resolution image, so their block
# size must be 2x the LL-domain BLOCK_SIZE (one level of Haar DWT halves
# resolution) for the grids to align exactly, block for block.
STRUCTURE_BLOCK_SIZE = BLOCK_SIZE * 2


def run_experiment(name: str, img: np.ndarray, num_bits: int, strength_map) -> dict:
    watermarked = embed(img, PAYLOAD_TEXT, BASE_STRENGTH, strength_map)
    recovered = extract(watermarked, num_bits, BASE_STRENGTH, strength_map)
    return {
        "name": name,
        "exact_match": recovered == PAYLOAD_TEXT,
        "recovered": recovered,
        "psnr": psnr(img, watermarked),
        "ssim": ssim(img, watermarked, channel_axis=2),
        "watermarked": watermarked,
    }


def main() -> None:
    img = cv2.imread(COVER_IMAGE_PATH)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    num_bits = len(text_to_bits(PAYLOAD_TEXT))

    variance_scores = local_variance_map(gray, STRUCTURE_BLOCK_SIZE)
    edge_scores = edge_density_map(gray, STRUCTURE_BLOCK_SIZE)

    variance_strength_map = 1.0 + ALPHA * variance_scores
    edge_strength_map = 1.0 + ALPHA * edge_scores

    results = [
        run_experiment("Experiment 1 - Uniform", img, num_bits, None),
        run_experiment("Experiment 2 - Local variance", img, num_bits, variance_strength_map),
        run_experiment("Experiment 3 - Edge density", img, num_bits, edge_strength_map),
    ]

    print(f"{'Experiment':<28s} {'Exact match':<12s} {'PSNR (dB)':<11s} {'SSIM':<8s}")
    print("-" * 60)
    for r in results:
        print(f"{r['name']:<28s} {str(r['exact_match']):<12s} {r['psnr']:<11.2f} {r['ssim']:<8.4f}")

    cv2.imwrite("results/exp1_uniform.png", results[0]["watermarked"])
    cv2.imwrite("results/exp2_variance.png", results[1]["watermarked"])
    cv2.imwrite("results/exp3_edge.png", results[2]["watermarked"])
    print("\nWatermarked outputs saved to results/exp{1,2,3}_*.png")


if __name__ == "__main__":
    main()
