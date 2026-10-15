"""
Phase 3 — structure maps.

Two simple, classical, training-free measures of "how structured" a region
of an image is, each returned as a 2D array of per-block scores normalized
to [0, 1]. Flat regions (clear sky, plain wall) score near 0; busy, textured
regions (foliage, gravel, edges) score near 1.

These are used to scale watermark embedding strength per block in
svd_watermark.py — see Experiment 2 (variance-based) and Experiment 3
(edge-based) in phase3_experiment.py.
"""

import cv2
import numpy as np


def local_variance_map(gray: np.ndarray, block_size: int) -> np.ndarray:
    """Per-block pixel variance, normalized to [0, 1]."""
    h, w = gray.shape
    rows, cols = h // block_size, w // block_size
    scores = np.zeros((rows, cols), dtype=np.float64)
    for r in range(rows):
        for c in range(cols):
            block = gray[r * block_size:(r + 1) * block_size, c * block_size:(c + 1) * block_size]
            scores[r, c] = block.var()
    return _normalize(scores)


def edge_density_map(gray: np.ndarray, block_size: int) -> np.ndarray:
    """Per-block Canny edge-pixel density, normalized to [0, 1]."""
    edges = cv2.Canny(gray, threshold1=100, threshold2=200)
    h, w = gray.shape
    rows, cols = h // block_size, w // block_size
    scores = np.zeros((rows, cols), dtype=np.float64)
    for r in range(rows):
        for c in range(cols):
            block = edges[r * block_size:(r + 1) * block_size, c * block_size:(c + 1) * block_size]
            scores[r, c] = np.count_nonzero(block) / block.size
    return _normalize(scores)


def _normalize(scores: np.ndarray) -> np.ndarray:
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-9:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


if __name__ == "__main__":
    # Quick sanity check on the real test image.
    img = cv2.imread("data/test_image.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    var_map = local_variance_map(gray, block_size=16)
    edge_map = edge_density_map(gray, block_size=16)

    print("Local variance map:  shape", var_map.shape, "mean", round(var_map.mean(), 3))
    print("Edge density map:    shape", edge_map.shape, "mean", round(edge_map.mean(), 3))
    print("Correlation between the two maps:", round(np.corrcoef(var_map.flatten(), edge_map.flatten())[0, 1], 3))
