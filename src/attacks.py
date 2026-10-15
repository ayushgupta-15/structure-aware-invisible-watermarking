"""
Phase 4 — conventional attacks.

Three standard image-processing distortions, used both as a sanity check
(do our embedders survive *ordinary* processing at all?) and as a point of
comparison against the diffusion-based regeneration attack: conventional
attacks are blind, mechanical operations with no understanding of image
content, which is exactly the contrast the project's research question is
built on.
"""

import cv2
import numpy as np


def jpeg_compress(image: np.ndarray, quality: int) -> np.ndarray:
    """Re-encode through JPEG at the given quality (1-100) and decode back."""
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def gaussian_noise(image: np.ndarray, sigma: float, seed: int = 0) -> np.ndarray:
    """Add zero-mean Gaussian noise with the given standard deviation (pixel units)."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, image.shape)
    noisy = image.astype(np.float64) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def gaussian_blur(image: np.ndarray, ksize: int) -> np.ndarray:
    """Gaussian blur with an odd kernel size (e.g. 3, 5, 7)."""
    if ksize % 2 == 0:
        raise ValueError("ksize must be odd")
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


if __name__ == "__main__":
    img = cv2.imread("data/test_image.png")
    if img is None:
        raise FileNotFoundError("Run this from the code/ directory: python src/attacks.py")

    jpeg_q50 = jpeg_compress(img, quality=50)
    noisy_s10 = gaussian_noise(img, sigma=10)
    blurred_k5 = gaussian_blur(img, ksize=5)

    cv2.imwrite("results/attack_jpeg_q50.png", jpeg_q50)
    cv2.imwrite("results/attack_noise_s10.png", noisy_s10)
    cv2.imwrite("results/attack_blur_k5.png", blurred_k5)
    print("Wrote results/attack_{jpeg_q50,noise_s10,blur_k5}.png")
