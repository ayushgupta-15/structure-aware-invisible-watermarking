"""
Phase 2 sanity check: embed a watermark with DwtDctSvd, extract it back with
no attack in between, and confirm (a) the watermark is invisible (high PSNR/SSIM)
and (b) the payload is recovered exactly.

Run from the `code/` directory with the venv activated:
    python src/baseline_test.py
"""

import cv2
import numpy as np
from imwatermark import WatermarkEncoder, WatermarkDecoder
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

COVER_IMAGE_PATH = "data/test_image.png"
WATERMARKED_OUTPUT_PATH = "results/watermarked_test_image.png"
PAYLOAD_TEXT = "DTU-ECE-25"  # arbitrary short payload for the sanity check


def make_test_image_if_missing(path: str, size: int = 512) -> None:
    """Create a synthetic test image if no real dataset image is available yet."""
    import os

    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rng = np.random.default_rng(seed=42)
    # A mix of flat and textured regions, so later structure-aware tests have
    # something meaningful to react to (top half flat, bottom half noisy).
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[: size // 2, :, :] = 180  # flat region
    noise = rng.integers(0, 255, size=(size // 2, size, 3), dtype=np.uint8)
    img[size // 2 :, :, :] = noise  # textured region
    cv2.imwrite(path, img)
    print(f"No test image found — generated a synthetic placeholder at {path}")
    print("Replace this with a real DIV2K sample before running actual experiments.")


def embed_watermark(cover_path: str, out_path: str, payload_text: str) -> None:
    image = cv2.imread(cover_path)
    if image is None:
        raise FileNotFoundError(f"Could not read cover image at {cover_path}")

    encoder = WatermarkEncoder()
    encoder.set_watermark("bytes", payload_text.encode("utf-8"))
    watermarked = encoder.encode(image, "dwtDctSvd")
    cv2.imwrite(out_path, watermarked)


def extract_watermark(watermarked_path: str, payload_length_bytes: int) -> str:
    image = cv2.imread(watermarked_path)
    decoder = WatermarkDecoder("bytes", payload_length_bytes * 8)
    recovered = decoder.decode(image, "dwtDctSvd")
    return recovered.decode("utf-8", errors="replace")


def main() -> None:
    make_test_image_if_missing(COVER_IMAGE_PATH)

    embed_watermark(COVER_IMAGE_PATH, WATERMARKED_OUTPUT_PATH, PAYLOAD_TEXT)

    cover = cv2.imread(COVER_IMAGE_PATH)
    watermarked = cv2.imread(WATERMARKED_OUTPUT_PATH)

    quality_psnr = psnr(cover, watermarked)
    quality_ssim = ssim(cover, watermarked, channel_axis=2)

    recovered_text = extract_watermark(WATERMARKED_OUTPUT_PATH, len(PAYLOAD_TEXT.encode("utf-8")))

    print("=== Phase 2 baseline sanity check ===")
    print(f"Payload embedded : {PAYLOAD_TEXT!r}")
    print(f"Payload recovered: {recovered_text!r}")
    print(f"Exact match      : {recovered_text == PAYLOAD_TEXT}")
    print(f"PSNR (cover vs watermarked): {quality_psnr:.2f} dB")
    print(f"SSIM (cover vs watermarked): {quality_ssim:.4f}")


if __name__ == "__main__":
    main()
