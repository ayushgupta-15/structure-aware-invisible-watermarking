"""
Phase 3 — our own DWT-DCT-SVD watermark embedder/extractor.

We can't use the `invisible-watermark` library for the structure-aware
variant because it embeds with one fixed strength for the whole image and
exposes no way to vary that strength per block. So this is a from-scratch
implementation of the same embedding family (DWT -> block DCT -> SVD),
small enough to reason about, with an explicit per-block strength that a
structure map can drive.

Pipeline, matching what's described in the project report:
  1. One-level 2D DWT on a single channel -> LL, (LH, HL, HH)
  2. Split LL into non-overlapping blocks
  3. Per block: 2D DCT -> SVD -> encode one payload bit via quantization
     index modulation (QIM) on the gap between the top two singular values
  4. Inverse DCT, reassemble LL, inverse DWT

Bit encoding (QIM on singular values):
  d = S[0] - S[1]                      (always >= 0, since SVD sorts S desc.)
  q = round(d / strength)
  bit 1 -> force q odd; bit 0 -> force q even
  new_d = q * strength
  rescale S[0], S[1] to hit new_d while preserving their sum (minimizes
  the distortion introduced for a given strength)

`strength` is the quantization step: larger = more robust, more visible.
This is exactly the knob the structure map scales per block.
"""

import cv2
import numpy as np
import pywt

BLOCK_SIZE = 8       # block size in the LL subband
WAVELET = "haar"
CHANNEL = 0          # BGR -> blue channel (least perceptually sensitive)


def text_to_bits(text: str) -> list[int]:
    data = text.encode("utf-8")
    bits = []
    for byte in data:
        bits.extend([(byte >> i) & 1 for i in range(7, -1, -1)])
    return bits


def bits_to_text(bits: list[int]) -> str:
    byte_vals = []
    for i in range(0, len(bits) - 7, 8):
        byte = 0
        for b in bits[i:i + 8]:
            byte = (byte << 1) | b
        byte_vals.append(byte)
    return bytes(byte_vals).decode("utf-8", errors="replace")


def _dct2(block: np.ndarray) -> np.ndarray:
    return cv2.dct(block.astype(np.float32))


def _idct2(block: np.ndarray) -> np.ndarray:
    return cv2.idct(block)


def _embed_bit_in_block(block: np.ndarray, bit: int, strength: float) -> np.ndarray:
    dct_block = _dct2(block)
    U, S, Vt = np.linalg.svd(dct_block)

    total = S[0] + S[1]
    # S[1]_new must stay >= S[2], or the matrix's *true* top two singular
    # values after reconstruction won't be the ones we just set — silently
    # breaking both embedding and extraction. This caps how large a gap we
    # may force onto this block.
    floor = S[2] if len(S) > 2 else 0.0
    max_d = max(total - 2 * floor, 0.0)

    d = S[0] - S[1]
    target_q = round(d / strength)

    chosen_q = None
    for delta in range(0, 200):
        for q in ({target_q} if delta == 0 else {target_q + delta, target_q - delta}):
            if q < 0 or q % 2 != bit:
                continue
            if q * strength <= max_d:
                chosen_q = q
                break
        if chosen_q is not None:
            break
    if chosen_q is None:
        # Essentially-flat block where even q=bit (smallest possible gap)
        # doesn't fit — fall back to the smallest valid gap of the right
        # parity, accepting it may not round-trip under heavy distortion.
        chosen_q = bit

    new_d = chosen_q * strength
    S[0] = (total + new_d) / 2
    S[1] = (total - new_d) / 2

    watermarked_dct = (U * S) @ Vt
    return _idct2(watermarked_dct)


def _extract_bit_from_block(block: np.ndarray, strength: float) -> int:
    dct_block = _dct2(block)
    S = np.linalg.svd(dct_block, compute_uv=False)
    d = S[0] - S[1]
    q = round(d / strength)
    return q % 2


def _block_grid(ll_shape: tuple[int, int], block_size: int) -> tuple[int, int]:
    return ll_shape[0] // block_size, ll_shape[1] // block_size


def embed(image: np.ndarray, payload_text: str, base_strength: float,
          strength_map: np.ndarray | None = None) -> np.ndarray:
    """
    strength_map: optional 2D array matching the LL block grid, each entry
    a multiplier applied to base_strength for that block (uniform embedding
    if None — this is Experiment 1).
    """
    bits = text_to_bits(payload_text)

    channel = image[:, :, CHANNEL].astype(np.float32)
    LL, (LH, HL, HH) = pywt.dwt2(channel, WAVELET)

    rows, cols = _block_grid(LL.shape, BLOCK_SIZE)
    if len(bits) > rows * cols:
        raise ValueError(f"Payload needs {len(bits)} blocks but only {rows * cols} are available.")

    for idx, bit in enumerate(bits):
        r, c = divmod(idx, cols)
        block = LL[r * BLOCK_SIZE:(r + 1) * BLOCK_SIZE, c * BLOCK_SIZE:(c + 1) * BLOCK_SIZE]
        strength = base_strength * (strength_map[r, c] if strength_map is not None else 1.0)
        strength = max(strength, 1e-3)  # guard against a zero-strength block
        LL[r * BLOCK_SIZE:(r + 1) * BLOCK_SIZE, c * BLOCK_SIZE:(c + 1) * BLOCK_SIZE] = \
            _embed_bit_in_block(block, bit, strength)

    watermarked_channel = pywt.idwt2((LL, (LH, HL, HH)), WAVELET)
    watermarked_channel = np.clip(watermarked_channel, 0, 255)

    out = image.copy()
    out[:, :, CHANNEL] = watermarked_channel[:image.shape[0], :image.shape[1]].astype(np.uint8)
    return out


def extract_bits(image: np.ndarray, num_bits: int, base_strength: float,
                  strength_map: np.ndarray | None = None) -> list[int]:
    channel = image[:, :, CHANNEL].astype(np.float32)
    LL, _ = pywt.dwt2(channel, WAVELET)

    _, cols = _block_grid(LL.shape, BLOCK_SIZE)
    bits = []
    for idx in range(num_bits):
        r, c = divmod(idx, cols)
        block = LL[r * BLOCK_SIZE:(r + 1) * BLOCK_SIZE, c * BLOCK_SIZE:(c + 1) * BLOCK_SIZE]
        strength = base_strength * (strength_map[r, c] if strength_map is not None else 1.0)
        strength = max(strength, 1e-3)
        bits.append(_extract_bit_from_block(block, strength))
    return bits


def extract(image: np.ndarray, num_bits: int, base_strength: float,
            strength_map: np.ndarray | None = None) -> str:
    return bits_to_text(extract_bits(image, num_bits, base_strength, strength_map))


def bit_error_rate(original_bits: list[int], recovered_bits: list[int]) -> float:
    """Fraction of bits that differ. 0.0 = perfect recovery, 0.5 = random guessing."""
    assert len(original_bits) == len(recovered_bits)
    wrong = sum(1 for a, b in zip(original_bits, recovered_bits) if a != b)
    return wrong / len(original_bits)
