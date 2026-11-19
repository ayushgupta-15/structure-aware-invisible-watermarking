"""
Phase 5 — full evaluation over a dataset folder.

For every image: embed with 3 variants (uniform / variance / edge), apply
conventional attacks and (optionally) diffusion regeneration, and record
PSNR, SSIM, BER and NC (normalized correlation) per (image, variant, attack).
Results go to a CSV; a mean-per-(variant, attack) summary is printed.

Usage (from code/):
    python src/run_experiments.py                    # conventional attacks only
    python src/run_experiments.py --diffusion --n 10 # + diffusion (needs GPU)
"""
import argparse
import csv
import glob
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, os.path.dirname(__file__))
from attacks import jpeg_compress, gaussian_noise, gaussian_blur
from structure_maps import local_variance_map, edge_density_map
from svd_watermark import embed, extract_bits, text_to_bits, bit_error_rate, BLOCK_SIZE

PAYLOAD_TEXT = "DTU-ECE-25"
BASE_STRENGTH = 50.0
ALPHA = 1.0
STRUCT_BLOCK = BLOCK_SIZE * 2

CONVENTIONAL = {
    "None": lambda i: i,
    "JPEG q50": lambda i: jpeg_compress(i, 50),
    "JPEG q30": lambda i: jpeg_compress(i, 30),
    "Noise s=10": lambda i: gaussian_noise(i, 10),
    "Noise s=20": lambda i: gaussian_noise(i, 20),
    "Blur k=5": lambda i: gaussian_blur(i, 5),
    "Blur k=7": lambda i: gaussian_blur(i, 7),
}


def normalized_correlation(a, b) -> float:
    a = np.array(a, float) * 2 - 1
    b = np.array(b, float) * 2 - 1
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/div2k")
    ap.add_argument("--n", type=int, default=None, help="max images")
    ap.add_argument("--diffusion", action="store_true")
    ap.add_argument("--regen-strengths", type=float, nargs="+", default=[0.2, 0.4, 0.6])
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.data, "*.png")))[:args.n]
    if not paths:
        sys.exit(f"No PNG images in {args.data}")
    os.makedirs(args.out, exist_ok=True)

    attacks = dict(CONVENTIONAL)
    if args.diffusion:
        from phase4_diffusion_attack import regenerate
        for s in args.regen_strengths:
            attacks[f"Diffusion s={s}"] = (lambda i, s=s: regenerate(i, strength=s))

    bits = text_to_bits(PAYLOAD_TEXT)
    rows = []
    for k, path in enumerate(paths, 1):
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        maps = {
            "Uniform": None,
            "Variance": 1.0 + ALPHA * local_variance_map(gray, STRUCT_BLOCK),
            "Edge": 1.0 + ALPHA * edge_density_map(gray, STRUCT_BLOCK),
        }
        for vname, smap in maps.items():
            wm = embed(img, PAYLOAD_TEXT, BASE_STRENGTH, smap)
            emb_psnr = psnr(img, wm)
            emb_ssim = ssim(img, wm, channel_axis=2)
            for aname, fn in attacks.items():
                att = fn(wm)
                rec = extract_bits(att, len(bits), BASE_STRENGTH, smap)
                rows.append(dict(
                    image=os.path.basename(path), variant=vname, attack=aname,
                    embed_psnr=emb_psnr, embed_ssim=emb_ssim,
                    attack_psnr=psnr(wm, att) if aname != "None" else float("nan"),
                    attack_ssim=ssim(wm, att, channel_axis=2) if aname != "None" else 1.0,
                    ber=bit_error_rate(bits, rec), nc=normalized_correlation(bits, rec)))
        print(f"[{k}/{len(paths)}] {os.path.basename(path)} done", flush=True)

    tag = "diffusion" if args.diffusion else "conventional"
    csv_path = os.path.join(args.out, f"{tag}_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print("wrote", csv_path)

    agg = defaultdict(list)
    for r in rows:
        agg[(r["attack"], r["variant"])].append(r)
    print(f"\n{'Attack':<18s}{'Variant':<10s}{'BER mean':<10s}{'BER std':<9s}{'NC mean':<9s}")
    for (a, v), rs in agg.items():
        b = [r["ber"] for r in rs]
        print(f"{a:<18s}{v:<10s}{np.mean(b):<10.3f}{np.std(b):<9.3f}{np.mean([r['nc'] for r in rs]):<9.3f}")
    print(f"\nImages: {len(paths)}  |  Embed PSNR (mean): "
          f"{np.mean([r['embed_psnr'] for r in rows]):.2f} dB")


if __name__ == "__main__":
    main()
