"""
Prepare a small DIV2K subset: resize shorter side to 512, center-crop to 512x512.

Usage (from code/):
    python src/prepare_dataset.py --zip DIV2K_valid_HR.zip --n 20
If --zip is omitted, the zip is downloaded from the official DIV2K mirror.
"""
import argparse
import io
import os
import urllib.request
import zipfile

import cv2
import numpy as np

URL = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"


def center_crop_512(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    scale = 512 / min(h, w)
    img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    top, left = (h - 512) // 2, (w - 512) // 2
    return img[top:top + 512, left:left + 512]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=None)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="data/div2k")
    args = ap.parse_args()

    zpath = args.zip
    if zpath is None:
        zpath = "DIV2K_valid_HR.zip"
        print("Downloading", URL)
        urllib.request.urlretrieve(URL, zpath)

    os.makedirs(args.out, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        names = sorted(n for n in z.namelist() if n.lower().endswith(".png"))[:args.n]
        for name in names:
            data = np.frombuffer(z.read(name), np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            out = os.path.join(args.out, os.path.basename(name))
            cv2.imwrite(out, center_crop_512(img))
            print("wrote", out)


if __name__ == "__main__":
    main()
