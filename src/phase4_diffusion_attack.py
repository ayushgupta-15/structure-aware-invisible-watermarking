"""
Phase 4b — diffusion-based regeneration attack.

NOT RUN ON THIS MACHINE. This machine has ~6.5 GB free disk and no GPU; a
Stable Diffusion checkpoint alone is 4-7 GB, and we already hit a disk-quota
crisis once from a much smaller dependency (Phase 2's torch install). This
matches what the README has said since Phase 1: the diffusion attack step
runs on a cloud GPU (e.g. Google Colab), at inference time only — no model
is trained here or there.

This script is complete and correct as written; it just needs an
environment with the disk/GPU to run `pip install diffusers transformers
accelerate` and download a pretrained checkpoint. On Colab: new notebook,
enable a GPU runtime, upload this repo (or just this file plus
svd_watermark.py / structure_maps.py / attacks.py), run.

Attack definition (SDEdit-style regeneration, matching Zhao et al. 2023 /
the WAVES benchmark's description of this attack class):
  watermarked image -> add partial noise (strength controls how much) ->
  run the diffusion model's denoising process -> regenerated image

`strength` (0-1) is the img2img noise fraction: 0 = no change, 1 = ignores
the input entirely and generates from scratch. Regeneration attacks
typically use a moderate strength (0.3-0.6) — enough to disrupt an
embedded signal while keeping the image recognizably the same scene, which
is exactly what makes this attack class different from a conventional one
that visibly degrades the image.
"""

import argparse

import cv2
import numpy as np


def pick_device() -> str:
    """cuda (Colab/Linux+NVIDIA) > mps (Apple Silicon) > cpu (everything else)."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def regenerate(image_bgr: np.ndarray, strength: float = 0.4, steps: int = 30,
                model_id: str = "runwayml/stable-diffusion-v1-5",
                device: str | None = None) -> np.ndarray:
    """
    Run one SDEdit-style regeneration pass over a BGR uint8 image.
    Requires: pip install diffusers transformers accelerate torch

    device=None auto-detects: cuda > mps (Apple Silicon) > cpu.
    """
    import torch
    from diffusers import StableDiffusionImg2ImgPipeline
    from PIL import Image

    if device is None:
        device = pick_device()
    # float16 is only reliably correct on CUDA — on MPS it can silently
    # produce black/garbled output, and CPU doesn't support it at all.
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Using device={device}, dtype={dtype}"
          + ("  (no GPU found — this will be slow, minutes per image)" if device == "cpu" else ""))

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=False)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb).resize((512, 512))

    # An empty prompt lets the model reconstruct the input's own content
    # rather than steering it toward something else — this is the
    # regeneration/purification setting, not creative editing.
    result = pipe(
        prompt="",
        image=pil_image,
        strength=strength,
        num_inference_steps=steps,
        guidance_scale=1.0,
    ).images[0]

    result_rgb = np.array(result)
    result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
    return cv2.resize(result_bgr, (image_bgr.shape[1], image_bgr.shape[0]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/exp1_uniform.png")
    parser.add_argument("--output", default="results/diffusion_regenerated.png")
    parser.add_argument("--strength", type=float, default=0.4)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--device", default=None,
                         help="cuda, mps, or cpu — omit to auto-detect")
    args = parser.parse_args()

    img = cv2.imread(args.input)
    if img is None:
        raise FileNotFoundError(args.input)

    regenerated = regenerate(img, strength=args.strength, steps=args.steps, device=args.device)
    cv2.imwrite(args.output, regenerated)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
