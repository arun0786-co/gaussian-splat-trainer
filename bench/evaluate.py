"""Test-view metrics for a trained scene: PSNR, SSIM and LPIPS (VGG), the
three numbers the 3DGS paper reports, plus render speed. LPIPS is given
twice: with the inputs scaled to [-1, 1] as the lpips network expects, and
unscaled, as the 3DGS evaluation script computes it.

    python bench/evaluate.py --scene data/tandt/truck --ply output/truck/point_cloud.ply --out results/truck.json

LPIPS needs `pip install lpips`, which downloads VGG-16 weights once.
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from splat.device import default_device, sync  # noqa: E402
from splat.ply import load_ply  # noqa: E402
from splat.render import psnr, render, ssim  # noqa: E402
from splat.scene import load_scene  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--ply", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--no-lpips", action="store_true")
    a = ap.parse_args()
    _, test, *_ = load_scene(a.scene)
    model = load_ply(a.ply, a.device)
    lp = None
    if not a.no_lpips:
        import lpips

        lp = lpips.LPIPS(net="vgg", verbose=False).to(a.device)
    rows = []
    times = []
    with torch.no_grad():
        for cam in test:
            cam = cam.to(a.device)
            for _ in range(2):  # warm up, then time
                sync(a.device)
                t = time.perf_counter()
                img = render(model, cam, 3)[0]
                sync(a.device)
            times.append(time.perf_counter() - t)
            # the reference scores renders saved as 8-bit png; quantise the same way
            img = (img.clamp(0, 1) * 255).round() / 255
            r = {"view": cam.name, "psnr": float(psnr(img, cam.image)), "ssim": float(ssim(img, cam.image))}
            if lp is not None:
                # the lpips network expects inputs in [-1, 1]
                r["lpips"] = float(lp(img[None] * 2 - 1, cam.image[None] * 2 - 1))
                # the 3DGS evaluation script (metrics.py) passes [0, 1] tensors
                # straight in; this is how the paper's LPIPS numbers were made
                r["lpips_3dgs_script"] = float(lp(img[None], cam.image[None]))
            rows.append(r)
    mean = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0] if k != "view"}
    res = {
        "scene": a.scene, "gaussians": len(model), "views": len(rows), **mean,
        "render_ms_median": round(1000 * sorted(times)[len(times) // 2], 1),
        "resolution": f"{test[0].width}x{test[0].height}", "per_view": rows,
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print({k: v for k, v in res.items() if k != "per_view"})


if __name__ == "__main__":
    main()
