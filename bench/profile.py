"""Time per training-step stage on a trained scene, Metal kernels against
the PyTorch versions of projection.

    python bench/profile.py --scene data/tandt/truck --ply output/truck/point_cloud.ply
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from splat.ply import load_ply  # noqa: E402
from splat.project import colors_from_sh, project  # noqa: E402
from splat.project_metal import project_and_color  # noqa: E402
from splat.raster import _Rasterize, bin_gaussians  # noqa: E402
from splat.render import ssim  # noqa: E402
from splat.scene import load_scene  # noqa: E402


def sync():
    torch.mps.synchronize()
    return time.perf_counter()


def timed(fn, n=10):
    for _ in range(2):
        fn()
    t = sync()
    for _ in range(n):
        fn()
    return (sync() - t) / n * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--ply", required=True)
    ap.add_argument("--out", default="results/profile.json")
    a = ap.parse_args()
    train, *_ = load_scene(a.scene)
    cam = train[0].to("mps")
    m = load_ply(a.ply, "mps")
    p = {k: v.requires_grad_() for k, v in m.params.items()}
    bg = torch.zeros(3, device="mps")
    W, H = cam.width, cam.height

    def proj_torch():
        m2, con, d, r = project(p["means"], p["log_scales"], p["quats"], cam)
        col = colors_from_sh(m.sh, p["means"], cam, 3)
        return m2, con, d, r, col

    def proj_metal():
        return project_and_color(p["means"], p["log_scales"], p["quats"], m.sh, cam, 3)

    def fwd_bwd(fn):
        def run():
            m2, con, d, r, col = fn()
            (m2.sum() + con.sum() + col.sum()).backward()
            for v in p.values():
                v.grad = None
        return run

    m2, con, d, r, col = proj_metal()
    m2, con, col = m2.detach(), con.detach(), col.detach()
    op = torch.sigmoid(p["opacity"]).detach()
    ranges, plist = bin_gaussians(m2, d, r, W, H)
    rows = {
        "gaussians": len(m),
        "tile overlaps": int(plist.numel()),
        "projection + SH, PyTorch, fwd+bwd ms": timed(fwd_bwd(proj_torch)),
        "projection + SH, Metal, fwd+bwd ms": timed(fwd_bwd(proj_metal)),
        "binning (sort) ms": timed(lambda: bin_gaussians(m2, d, r, W, H)),
    }
    leaves = [x.clone().requires_grad_() for x in (m2, con, col, op)]

    def raster_fwd():
        with torch.no_grad():
            _Rasterize.apply(*leaves, ranges, plist, W, H, bg, False)

    def raster_fwd_bwd():
        img = _Rasterize.apply(*leaves, ranges, plist, W, H, bg, False)
        img.sum().backward()

    rows["rasterize forward ms"] = timed(raster_fwd)
    rows["rasterize forward+backward ms"] = timed(raster_fwd_bwd)
    img = torch.rand(3, H, W, device="mps", requires_grad=True)

    def loss():
        l = 0.8 * (img - cam.image).abs().mean() + 0.2 * (1 - ssim(img, cam.image))
        l.backward()

    rows["loss (L1 + SSIM) fwd+bwd ms"] = timed(loss)
    rows = {k: (round(v, 1) if isinstance(v, float) else v) for k, v in rows.items()}
    rows["resolution"] = f"{W}x{H}"
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rows, f, indent=1)
    for k, v in rows.items():
        print(f"{k:42s} {v}")


if __name__ == "__main__":
    main()
