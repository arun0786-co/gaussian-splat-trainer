"""Trains a scene.

    python -m splat.train --scene data/tandt/truck --out output/truck --iters 30000
"""

import argparse
import json
import os
import random
import time

import torch

from .gaussians import Adam, Densifier, Gaussians, lr_log_linear
from .ply import save_ply
from .render import psnr, render, ssim
from .scene import load_scene


@torch.no_grad()
def evaluate(model, cams, degree, bg, save_dir=None, save_n=0):
    ps, ss = [], []
    for i, cam in enumerate(cams):
        img = render(model, cam, degree, bg)[0].clamp(0, 1)
        gt = cam.image
        ps.append(float(psnr(img, gt)))
        ss.append(float(ssim(img, gt)))
        if save_dir and i < save_n:
            from PIL import Image

            both = torch.cat([gt, img], dim=2).permute(1, 2, 0).mul(255).round().byte().cpu().numpy()
            Image.fromarray(both).save(os.path.join(save_dir, f"{i:03d}_{cam.name.rsplit('.', 1)[0]}.jpg"), quality=92)
    return {"psnr": sum(ps) / len(ps), "ssim": sum(ss) / len(ss), "views": len(cams)}


def train(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    dev = args.device
    os.makedirs(args.out, exist_ok=True)
    train_cams, test_cams, xyz, rgb, extent = load_scene(args.scene, args.downscale)
    train_cams = [c.to(dev) for c in train_cams]
    test_cams = [c.to(dev) for c in test_cams]
    print(f"{len(train_cams)} train / {len(test_cams)} test views, {train_cams[0].width}x{train_cams[0].height}, "
          f"{xyz.shape[0]} sfm points, extent {extent:.2f}", flush=True)

    model = Gaussians.from_points(xyz, rgb, dev)
    lrs = {"means": 1.6e-4 * extent, "sh_dc": 2.5e-3, "sh_rest": 2.5e-3 / 20, "opacity": 0.05,
           "log_scales": 5e-3, "quats": 1e-3}
    opt = Adam(model.params, lrs)
    dens = Densifier(model, opt, extent)
    bg = torch.zeros(3, device=dev)
    order = []
    log = {"args": vars(args), "steps": [], "evals": {}}
    t0 = time.perf_counter()
    train_time = 0.0
    for step in range(1, args.iters + 1):
        ts = time.perf_counter()
        opt.lr["means"] = lr_log_linear(step, 1.6e-4 * extent, 1.6e-6 * extent, args.iters)
        degree = min(3, step // 1000)
        if not order:
            order = list(range(len(train_cams)))
            random.shuffle(order)
        cam = train_cams[order.pop()]

        img, means2d, radii = render(model, cam, degree, bg)
        means2d.retain_grad()
        l1 = (img - cam.image).abs().mean()
        loss = 0.8 * l1 + 0.2 * (1 - ssim(img, cam.image))
        loss.backward()

        with torch.no_grad():
            if step < args.densify_until:
                dens.add_stats(means2d.grad, radii, cam.width, cam.height)
            opt.step()
            info = None
            if step < args.densify_until and step > 500 and step % 100 == 0:
                info = dens.densify_and_prune(prune_big=step > 3000)
                # tensor sizes change every step (the tile list) and every 100
                # steps (the Gaussians); without this the mps allocator's cache
                # of odd-sized freed blocks grew past 20 GB
                if dev == "mps":
                    torch.mps.empty_cache()
            if step < args.densify_until and step % 3000 == 0:
                dens.reset_opacity()
        if dev == "mps":
            torch.mps.synchronize()
        train_time += time.perf_counter() - ts

        if step % 100 == 0 or step == 1:
            rec = {"step": step, "loss": round(loss.item(), 5), "gaussians": len(model),
                   "train_s": round(train_time, 1)}
            if info:
                rec["densify"] = info
            if dev == "mps":
                rec["gpu_mem_gb"] = round(torch.mps.driver_allocated_memory() / 2**30, 2)
            log["steps"].append(rec)
            if step % 500 == 0 or step == 1:
                print(rec, flush=True)
        if step in args.eval_at or step == args.iters:
            d = os.path.join(args.out, f"renders_{step}")
            os.makedirs(d, exist_ok=True)
            res = evaluate(model, test_cams, degree, bg, d, save_n=6)
            res.update({"gaussians": len(model), "train_s": round(train_time, 1)})
            log["evals"][step] = res
            print("eval", step, res, flush=True)
            with open(os.path.join(args.out, "log.json"), "w") as f:
                json.dump(log, f, indent=1)
    save_ply(model, os.path.join(args.out, "point_cloud.ply"))
    print(f"done in {time.perf_counter() - t0:.0f} s", flush=True)
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--densify-until", type=int, default=15000)
    ap.add_argument("--downscale", type=float, default=1)
    ap.add_argument("--eval-at", type=int, nargs="*", default=[7000, 30000])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
