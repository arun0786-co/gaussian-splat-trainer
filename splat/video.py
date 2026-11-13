"""Renders a fly-through along the training cameras of a trained scene.

    python -m splat.video --scene data/tandt/truck --ply output/truck/point_cloud.ply --out truck.mp4
"""

import argparse
import os
import subprocess
import tempfile

import numpy as np
import torch
from PIL import Image

from .ply import load_ply
from .render import render
from .scene import Camera, load_scene


def slerp(q0, q1, t):
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(d)
    return (np.sin((1 - t) * th) * q0 + np.sin(t * th) * q1) / np.sin(th)


def rot_to_quat(R):
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return np.array([w, x, y, z])


def quat_to_rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def path(cams, frames):
    """Camera poses interpolated along the capture order: positions
    linearly, orientations by slerp."""
    keys = cams[:: max(1, len(cams) // 24)]
    centers = [c.center.cpu().numpy() for c in keys]
    quats = [rot_to_quat(c.R.cpu().numpy()) for c in keys]
    out = []
    for f in range(frames):
        s = f / frames * (len(keys) - 1)
        i, t = int(s), s - int(s)
        j = min(i + 1, len(keys) - 1)
        c = (1 - t) * centers[i] + t * centers[j]
        R = quat_to_rot(slerp(quats[i], quats[j], t))
        k = keys[i]
        out.append(Camera(f"f{f}", torch.tensor(R, dtype=torch.float32), torch.tensor(-R @ c, dtype=torch.float32),
                          k.fx, k.fy, k.cx, k.cy, k.width, k.height))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--ply", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--downscale", type=float, default=1)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    train, test, *_ = load_scene(a.scene, a.downscale)
    cams = sorted(train + test, key=lambda c: c.name)
    model = load_ply(a.ply, a.device)
    with tempfile.TemporaryDirectory() as d:
        with torch.no_grad():
            for i, cam in enumerate(path(cams, a.frames)):
                img = render(model, cam.to(a.device), 3)[0].clamp(0, 1)
                arr = img.permute(1, 2, 0).mul(255).round().byte().cpu().numpy()
                h, w = arr.shape[:2]
                Image.fromarray(arr[: h - h % 2, : w - w % 2]).save(os.path.join(d, f"{i:05d}.png"))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "30", "-i", os.path.join(d, "%05d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", a.out], check=True)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
