"""Cameras and images of a COLMAP scene."""

import os
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from .colmap import qvec_to_rotmat, read_model


@dataclass
class Camera:
    name: str
    R: torch.Tensor  # [3, 3] world to camera
    t: torch.Tensor  # [3]
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    image: torch.Tensor | None = None  # [3, h, w] in [0, 1]

    @property
    def center(self):
        return -self.R.T @ self.t

    def to(self, device):
        img = self.image.to(device) if self.image is not None else None
        return Camera(self.name, self.R.to(device), self.t.to(device), self.fx, self.fy, self.cx, self.cy,
                      self.width, self.height, img)


def load_scene(path, downscale=1, test_every=8, images_dir="images"):
    """Returns train cameras, test cameras, points [n, 3], colors [n, 3] in
    [0, 1] and the scene radius. Every test_every-th image, in name order,
    is held out for testing, the split used by the 3DGS paper."""
    cams, imgs, (xyz, rgb) = read_model(path)
    out = []
    for im in sorted(imgs.values(), key=lambda i: i.name):
        c = cams[im.camera_id]
        fx, fy, cx, cy = c.intrinsics()
        pil = Image.open(os.path.join(path, images_dir, im.name)).convert("RGB")
        w, h = pil.size
        # the images can be stored at a different size than the model's
        sx, sy = w / c.width, h / c.height
        if downscale != 1:
            w, h = round(w / downscale), round(h / downscale)
            pil = pil.resize((w, h), Image.LANCZOS)
            sx, sy = sx / downscale, sy / downscale
        img = torch.from_numpy(np.asarray(pil).copy()).permute(2, 0, 1).float() / 255.0
        out.append(
            Camera(
                im.name,
                torch.tensor(qvec_to_rotmat(im.qvec), dtype=torch.float32),
                torch.tensor(im.tvec, dtype=torch.float32),
                fx * sx, fy * sy, cx * sx, cy * sy, w, h, img,
            )
        )
    test = [c for i, c in enumerate(out) if test_every and i % test_every == 0]
    train = [c for i, c in enumerate(out) if not (test_every and i % test_every == 0)]
    centers = torch.stack([c.center for c in out])
    radius = 1.1 * float((centers - centers.mean(0)).norm(dim=1).max())
    return train, test, torch.tensor(xyz, dtype=torch.float32), torch.tensor(rgb, dtype=torch.float32) / 255.0, radius
