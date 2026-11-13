"""Reads COLMAP's binary sparse model: cameras.bin, images.bin, points3D.bin.

Format reference: https://colmap.github.io/format.html
"""

import os
import struct
from dataclasses import dataclass

import numpy as np

# model id -> (name, number of parameters)
MODELS = {0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4), 3: ("RADIAL", 5), 4: ("OPENCV", 8)}


@dataclass
class ColmapCamera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray

    def intrinsics(self):
        """fx, fy, cx, cy. Distortion terms are ignored: the scenes used
        here are already undistorted."""
        p = self.params
        if self.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
            return p[0], p[0], p[1], p[2]
        return p[0], p[1], p[2], p[3]


@dataclass
class ColmapImage:
    id: int
    qvec: np.ndarray  # world-to-camera rotation, (w, x, y, z)
    tvec: np.ndarray  # world-to-camera translation
    camera_id: int
    name: str


def qvec_to_rotmat(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ]
    )


class _Reader:
    def __init__(self, path):
        with open(path, "rb") as f:
            self.buf = f.read()
        self.pos = 0

    def read(self, fmt):
        out = struct.unpack_from("<" + fmt, self.buf, self.pos)
        self.pos += struct.calcsize("<" + fmt)
        return out

    def cstring(self):
        end = self.buf.index(b"\0", self.pos)
        s = self.buf[self.pos : end].decode()
        self.pos = end + 1
        return s


def read_cameras(path):
    r = _Reader(path)
    (n,) = r.read("Q")
    cams = {}
    for _ in range(n):
        cid, model, w, h = r.read("iiQQ")
        name, k = MODELS[model]
        cams[cid] = ColmapCamera(cid, name, w, h, np.array(r.read("d" * k)))
    return cams


def read_images(path):
    r = _Reader(path)
    (n,) = r.read("Q")
    imgs = {}
    for _ in range(n):
        iid, *rest = r.read("idddddddi")
        q, t, cid = np.array(rest[:4]), np.array(rest[4:7]), rest[7]
        name = r.cstring()
        (npts,) = r.read("Q")
        r.pos += npts * 24  # x, y, point3D id: not needed
        imgs[iid] = ColmapImage(iid, q, t, cid, name)
    return imgs


def read_points(path):
    """Returns xyz [n, 3] float64 and rgb [n, 3] uint8."""
    r = _Reader(path)
    (n,) = r.read("Q")
    xyz = np.empty((n, 3))
    rgb = np.empty((n, 3), dtype=np.uint8)
    for i in range(n):
        _, x, y, z, cr, cg, cb, _, track = r.read("QdddBBBdQ")
        xyz[i] = (x, y, z)
        rgb[i] = (cr, cg, cb)
        r.pos += track * 8
    return xyz, rgb


def read_model(scene):
    d = os.path.join(scene, "sparse", "0")
    return read_cameras(os.path.join(d, "cameras.bin")), read_images(os.path.join(d, "images.bin")), read_points(
        os.path.join(d, "points3D.bin")
    )
