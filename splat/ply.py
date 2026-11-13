"""Saves Gaussians in the .ply layout of the 3DGS reference code, which
most splat viewers read."""

import numpy as np
import torch


def save_ply(model, path):
    p = {k: v.detach().float().cpu() for k, v in model.params.items()}
    n = p["means"].shape[0]
    # f_rest is stored channel-major: all 15 red coefficients, then green, then blue
    rest = p["sh_rest"].transpose(1, 2).reshape(n, -1)
    cols = [p["means"], torch.zeros(n, 3), p["sh_dc"].reshape(n, 3), rest, p["opacity"][:, None],
            p["log_scales"], p["quats"]]
    data = torch.cat(cols, dim=1).numpy().astype(np.float32)
    names = (["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(3)] + [f"f_rest_{i}" for i in range(45)]
             + ["opacity"] + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)])
    assert data.shape[1] == len(names)
    header = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % n
    header += "".join(f"property float {nm}\n" for nm in names) + "end_header\n"
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(data.tobytes())


def load_ply(path, device="cpu"):
    from .gaussians import Gaussians

    with open(path, "rb") as f:
        names = []
        n = 0
        while True:
            line = f.readline().decode().strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property"):
                names.append(line.split()[-1])
            elif line == "end_header":
                break
        data = torch.from_numpy(np.frombuffer(f.read(), dtype=np.float32).reshape(n, len(names)).copy())
    col = {nm: i for i, nm in enumerate(names)}

    def take(prefix, k):
        return data[:, [col[f"{prefix}{i}"] for i in range(k)]]

    params = {
        "means": data[:, [col["x"], col["y"], col["z"]]],
        "sh_dc": take("f_dc_", 3)[:, None, :],
        "sh_rest": take("f_rest_", 45).reshape(n, 3, 15).transpose(1, 2),
        "opacity": data[:, col["opacity"]],
        "log_scales": take("scale_", 3),
        "quats": take("rot_", 4),
    }
    return Gaussians({k: v.contiguous().to(device) for k, v in params.items()})
