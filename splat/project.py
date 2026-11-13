"""From 3D Gaussians to 2D screen-space Gaussians (EWA splatting).

Everything here is plain PyTorch, so autograd supplies the gradients back
to positions, scales, rotations and colours. Only rasterization (the part
that loops over pixels) has hand-written kernels.

Convention: pixel (px, py) has its centre at (px, py), so the projected
mean is f * x / z + c - 0.5 for COLMAP intrinsics, whose top-left pixel
centre is at (0.5, 0.5).
"""

import math

import torch

NEAR = 0.2
SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396)
SH_C3 = (
    -0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
    -0.4570457994644658, 1.445305721320277, -0.5900435899266435,
)


def quat_to_rotmat(q):
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).view(*q.shape[:-1], 3, 3)


def eval_sh(sh, dirs, degree):
    """sh [n, 16, 3] coefficients, dirs [n, 3] unit view directions.
    Real spherical harmonics up to the given degree, as in the 3DGS code."""
    out = SH_C0 * sh[:, 0]
    if degree < 1:
        return out
    x, y, z = dirs[:, 0:1], dirs[:, 1:2], dirs[:, 2:3]
    out = out - SH_C1 * y * sh[:, 1] + SH_C1 * z * sh[:, 2] - SH_C1 * x * sh[:, 3]
    if degree < 2:
        return out
    xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
    out = (
        out
        + SH_C2[0] * xy * sh[:, 4]
        + SH_C2[1] * yz * sh[:, 5]
        + SH_C2[2] * (2 * zz - xx - yy) * sh[:, 6]
        + SH_C2[3] * xz * sh[:, 7]
        + SH_C2[4] * (xx - yy) * sh[:, 8]
    )
    if degree < 3:
        return out
    return (
        out
        + SH_C3[0] * y * (3 * xx - yy) * sh[:, 9]
        + SH_C3[1] * xy * z * sh[:, 10]
        + SH_C3[2] * y * (4 * zz - xx - yy) * sh[:, 11]
        + SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[:, 12]
        + SH_C3[4] * x * (4 * zz - xx - yy) * sh[:, 13]
        + SH_C3[5] * z * (xx - yy) * sh[:, 14]
        + SH_C3[6] * x * (xx - 3 * yy) * sh[:, 15]
    )


def project(means, log_scales, quats, cam, scale_mod=1.0):
    """Returns screen means [n, 2], conics [n, 3] (a, b, c of the inverse
    2D covariance), depths [n] and integer radii [n] (0 = culled)."""
    p = means @ cam.R.T + cam.t
    x, y, z = p.unbind(-1)
    visible = z > NEAR
    z = torch.where(visible, z, torch.ones_like(z))

    # 3D covariance R S S^T R^T in camera space
    M = quat_to_rotmat(quats) * (torch.exp(log_scales) * scale_mod)[:, None, :]
    cov = M @ M.transpose(1, 2)
    cov = cam.R @ cov @ cam.R.T

    # Jacobian of the perspective projection, taken at a point clamped to
    # 1.3x the field of view (keeps it stable for Gaussians far off-screen)
    limx = 1.3 * cam.width / (2 * cam.fx)
    limy = 1.3 * cam.height / (2 * cam.fy)
    tx = (x / z).clamp(-limx, limx) * z
    ty = (y / z).clamp(-limy, limy) * z
    zero = torch.zeros_like(z)
    J = torch.stack(
        [cam.fx / z, zero, -cam.fx * tx / (z * z), zero, cam.fy / z, -cam.fy * ty / (z * z)], dim=-1
    ).view(-1, 2, 3)
    cov2 = J @ cov @ J.transpose(1, 2)
    # a 0.3 pixel^2 low-pass filter keeps every Gaussian at least ~1 px wide
    a = cov2[:, 0, 0] + 0.3
    b = cov2[:, 0, 1]
    c = cov2[:, 1, 1] + 0.3
    det = a * c - b * b
    visible = visible & (det > 0)
    det = torch.where(visible, det, torch.ones_like(det))
    conics = torch.stack([c / det, -b / det, a / det], dim=-1)

    mid = 0.5 * (a + c)
    lam = mid + torch.sqrt((mid * mid - det).clamp(min=0.1))
    radii = torch.ceil(3.0 * torch.sqrt(lam)).detach()
    means2d = torch.stack([cam.fx * x / z + cam.cx - 0.5, cam.fy * y / z + cam.cy - 0.5], dim=-1)

    # off-screen (whole 3-sigma box outside the image) counts as culled
    mx, my = means2d.detach().unbind(-1)
    on = (mx + radii > 0) & (mx - radii < cam.width) & (my + radii > 0) & (my - radii < cam.height)
    radii = torch.where(visible & on, radii, torch.zeros_like(radii)).int()
    return means2d, conics, z, radii


def colors_from_sh(sh, means, cam, degree):
    dirs = means - cam.center
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    return (eval_sh(sh, dirs, degree) + 0.5).clamp(min=0.0)


def fov_to_focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))
