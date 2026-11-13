import torch
import torch.nn.functional as F

from .project import colors_from_sh, project
from .project_metal import project_and_color
from .raster import rasterize, rasterize_reference


def render(model, cam, sh_degree=3, bg=None, reference=False):
    """Returns the image [3, h, w] and what training needs: the projected
    means (for the densification gradient) and the radii.

    reference=True runs the PyTorch versions of every stage (autograd
    gradients, dense rasterizer); otherwise the Metal kernels run."""
    p = model.params
    bg = bg if bg is not None else torch.zeros(3, device=p["means"].device)
    if reference:
        means2d, conics, depths, radii = project(p["means"], p["log_scales"], p["quats"], cam)
        colors = colors_from_sh(model.sh, p["means"], cam, sh_degree)
    else:
        means2d, conics, depths, radii, colors = project_and_color(
            p["means"], p["log_scales"], p["quats"], model.sh, cam, sh_degree
        )
    opac = torch.sigmoid(p["opacity"])
    fn = rasterize_reference if reference else rasterize
    img = fn(means2d, conics, colors, opac, depths, radii, cam.width, cam.height, bg)
    return img, means2d, radii


def _window(size=11, sigma=1.5, device="cpu"):
    x = torch.arange(size, device=device, dtype=torch.float32) - size // 2
    g = torch.exp(-(x * x) / (2 * sigma * sigma))
    g = g / g.sum()
    return (g[:, None] @ g[None, :]).expand(3, 1, size, size).contiguous()


def ssim(a, b):
    """Mean SSIM of two [3, h, w] images, 11x11 Gaussian window (sigma
    1.5), zero padding, as in the 3DGS evaluation code."""
    w = _window(device=a.device)
    a, b = a[None], b[None]
    mu_a = F.conv2d(a, w, padding=5, groups=3)
    mu_b = F.conv2d(b, w, padding=5, groups=3)
    s_aa = F.conv2d(a * a, w, padding=5, groups=3) - mu_a * mu_a
    s_bb = F.conv2d(b * b, w, padding=5, groups=3) - mu_b * mu_b
    s_ab = F.conv2d(a * b, w, padding=5, groups=3) - mu_a * mu_b
    c1, c2 = 0.01**2, 0.03**2
    m = ((2 * mu_a * mu_b + c1) * (2 * s_ab + c2)) / ((mu_a**2 + mu_b**2 + c1) * (s_aa + s_bb + c2))
    return m.mean()


def psnr(a, b):
    return -10 * torch.log10(((a - b) ** 2).mean())
