"""Metal rasterizer against the dense PyTorch reference: same image, same
gradients."""

import pytest
import torch

from splat.raster import TILE, bin_gaussians, rasterize, rasterize_reference, tile_rects

mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs an apple gpu")


def random_2d(n, width, height, seed=0, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    means = torch.rand(n, 2, generator=g) * torch.tensor([width + 20.0, height + 20.0]) - 10
    # random covariances: rotation and two scales between 0.7 and 12 px
    ang = torch.rand(n, generator=g) * 3.1416
    s = torch.exp(torch.rand(n, 2, generator=g) * 2.8 - 0.4)
    c, si = torch.cos(ang), torch.sin(ang)
    R = torch.stack([c, -si, si, c], -1).view(n, 2, 2)
    cov = R @ torch.diag_embed(s * s) @ R.transpose(1, 2)
    a, b, cc = cov[:, 0, 0], cov[:, 0, 1], cov[:, 1, 1]
    det = a * cc - b * b
    conics = torch.stack([cc / det, -b / det, a / det], -1)
    mid = 0.5 * (a + cc)
    radii = torch.ceil(3 * torch.sqrt(mid + torch.sqrt((mid * mid - det).clamp(min=0.1)))).int()
    colors = torch.rand(n, 3, generator=g)
    opac = torch.rand(n, generator=g) * 0.98 + 0.01
    depths = torch.rand(n, generator=g) * 10 + 1
    radii[torch.rand(n, generator=g) < 0.05] = 0  # some culled
    out = [means, conics, colors, opac, depths, radii]
    return [t.to(device) for t in out]


def test_binning_matches_brute_force():
    w, h = 70, 45
    means, conics, colors, opac, depths, radii = random_2d(300, w, h, seed=1)
    ranges, plist = bin_gaussians(means, depths, radii, w, h)
    x0, x1, y0, y1 = tile_rects(means, radii, w, h)
    tx = -(-w // TILE)
    for t in range(ranges.shape[0]):
        ix, iy = t % tx, t // tx
        hit = ((radii > 0) & (x0 <= ix) & (ix < x1) & (y0 <= iy) & (iy < y1)).nonzero()[:, 0]
        expect = hit[torch.argsort(depths[hit], stable=True)]
        got = plist[ranges[t, 0] : ranges[t, 1]].long()
        assert torch.equal(got, expect)


@mps
@pytest.mark.parametrize("n,w,h", [(1, 16, 16), (300, 70, 45), (2000, 96, 64)])
def test_forward_matches_reference(n, w, h):
    args = random_2d(n, w, h, seed=n, device="mps")
    bg = torch.tensor([0.2, 0.5, 0.9], device="mps")
    got = rasterize(*args, w, h, bg)
    ref = rasterize_reference(*args, w, h, bg)
    assert (got - ref).abs().max() < 1e-5


@mps
@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize("n,w,h", [(300, 70, 45), (2000, 96, 64)])
def test_backward_matches_reference(n, w, h, exact):
    means, conics, colors, opac, depths, radii = random_2d(n, w, h, seed=n + 7, device="mps")
    bg = torch.tensor([0.2, 0.5, 0.9], device="mps")
    target = torch.rand(3, h, w, device="mps", generator=torch.Generator(device="mps").manual_seed(0))
    grads = []
    for fn in (rasterize, rasterize_reference):
        p = [t.clone().requires_grad_() for t in (means, conics, colors, opac)]
        img = fn(p[0], p[1], p[2], p[3], depths, radii, w, h, bg, exact_clamp=exact)
        ((img - target) ** 2).sum().backward()
        grads.append([q.grad for q in p])
    for name, a, b in zip(["means", "conics", "colors", "opacities"], *grads):
        scale = b.abs().max().item()
        assert scale > 0
        assert (a - b).abs().max().item() < 1e-4 * max(1.0, scale), name
