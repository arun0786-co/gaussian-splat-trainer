"""Metal projection + colour kernels against the PyTorch reference and
its autograd gradients."""

import math

import pytest
import torch

from splat.project import colors_from_sh, project
from splat.scene import Camera

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs an apple gpu")


def setup(n, seed, degree):
    g = torch.Generator().manual_seed(seed)
    means = torch.randn(n, 3, generator=g) * torch.tensor([3.0, 2.0, 3.0]) + torch.tensor([0, 0, 4.0])
    log_scales = torch.log(torch.rand(n, 3, generator=g) * 0.4 + 0.01)
    quats = torch.randn(n, 4, generator=g)
    sh = torch.randn(n, 16, 3, generator=g) * 0.5
    ang = 0.3
    R = torch.tensor([[math.cos(ang), 0, math.sin(ang)], [0, 1, 0], [-math.sin(ang), 0, math.cos(ang)]])
    cam = Camera("c", R, torch.tensor([0.2, -0.1, 0.5]), 300.0, 280.0, 160.0, 120.0, 320, 240)
    dev = "mps"
    return [x.to(dev) for x in (means, log_scales, quats, sh)], cam.to(dev)


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_matches_reference_and_autograd(degree):
    from splat.project_metal import project_and_color

    (means, ls, q, sh), cam = setup(3000, degree, degree)
    outs, grads = [], []
    for fn in ("metal", "ref"):
        p = [x.clone().requires_grad_() for x in (means, ls, q, sh)]
        if fn == "metal":
            m2, con, depth, radii, col = project_and_color(*p, cam, degree)
        else:
            m2, con, depth, radii = project(p[0], p[1], p[2], cam)
            col = colors_from_sh(p[3], p[0], cam, degree)
        outs.append((m2, con, depth, radii, col))
        # upstream gradients as the rasterizer would give them: zero for culled
        g = torch.Generator(device="mps").manual_seed(1)
        vis = (radii > 0).float()[:, None]
        loss = ((m2 * torch.randn(m2.shape, device="mps", generator=g) * vis).sum()
                + (con * torch.randn(con.shape, device="mps", generator=g) * vis).sum()
                + (col * torch.randn(col.shape, device="mps", generator=g)).sum())
        loss.backward()
        grads.append([x.grad for x in p])
    (m2a, ca, da, ra, cola), (m2b, cb, db, rb, colb) = outs
    vis = rb > 0
    assert 0.2 < vis.float().mean() < 0.95  # some culled, most not
    assert torch.equal(ra, rb.int())
    assert torch.allclose(m2a[vis], m2b[vis], atol=1e-3)
    assert torch.allclose(ca[vis], cb[vis], rtol=1e-4, atol=1e-6)
    assert torch.allclose(cola, colb, atol=1e-5)
    for name, a, b in zip(["means", "log_scales", "quats", "sh"], *grads):
        err = (a - b).abs().max().item()
        assert err <= 1e-3 * max(1.0, b.abs().max().item()), (name, err, b.abs().max().item())
