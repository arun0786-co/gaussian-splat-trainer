import math

import torch

from splat.gaussians import Adam, Densifier, Gaussians, inverse_sigmoid
from splat.ply import load_ply, save_ply
from splat.project import project
from splat.render import psnr, render
from splat.scene import Camera

DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
REF = DEV == "cpu"  # no gpu: use the dense reference rasterizer


def cam_at(eye, target, f=60.0, size=64, device="cpu"):
    """A camera at eye looking at target, y down (COLMAP convention)."""
    eye, target = torch.tensor(eye, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)
    z = target - eye
    z = z / z.norm()
    x = torch.linalg.cross(z, torch.tensor([0.0, -1.0, 0.0]))
    x = x / x.norm()
    y = torch.linalg.cross(z, x)
    R = torch.stack([x, y, z])
    return Camera("c", R.to(device), (-R @ eye).to(device), f, f, size / 2, size / 2, size, size)


def test_projection_against_pinhole():
    cam = Camera("c", torch.eye(3), torch.zeros(3), 100.0, 100.0, 50.0, 50.0, 100, 100)
    means = torch.tensor([[0.3, -0.2, 4.0], [0.0, 0.0, 4.0], [0.0, 0.0, -1.0]])
    s = 0.05
    log_scales = torch.full((3, 3), math.log(s))
    quats = torch.tensor([[1.0, 0, 0, 0]] * 3)
    m2, conics, depth, radii = project(means, log_scales, quats, cam)
    assert torch.allclose(m2[0], torch.tensor([100 * 0.3 / 4 + 49.5, 100 * -0.2 / 4 + 49.5]))
    # on the optical axis an isotropic Gaussian projects to (f s / z)^2 + 0.3
    var = (100 * s / 4) ** 2 + 0.3
    assert torch.allclose(conics[1], torch.tensor([1 / var, 0.0, 1 / var]), rtol=1e-5)
    assert radii[1] == math.ceil(3 * math.sqrt(var))
    assert radii[2] == 0  # behind the camera


def random_model(n, seed, device):
    g = torch.Generator().manual_seed(seed)
    params = {
        "means": torch.randn(n, 3, generator=g) * 0.6,
        "sh_dc": torch.randn(n, 1, 3, generator=g),
        "sh_rest": torch.randn(n, 15, 3, generator=g) * 0.1,
        "log_scales": torch.log(torch.rand(n, 3, generator=g) * 0.15 + 0.03),
        "quats": torch.randn(n, 4, generator=g),
        "opacity": inverse_sigmoid(torch.rand(n, generator=g) * 0.8 + 0.1),
    }
    return Gaussians({k: v.to(device) for k, v in params.items()})


def test_ply_roundtrip(tmp_path):
    m = random_model(50, 0, "cpu")
    save_ply(m, tmp_path / "a.ply")
    back = load_ply(tmp_path / "a.ply")
    for k in Gaussians.KEYS:
        assert torch.equal(m.params[k], back.params[k]), k


def test_densify_keeps_rows_aligned():
    m = random_model(200, 1, "cpu")
    opt = Adam(m.params, {k: 1e-3 for k in Gaussians.KEYS})
    for k in Gaussians.KEYS:
        m.params[k].grad = torch.randn_like(m.params[k])
    opt.step()
    d = Densifier(m, opt, extent=10.0)
    d.grad_accum[:60] = 1.0  # large gradients for the first 60
    d.denom[:] = 1
    with torch.no_grad():
        m.params["opacity"][-10:] = -10.0  # transparent: pruned
        m.params["log_scales"][:30] = math.log(0.05)  # small: cloned
        m.params["log_scales"][30:60] = math.log(0.5)  # large: split
    info = d.densify_and_prune(prune_big=False)
    assert info == {"cloned": 30, "split": 30, "pruned": 10, "before": 200, "after": 200 + 30 + 30 - 10}
    n = len(m)
    for k in Gaussians.KEYS:
        assert m.params[k].shape[0] == opt.m[k].shape[0] == opt.v[k].shape[0] == n
        assert opt.p[k] is m.params[k]
    # clones are exact copies; their moments start at zero
    assert torch.equal(m.params["means"][n - 90 : n - 60], m.params["means"][:30])
    assert (opt.m["means"][n - 90 :] == 0).all()


def test_fits_a_synthetic_scene():
    """Renders a random 'true' scene from 12 views, starts from a perturbed
    copy with wrong colours, and checks that training recovers it."""
    torch.manual_seed(0)
    truth = random_model(60, 2, DEV)
    size = 48 if REF else 64
    cams = []
    for i in range(12):
        a = 2 * math.pi * i / 12
        c = cam_at([3 * math.cos(a), 0.6 * math.sin(3 * a), 3 * math.sin(a)], [0, 0, 0], f=size, size=size, device=DEV)
        with torch.no_grad():
            c.image = render(truth, c, 0, reference=REF)[0]
        cams.append(c)
    start = {k: v.detach().clone() for k, v in truth.params.items()}
    start["means"] += 0.08 * torch.randn_like(start["means"])
    start["sh_dc"] = torch.zeros_like(start["sh_dc"])
    model = Gaussians(start)
    opt = Adam(model.params, {"means": 2e-3, "sh_dc": 2e-2, "sh_rest": 1e-3, "opacity": 2e-2, "log_scales": 5e-3,
                              "quats": 1e-3})

    def mean_psnr():
        with torch.no_grad():
            return sum(float(psnr(render(model, c, 0, reference=REF)[0], c.image)) for c in cams) / len(cams)

    before = mean_psnr()
    for step in range(300 if REF else 600):
        c = cams[step % len(cams)]
        img = render(model, c, 0, reference=REF)[0]
        (img - c.image).abs().mean().backward()
        opt.step()
    after = mean_psnr()
    assert after > before + 10, (before, after)
