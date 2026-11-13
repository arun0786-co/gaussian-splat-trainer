"""The Gaussian parameters, their optimizer, and adaptive density control."""

import math

import numpy as np
import torch

from .project import SH_C0, quat_to_rotmat


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


class Adam:
    """Adam over a dict of tensors whose rows can be appended, removed or
    reset, which densification needs. torch.optim cannot resize state."""

    def __init__(self, params, lrs, betas=(0.9, 0.999), eps=1e-15):
        self.p, self.lr = params, dict(lrs)
        self.b1, self.b2, self.eps = betas[0], betas[1], eps
        self.m = {k: torch.zeros_like(v) for k, v in params.items()}
        self.v = {k: torch.zeros_like(v) for k, v in params.items()}
        self.t = 0

    @torch.no_grad()
    def step(self):
        self.t += 1
        c1, c2 = 1 - self.b1**self.t, 1 - self.b2**self.t
        for k, p in self.p.items():
            if p.grad is None:
                continue
            g = p.grad
            self.m[k].mul_(self.b1).add_(g, alpha=1 - self.b1)
            self.v[k].mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            p.addcdiv_(self.m[k] / c1, (self.v[k] / c2).sqrt_().add_(self.eps), value=-self.lr[k])
            p.grad = None


class Gaussians:
    KEYS = ("means", "sh_dc", "sh_rest", "log_scales", "quats", "opacity")

    def __init__(self, params):
        self.params = {k: v.detach().requires_grad_() for k, v in params.items()}

    def __getattr__(self, k):
        if k in Gaussians.KEYS:
            return self.params[k]
        raise AttributeError(k)

    def __len__(self):
        return self.params["means"].shape[0]

    @classmethod
    def from_points(cls, xyz, rgb, device, init_opacity=0.1):
        from scipy.spatial import cKDTree

        n = xyz.shape[0]
        # initial size: mean distance to the 3 nearest neighbours
        d, _ = cKDTree(xyz.numpy()).query(xyz.numpy(), k=4)
        d2 = np.maximum((d[:, 1:] ** 2).mean(1), 1e-7)
        params = {
            "means": xyz.clone(),
            "sh_dc": ((rgb - 0.5) / SH_C0)[:, None, :],
            "sh_rest": torch.zeros(n, 15, 3),
            "log_scales": torch.log(torch.sqrt(torch.tensor(d2, dtype=torch.float32)))[:, None].repeat(1, 3),
            "quats": torch.tensor([1.0, 0, 0, 0]).repeat(n, 1),
            "opacity": inverse_sigmoid(torch.full((n,), init_opacity)),
        }
        return cls({k: v.to(device) for k, v in params.items()})

    @property
    def sh(self):
        return torch.cat([self.params["sh_dc"], self.params["sh_rest"]], dim=1)


class Densifier:
    """Adaptive density control from the 3DGS paper.

    Gaussians whose projected centre keeps receiving large gradients sit in
    regions the current set cannot fit. Small ones are cloned (under-
    reconstruction), large ones are split in two smaller ones sampled
    inside them (over-reconstruction). Nearly transparent and oversized
    ones are pruned, and every 3,000 steps all opacities are pulled down
    to 0.01 so that floaters have to earn their opacity back.
    """

    def __init__(self, model: Gaussians, opt: Adam, extent, grad_thresh=0.0002, percent_dense=0.01):
        self.model, self.opt, self.extent = model, opt, extent
        self.grad_thresh, self.percent_dense = grad_thresh, percent_dense
        self.reset_stats()

    def reset_stats(self):
        n, dev = len(self.model), self.model.means.device
        self.grad_accum = torch.zeros(n, device=dev)
        self.denom = torch.zeros(n, device=dev)
        self.max_radii = torch.zeros(n, device=dev)

    @torch.no_grad()
    def add_stats(self, means2d_grad, radii, width, height):
        vis = radii > 0
        # the paper's threshold is for gradients in normalised device
        # coordinates; ours are in pixels
        g = means2d_grad * torch.tensor([width / 2, height / 2], device=means2d_grad.device)
        self.grad_accum += torch.where(vis, g.norm(dim=-1), torch.zeros_like(self.grad_accum))
        self.denom += vis.float()
        self.max_radii = torch.maximum(self.max_radii, radii.float())

    def _rebuild(self, keep, new):
        """Keeps rows where keep is True and appends the rows in new."""
        p, opt = self.model.params, self.opt
        for k in Gaussians.KEYS:
            data = torch.cat([p[k].detach()[keep], new[k]]) if new else p[k].detach()[keep]
            m = torch.cat([opt.m[k][keep], torch.zeros_like(new[k])]) if new else opt.m[k][keep]
            v = torch.cat([opt.v[k][keep], torch.zeros_like(new[k])]) if new else opt.v[k][keep]
            p[k] = data.requires_grad_()
            opt.p[k], opt.m[k], opt.v[k] = p[k], m, v
        n_new = new["means"].shape[0] if new else 0
        dev = keep.device
        self.grad_accum = torch.cat([self.grad_accum[keep], torch.zeros(n_new, device=dev)])
        self.denom = torch.cat([self.denom[keep], torch.zeros(n_new, device=dev)])
        self.max_radii = torch.cat([self.max_radii[keep], torch.zeros(n_new, device=dev)])

    @torch.no_grad()
    def densify_and_prune(self, prune_big):
        p = self.model.params
        grads = self.grad_accum / self.denom.clamp(min=1)
        big_grad = grads >= self.grad_thresh
        max_scale = torch.exp(p["log_scales"]).max(dim=1).values
        small = max_scale <= self.percent_dense * self.extent
        clone = big_grad & small
        split = big_grad & ~small
        n = len(self.model)

        # clone: an exact copy, which then moves with its own gradient
        new = {k: p[k][clone] for k in Gaussians.KEYS}
        # split: two samples drawn from the Gaussian itself, 1.6x smaller
        s = torch.exp(p["log_scales"][split]).repeat(2, 1)
        R = quat_to_rotmat(p["quats"][split]).repeat(2, 1, 1)
        offs = (R @ (torch.randn_like(s) * s)[..., None])[..., 0]
        split_new = {k: p[k][split].repeat(2, *([1] * (p[k].dim() - 1))) for k in Gaussians.KEYS}
        split_new["means"] = split_new["means"] + offs
        split_new["log_scales"] = torch.log(s / 1.6)
        new = {k: torch.cat([new[k], split_new[k]]) for k in Gaussians.KEYS}

        opacity = torch.sigmoid(p["opacity"])
        prune = split | (opacity < 0.005)
        if prune_big:
            prune |= (self.max_radii > 20) | (max_scale > 0.1 * self.extent)
        keep = ~prune
        self._rebuild(keep, new)
        self.reset_stats()
        return {"cloned": int(clone.sum()), "split": int(split.sum()), "pruned": int(prune.sum() - split.sum()),
                "before": n, "after": len(self.model)}

    @torch.no_grad()
    def reset_opacity(self):
        p = self.model.params
        p["opacity"].copy_(torch.minimum(p["opacity"], inverse_sigmoid(torch.tensor(0.01))))
        self.opt.m["opacity"].zero_()
        self.opt.v["opacity"].zero_()


def lr_log_linear(step, start, end, steps):
    t = min(max(step / steps, 0.0), 1.0)
    return math.exp(math.log(start) * (1 - t) + math.log(end) * t)
