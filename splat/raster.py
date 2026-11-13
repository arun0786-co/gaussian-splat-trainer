"""Rasterizing 2D Gaussians: tile binning, a dense reference in PyTorch,
and the Metal kernels used for training.

A pixel's colour is front-to-back alpha compositing over the Gaussians
sorted by depth:

    alpha_i = min(0.99, opacity_i * exp(-0.5 d^T conic_i d)),  d = mean_i - pixel
    C = sum_i c_i alpha_i T_i + T_final * background,  T_i = prod_{j<i} (1 - alpha_j)

with the rules of the 3DGS rasterizer: Gaussians with alpha < 1/255 are
skipped, a pixel stops once its transmittance would fall below 1e-4, and
a Gaussian only touches pixels in the 16x16 tiles its 3-sigma box covers.
The reference implements exactly these rules with dense tensors, so the
kernels can be checked against it value for value and gradient for
gradient.
"""

import torch

TILE = 16


def tile_rects(means2d, radii, width, height):
    """Tile range [x0, x1) x [y0, y1) covered by each Gaussian's box."""
    tx, ty = -(-width // TILE), -(-height // TILE)
    m = means2d.detach()
    r = radii.float()
    x0 = torch.floor((m[:, 0] - r) / TILE).clamp(0, tx).int()
    x1 = torch.floor((m[:, 0] + r + TILE - 1) / TILE).clamp(0, tx).int()
    y0 = torch.floor((m[:, 1] - r) / TILE).clamp(0, ty).int()
    y1 = torch.floor((m[:, 1] + r + TILE - 1) / TILE).clamp(0, ty).int()
    return x0, x1, y0, y1


def rasterize_reference(means2d, conics, colors, opacities, depths, radii, width, height, bg):
    """Dense O(pixels x Gaussians) version, differentiable by autograd.
    Only for small images: tests and checks."""
    dev = means2d.device
    vis = (radii > 0).nonzero()[:, 0]
    order = vis[torch.argsort(depths[vis], stable=True)]
    m, con, col, op = means2d[order], conics[order], colors[order], opacities[order]
    x0, x1, y0, y1 = (v[order] for v in tile_rects(means2d, radii, width, height))
    py, px = torch.meshgrid(torch.arange(height, device=dev), torch.arange(width, device=dev), indexing="ij")
    px, py = px.reshape(-1), py.reshape(-1)
    dx = m[None, :, 0] - px[:, None].float()
    dy = m[None, :, 1] - py[:, None].float()
    power = -0.5 * (con[None, :, 0] * dx * dx + con[None, :, 2] * dy * dy) - con[None, :, 1] * dx * dy
    alpha = (op[None, :] * torch.exp(power)).clamp(max=0.99)
    tx, ty = (px // TILE)[:, None], (py // TILE)[:, None]
    in_tile = (tx >= x0[None]) & (tx < x1[None]) & (ty >= y0[None]) & (ty < y1[None])
    valid = in_tile & (power <= 0) & (alpha >= 1.0 / 255.0)
    a = torch.where(valid, alpha, torch.zeros_like(alpha))
    # stop once transmittance would drop below 1e-4; that Gaussian and all
    # later ones are left out
    keep = torch.cumprod(1 - a.detach(), dim=1) >= 1e-4
    a = torch.where(keep, a, torch.zeros_like(a))
    one = torch.ones_like(a[:, :1])
    T = torch.cumprod(torch.cat([one, 1 - a], dim=1), dim=1)
    img = (a * T[:, :-1]) @ col + T[:, -1:] * bg[None, :]
    return img.T.reshape(3, height, width)


def bin_gaussians(means2d, depths, radii, width, height):
    """Every (tile, Gaussian) overlap, sorted by tile then depth.

    The Gaussians are put in depth order first and their overlaps listed
    in that order, so one stable sort by tile id finishes the job. Tile ids
    fit in int16 (up to 32,767 tiles, about 4K resolution), and a stable
    int16 sort on MPS is 4.5x faster than sorting a combined int64
    (tile, depth) key.

    Returns ranges [tiles, 2] (start, end into the list) and the list of
    Gaussian ids, both int32."""
    dev = means2d.device
    tx, ty = -(-width // TILE), -(-height // TILE)
    assert tx * ty < 2**15, "image too large for int16 tile ids"
    x0, x1, y0, y1 = tile_rects(means2d, radii, width, height)
    w = (x1 - x0).clamp(min=0)
    counts = w * (y1 - y0).clamp(min=0) * (radii > 0)
    order = torch.argsort(depths.detach(), stable=True)
    counts_sorted = counts[order]
    total = int(counts_sorted.sum())
    if total == 0:
        return torch.zeros(tx * ty, 2, dtype=torch.int32, device=dev), torch.zeros(1, dtype=torch.int32, device=dev)
    gid = torch.repeat_interleave(order.int(), counts_sorted.long(), output_size=total)
    first = torch.cumsum(counts_sorted, 0, dtype=torch.int32) - counts_sorted
    k = torch.arange(total, device=dev, dtype=torch.int32) - torch.repeat_interleave(first, counts_sorted.long(), output_size=total)
    g = gid.long()
    wg = w[g]
    tile = ((y0[g] + k // wg) * tx + x0[g] + k % wg).to(torch.int16)
    tile_sorted, perm = torch.sort(tile, stable=True)
    point_list = gid[perm]
    bounds = torch.searchsorted(tile_sorted.int(), torch.arange(tx * ty + 1, device=dev, dtype=torch.int32))
    ranges = torch.stack([bounds[:-1], bounds[1:]], dim=1).int()
    return ranges, point_list


SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

constant constexpr int BS = 16;          // tile side
constant constexpr int BN = BS * BS;     // threads per tile, Gaussians per batch

kernel void render_forward(
    device const int* ranges [[buffer(0)]],
    device const int* point_list [[buffer(1)]],
    device const float2* means [[buffer(2)]],
    device const float4* conic_op [[buffer(3)]],   // a, b, c, opacity
    device const float* colors [[buffer(4)]],      // [n, 3]
    device const float* bg [[buffer(5)]],
    device float* out [[buffer(6)]],               // [3, h, w]
    device float* final_T [[buffer(7)]],           // [h, w]
    device int* n_contrib [[buffer(8)]],           // [h, w]
    constant int& W [[buffer(9)]],
    constant int& H [[buffer(10)]],
    constant int& tiles_x [[buffer(11)]],
    uint2 tile [[threadgroup_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint tid [[thread_index_in_threadgroup]])
{
    const int px = tile.x * BS + lid.x, py = tile.y * BS + lid.y;
    const bool inside = px < W && py < H;
    const int t = tile.y * tiles_x + tile.x;
    const int start = ranges[2 * t], end = ranges[2 * t + 1];
    const float2 pix = float2(px, py);

    threadgroup int s_id[BN];
    threadgroup float2 s_xy[BN];
    threadgroup float4 s_co[BN];
    threadgroup float3 s_col[BN];
    threadgroup atomic_int n_done;

    bool done = !inside;
    float T = 1.0f;
    float3 C = float3(0.0f);
    int contributor = 0, last = 0;
    int todo = end - start;
    for (int base = start; base < end; base += BN, todo -= BN) {
        // stop early once every pixel of the tile is saturated
        if (tid == 0) atomic_store_explicit(&n_done, 0, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (done) atomic_fetch_add_explicit(&n_done, 1, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (atomic_load_explicit(&n_done, memory_order_relaxed) == BN) break;

        const int i = base + int(tid);
        if (i < end) {
            const int g = point_list[i];
            s_id[tid] = g;
            s_xy[tid] = means[g];
            s_co[tid] = conic_op[g];
            s_col[tid] = float3(colors[3 * g], colors[3 * g + 1], colors[3 * g + 2]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int n = min(BN, todo);
        for (int j = 0; !done && j < n; j++) {
            contributor++;
            const float2 d = s_xy[j] - pix;
            const float4 co = s_co[j];
            const float power = -0.5f * (co.x * d.x * d.x + co.z * d.y * d.y) - co.y * d.x * d.y;
            if (power > 0.0f) continue;
            const float alpha = min(0.99f, co.w * exp(power));
            if (alpha < 1.0f / 255.0f) continue;
            const float test_T = T * (1.0f - alpha);
            if (test_T < 0.0001f) { done = true; continue; }
            C += s_col[j] * (alpha * T);
            T = test_T;
            last = contributor;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (inside) {
        const int p = py * W + px;
        out[p] = C.x + T * bg[0];
        out[H * W + p] = C.y + T * bg[1];
        out[2 * H * W + p] = C.z + T * bg[2];
        final_T[p] = T;
        n_contrib[p] = last;
    }
}

// Walks each pixel's Gaussians back to front, recovering T_i = T_{i+1} /
// (1 - alpha_i) and the colour accumulated behind each Gaussian. A
// simdgroup's 32 pixels always look at the same Gaussian at the same
// time, so their gradients are summed with simd_sum and written with 9
// atomics per simdgroup instead of 9 per pixel.
kernel void render_backward(
    device const int* ranges [[buffer(0)]],
    device const int* point_list [[buffer(1)]],
    device const float2* means [[buffer(2)]],
    device const float4* conic_op [[buffer(3)]],
    device const float* colors [[buffer(4)]],
    device const float* bg [[buffer(5)]],
    device const float* final_T [[buffer(6)]],
    device const int* n_contrib [[buffer(7)]],
    device const float* dL_dout [[buffer(8)]],     // [3, h, w]
    device atomic_float* g_means [[buffer(9)]],    // [n, 2]
    device atomic_float* g_conic [[buffer(10)]],   // [n, 3]
    device atomic_float* g_op [[buffer(11)]],      // [n]
    device atomic_float* g_col [[buffer(12)]],     // [n, 3]
    constant int& W [[buffer(13)]],
    constant int& H [[buffer(14)]],
    constant int& tiles_x [[buffer(15)]],
    uint2 tile [[threadgroup_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const int px = tile.x * BS + lid.x, py = tile.y * BS + lid.y;
    const bool inside = px < W && py < H;
    const int t = tile.y * tiles_x + tile.x;
    const int start = ranges[2 * t], end = ranges[2 * t + 1];
    const float2 pix = float2(px, py);
    const int p = py * W + px;

    threadgroup int s_id[BN];
    threadgroup float2 s_xy[BN];
    threadgroup float4 s_co[BN];
    threadgroup float3 s_col[BN];

    const float T_final = inside ? final_T[p] : 0.0f;
    float T = T_final;
    const int last = inside ? n_contrib[p] : 0;
    const float3 dL = inside ? float3(dL_dout[p], dL_dout[H * W + p], dL_dout[2 * H * W + p]) : float3(0.0f);
    const float bg_dot = dot(float3(bg[0], bg[1], bg[2]), dL);
    float3 accum = float3(0.0f), last_color = float3(0.0f);
    float last_alpha = 0.0f;
    int contributor = end - start;
    int todo = end - start;

    for (int base = end; base > start; base -= BN, todo -= BN) {
        const int i = base - 1 - int(tid);
        if (i >= start) {
            const int g = point_list[i];
            s_id[tid] = g;
            s_xy[tid] = means[g];
            s_co[tid] = conic_op[g];
            s_col[tid] = float3(colors[3 * g], colors[3 * g + 1], colors[3 * g + 2]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int n = min(BN, todo);
        for (int j = 0; j < n; j++) {
            contributor--;
            bool active = inside && contributor < last;
            const float2 d = s_xy[j] - pix;
            const float4 co = s_co[j];
            const float power = -0.5f * (co.x * d.x * d.x + co.z * d.y * d.y) - co.y * d.x * d.y;
            const float G = exp(power);
            const float raw = co.w * G;
            const float alpha = min(0.99f, raw);
            active = active && power <= 0.0f && alpha >= 1.0f / 255.0f;
            float gm[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};  // mean xy, conic abc, opacity, colour rgb
            if (active) {
                T = T / (1.0f - alpha);
                const float3 c = s_col[j];
                accum = last_alpha * last_color + (1.0f - last_alpha) * accum;
                last_color = c;
                const float3 gc = alpha * T * dL;
                float dL_dalpha = dot(c - accum, dL) * T;
                last_alpha = alpha;
                dL_dalpha += (-T_final / (1.0f - alpha)) * bg_dot;
                // alpha is clamped at 0.99: no gradient flows through the clamp
                const float dL_dG = raw > 0.99f ? 0.0f : co.w * dL_dalpha;
                const float gdx = G * d.x, gdy = G * d.y;
                gm[0] = dL_dG * (-gdx * co.x - gdy * co.y);
                gm[1] = dL_dG * (-gdy * co.z - gdx * co.y);
                gm[2] = -0.5f * gdx * d.x * dL_dG;
                gm[3] = -gdx * d.y * dL_dG;
                gm[4] = -0.5f * gdy * d.y * dL_dG;
                gm[5] = raw > 0.99f ? 0.0f : G * dL_dalpha;
                gm[6] = gc.x; gm[7] = gc.y; gm[8] = gc.z;
            }
            if (!simd_any(active)) continue;
            float v = 0.0f;
            for (int k = 0; k < 9; k++) {
                const float s = simd_sum(gm[k]);
                if (lane == uint(k)) v = s;
            }
            const int g = s_id[j];
            if (lane < 2) atomic_fetch_add_explicit(&g_means[2 * g + lane], v, memory_order_relaxed);
            else if (lane < 5) atomic_fetch_add_explicit(&g_conic[3 * g + lane - 2], v, memory_order_relaxed);
            else if (lane == 5) atomic_fetch_add_explicit(&g_op[g], v, memory_order_relaxed);
            else if (lane < 9) atomic_fetch_add_explicit(&g_col[3 * g + lane - 6], v, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
"""

_lib = None


def _kernels():
    global _lib
    if _lib is None:
        _lib = torch.mps.compile_shader(SOURCE)
    return _lib


class _Rasterize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means2d, conics, colors, opacities, ranges, point_list, width, height, bg):
        lib = _kernels()
        tx, ty = -(-width // TILE), -(-height // TILE)
        means2d = means2d.contiguous()
        conic_op = torch.cat([conics, opacities[:, None]], dim=1).contiguous()
        colors = colors.contiguous()
        out = torch.empty(3, height, width, device=means2d.device)
        final_T = torch.empty(height, width, device=means2d.device)
        n_contrib = torch.empty(height, width, dtype=torch.int32, device=means2d.device)
        lib.render_forward(
            ranges, point_list, means2d, conic_op, colors, bg, out, final_T, n_contrib, width, height, tx,
            threads=(tx * TILE, ty * TILE, 1), group_size=(TILE, TILE, 1),
        )
        ctx.save_for_backward(ranges, point_list, means2d, conic_op, colors, bg, final_T, n_contrib)
        ctx.size = (width, height)
        return out

    @staticmethod
    def backward(ctx, dout):
        lib = _kernels()
        ranges, point_list, means2d, conic_op, colors, bg, final_T, n_contrib = ctx.saved_tensors
        width, height = ctx.size
        tx, ty = -(-width // TILE), -(-height // TILE)
        n = means2d.shape[0]
        dev = means2d.device
        g_means = torch.zeros(n, 2, device=dev)
        g_conic = torch.zeros(n, 3, device=dev)
        g_op = torch.zeros(n, device=dev)
        g_col = torch.zeros(n, 3, device=dev)
        lib.render_backward(
            ranges, point_list, means2d, conic_op, colors, bg, final_T, n_contrib, dout.contiguous(),
            g_means, g_conic, g_op, g_col, width, height, tx,
            threads=(tx * TILE, ty * TILE, 1), group_size=(TILE, TILE, 1),
        )
        return g_means, g_conic, g_col, g_op, None, None, None, None, None


def rasterize(means2d, conics, colors, opacities, depths, radii, width, height, bg):
    """Renders [3, h, w] with the Metal kernels."""
    ranges, point_list = bin_gaussians(means2d, depths, radii, width, height)
    return _Rasterize.apply(means2d, conics, colors, opacities, ranges, point_list, width, height, bg.contiguous())
