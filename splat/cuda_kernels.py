"""The Metal kernels of raster.py and project_metal.py, ported to CUDA.

Same algorithms, line for line where possible: threadgroup memory becomes
shared memory, simd_sum becomes a warp butterfly reduction, the
threadgroup atomic counter for the early exit becomes __syncthreads_count.
Compiled on first use with torch.utils.cpp_extension.load_inline (needs
nvcc); TORCH_EXTENSIONS_DIR chooses where the build goes.
"""

import torch

from .project import NEAR, SH_C0, SH_C1, SH_C2, SH_C3

CUDA_SRC = (
    r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// ---------------------------------------------------------------- helpers
__device__ __forceinline__ float2 operator-(float2 a, float2 b) { return make_float2(a.x - b.x, a.y - b.y); }
__device__ __forceinline__ float3 operator+(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ __forceinline__ float3 operator-(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ __forceinline__ float3 operator*(float3 a, float s) { return make_float3(a.x * s, a.y * s, a.z * s); }
__device__ __forceinline__ float3 operator*(float s, float3 a) { return a * s; }
__device__ __forceinline__ float3 operator/(float3 a, float s) { return make_float3(a.x / s, a.y / s, a.z / s); }
__device__ __forceinline__ void operator+=(float3& a, float3 b) { a.x += b.x; a.y += b.y; a.z += b.z; }
__device__ __forceinline__ float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ __forceinline__ float length(float3 a) { return sqrtf(dot(a, a)); }
__device__ __forceinline__ float4 operator-(float4 a, float4 b) { return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w); }
__device__ __forceinline__ float4 operator*(float4 a, float s) { return make_float4(a.x * s, a.y * s, a.z * s, a.w * s); }
__device__ __forceinline__ float4 operator/(float4 a, float s) { return make_float4(a.x / s, a.y / s, a.z / s, a.w / s); }
__device__ __forceinline__ float dot4(float4 a, float4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }

struct mat3 { float3 c[3]; };  // three columns, like Metal's float3x3
__device__ __forceinline__ mat3 mk(float3 c0, float3 c1, float3 c2) { mat3 m; m.c[0] = c0; m.c[1] = c1; m.c[2] = c2; return m; }
__device__ __forceinline__ float3 mv(const mat3& m, float3 v) { return m.c[0] * v.x + m.c[1] * v.y + m.c[2] * v.z; }
__device__ __forceinline__ mat3 mm(const mat3& a, const mat3& b) { return mk(mv(a, b.c[0]), mv(a, b.c[1]), mv(a, b.c[2])); }
__device__ __forceinline__ mat3 tr(const mat3& m) {
    return mk(make_float3(m.c[0].x, m.c[1].x, m.c[2].x), make_float3(m.c[0].y, m.c[1].y, m.c[2].y),
              make_float3(m.c[0].z, m.c[1].z, m.c[2].z));
}
__device__ __forceinline__ float comp(float3 v, int k) { return k == 0 ? v.x : (k == 1 ? v.y : v.z); }

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// ---------------------------------------------------------------- raster
constexpr int BS = 16;
constexpr int BN = BS * BS;

__global__ void render_forward_k(
    const int* __restrict__ ranges, const int* __restrict__ point_list, const float2* __restrict__ means,
    const float4* __restrict__ conic_op, const float* __restrict__ colors, const float* __restrict__ bg,
    float* __restrict__ out, float* __restrict__ final_T, int* __restrict__ n_contrib, int W, int H, int tiles_x)
{
    const int px = blockIdx.x * BS + threadIdx.x, py = blockIdx.y * BS + threadIdx.y;
    const int tid = threadIdx.y * BS + threadIdx.x;
    const bool inside = px < W && py < H;
    const int t = blockIdx.y * tiles_x + blockIdx.x;
    const int start = ranges[2 * t], end = ranges[2 * t + 1];
    const float2 pix = make_float2(px, py);
    __shared__ int s_id[BN];
    __shared__ float2 s_xy[BN];
    __shared__ float4 s_co[BN];
    __shared__ float3 s_col[BN];
    bool done = !inside;
    float T = 1.f;
    float3 C = make_float3(0.f, 0.f, 0.f);
    int contributor = 0, last = 0;
    int todo = end - start;
    for (int base = start; base < end; base += BN, todo -= BN) {
        // also the barrier before the shared arrays are overwritten
        if (__syncthreads_count(done) == BN) break;
        const int i = base + tid;
        if (i < end) {
            const int g = point_list[i];
            s_id[tid] = g;
            s_xy[tid] = means[g];
            s_co[tid] = conic_op[g];
            s_col[tid] = make_float3(colors[3 * g], colors[3 * g + 1], colors[3 * g + 2]);
        }
        __syncthreads();
        const int n = min(BN, todo);
        for (int j = 0; !done && j < n; j++) {
            contributor++;
            const float2 d = s_xy[j] - pix;
            const float4 co = s_co[j];
            const float power = -0.5f * (co.x * d.x * d.x + co.z * d.y * d.y) - co.y * d.x * d.y;
            if (power > 0.f) continue;
            const float alpha = fminf(0.99f, co.w * __expf(power));
            if (alpha < 1.f / 255.f) continue;
            const float test_T = T * (1.f - alpha);
            if (test_T < 0.0001f) { done = true; continue; }
            C += s_col[j] * (alpha * T);
            T = test_T;
            last = contributor;
        }
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

__global__ void render_backward_k(
    const int* __restrict__ ranges, const int* __restrict__ point_list, const float2* __restrict__ means,
    const float4* __restrict__ conic_op, const float* __restrict__ colors, const float* __restrict__ bg,
    const float* __restrict__ final_T, const int* __restrict__ n_contrib, const float* __restrict__ dL_dout,
    float* g_means, float* g_conic, float* g_op, float* g_col, int W, int H, int tiles_x, int exact_clamp)
{
    const int px = blockIdx.x * BS + threadIdx.x, py = blockIdx.y * BS + threadIdx.y;
    const int tid = threadIdx.y * BS + threadIdx.x;
    const int lane = tid & 31;
    const bool inside = px < W && py < H;
    const int t = blockIdx.y * tiles_x + blockIdx.x;
    const int start = ranges[2 * t], end = ranges[2 * t + 1];
    const float2 pix = make_float2(px, py);
    const int p = py * W + px;
    __shared__ int s_id[BN];
    __shared__ float2 s_xy[BN];
    __shared__ float4 s_co[BN];
    __shared__ float3 s_col[BN];

    const float T_final = inside ? final_T[p] : 0.f;
    float T = T_final;
    const int last = inside ? n_contrib[p] : 0;
    const float3 dL = inside ? make_float3(dL_dout[p], dL_dout[H * W + p], dL_dout[2 * H * W + p])
                             : make_float3(0.f, 0.f, 0.f);
    const float bg_dot = dot(make_float3(bg[0], bg[1], bg[2]), dL);
    float3 accum = make_float3(0.f, 0.f, 0.f), last_color = make_float3(0.f, 0.f, 0.f);
    float last_alpha = 0.f;
    int contributor = end - start;
    int todo = end - start;
    for (int base = end; base > start; base -= BN, todo -= BN) {
        __syncthreads();
        const int i = base - 1 - tid;
        if (i >= start) {
            const int g = point_list[i];
            s_id[tid] = g;
            s_xy[tid] = means[g];
            s_co[tid] = conic_op[g];
            s_col[tid] = make_float3(colors[3 * g], colors[3 * g + 1], colors[3 * g + 2]);
        }
        __syncthreads();
        const int n = min(BN, todo);
        for (int j = 0; j < n; j++) {
            contributor--;
            bool active = inside && contributor < last;
            const float2 d = s_xy[j] - pix;
            const float4 co = s_co[j];
            const float power = -0.5f * (co.x * d.x * d.x + co.z * d.y * d.y) - co.y * d.x * d.y;
            const float G = __expf(power);
            const float raw = co.w * G;
            const float alpha = fminf(0.99f, raw);
            active = active && power <= 0.f && alpha >= 1.f / 255.f;
            float gm[9] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            if (active) {
                T = T / (1.f - alpha);
                const float3 c = s_col[j];
                accum = last_alpha * last_color + (1.f - last_alpha) * accum;
                last_color = c;
                const float3 gc = dL * (alpha * T);
                float dL_dalpha = dot(c - accum, dL) * T;
                last_alpha = alpha;
                dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot;
                const bool cut = exact_clamp != 0 && raw > 0.99f;
                const float dL_dG = cut ? 0.f : co.w * dL_dalpha;
                const float gdx = G * d.x, gdy = G * d.y;
                gm[0] = dL_dG * (-gdx * co.x - gdy * co.y);
                gm[1] = dL_dG * (-gdy * co.z - gdx * co.y);
                gm[2] = -0.5f * gdx * d.x * dL_dG;
                gm[3] = -gdx * d.y * dL_dG;
                gm[4] = -0.5f * gdy * d.y * dL_dG;
                gm[5] = cut ? 0.f : G * dL_dalpha;
                gm[6] = gc.x; gm[7] = gc.y; gm[8] = gc.z;
            }
            if (!__any_sync(0xffffffffu, active)) continue;
            float v = 0.f;
            #pragma unroll
            for (int k = 0; k < 9; k++) {
                const float s = warp_sum(gm[k]);
                if (lane == k) v = s;
            }
            const int g = s_id[j];
            if (lane < 2) atomicAdd(&g_means[2 * g + lane], v);
            else if (lane < 5) atomicAdd(&g_conic[3 * g + lane - 2], v);
            else if (lane == 5) atomicAdd(&g_op[g], v);
            else if (lane < 9) atomicAdd(&g_col[3 * g + lane - 6], v);
        }
    }
}

// ---------------------------------------------------------------- projection
struct Cam {
    float R[9];
    float t[3];
    float fx, fy, cx, cy;
    float W, H;
    float C[3];
    float limx, limy;
    float degree;
};
"""
    + f"""
#define NEAR {NEAR}f
#define SC0 {SH_C0}f
#define SC1 {SH_C1}f
__constant__ float SC2[5] = {{{", ".join(f"{c}f" for c in SH_C2)}}};
__constant__ float SC3[7] = {{{", ".join(f"{c}f" for c in SH_C3)}}};
"""
    + r"""
__device__ void sh_basis(float3 d, int deg, float* Y) {
    const float x = d.x, y = d.y, z = d.z;
    Y[0] = SC0;
    if (deg < 1) return;
    Y[1] = -SC1 * y; Y[2] = SC1 * z; Y[3] = -SC1 * x;
    if (deg < 2) return;
    const float xx = x * x, yy = y * y, zz = z * z;
    Y[4] = SC2[0] * x * y; Y[5] = SC2[1] * y * z; Y[6] = SC2[2] * (2 * zz - xx - yy);
    Y[7] = SC2[3] * x * z; Y[8] = SC2[4] * (xx - yy);
    if (deg < 3) return;
    Y[9] = SC3[0] * y * (3 * xx - yy); Y[10] = SC3[1] * x * y * z; Y[11] = SC3[2] * y * (4 * zz - xx - yy);
    Y[12] = SC3[3] * z * (2 * zz - 3 * xx - 3 * yy); Y[13] = SC3[4] * x * (4 * zz - xx - yy);
    Y[14] = SC3[5] * z * (xx - yy); Y[15] = SC3[6] * x * (xx - 3 * yy);
}

__device__ void sh_basis_grad(float3 d, int deg, float3* dY) {
    const float x = d.x, y = d.y, z = d.z;
    for (int k = 0; k < 16; k++) dY[k] = make_float3(0.f, 0.f, 0.f);
    if (deg < 1) return;
    dY[1] = make_float3(0, -SC1, 0); dY[2] = make_float3(0, 0, SC1); dY[3] = make_float3(-SC1, 0, 0);
    if (deg < 2) return;
    dY[4] = SC2[0] * make_float3(y, x, 0);
    dY[5] = SC2[1] * make_float3(0, z, y);
    dY[6] = SC2[2] * make_float3(-2 * x, -2 * y, 4 * z);
    dY[7] = SC2[3] * make_float3(z, 0, x);
    dY[8] = SC2[4] * make_float3(2 * x, -2 * y, 0);
    if (deg < 3) return;
    const float xx = x * x, yy = y * y, zz = z * z;
    dY[9] = SC3[0] * make_float3(6 * x * y, 3 * xx - 3 * yy, 0);
    dY[10] = SC3[1] * make_float3(y * z, x * z, x * y);
    dY[11] = SC3[2] * make_float3(-2 * x * y, 4 * zz - xx - 3 * yy, 8 * y * z);
    dY[12] = SC3[3] * make_float3(-6 * x * z, -6 * y * z, 6 * zz - 3 * xx - 3 * yy);
    dY[13] = SC3[4] * make_float3(4 * zz - 3 * xx - yy, -2 * x * y, 8 * x * z);
    dY[14] = SC3[5] * make_float3(2 * x * z, -2 * y * z, xx - yy);
    dY[15] = SC3[6] * make_float3(3 * xx - 3 * yy, -6 * x * y, 0);
}

__device__ mat3 quat_rot(float4 q) {  // q = (w, x, y, z), normalised
    const float w = q.x, x = q.y, y = q.z, z = q.w;
    return mk(make_float3(1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)),
              make_float3(2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)),
              make_float3(2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)));
}

__device__ __forceinline__ mat3 cam_rot(const Cam& cam) {
    return mk(make_float3(cam.R[0], cam.R[3], cam.R[6]), make_float3(cam.R[1], cam.R[4], cam.R[7]),
              make_float3(cam.R[2], cam.R[5], cam.R[8]));
}

__global__ void project_forward_k(
    const float* __restrict__ means, const float* __restrict__ log_scales, const float* __restrict__ quats,
    const float* __restrict__ sh, const Cam* __restrict__ camp, float* means2d, float* conics, float* depths,
    int* radii, float* colors, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const Cam cam = *camp;
    const float3 m = make_float3(means[3 * i], means[3 * i + 1], means[3 * i + 2]);
    const mat3 Rc = cam_rot(cam);
    const float3 p = mv(Rc, m) + make_float3(cam.t[0], cam.t[1], cam.t[2]);
    radii[i] = 0;
    depths[i] = p.z;
    const float3 v = m - make_float3(cam.C[0], cam.C[1], cam.C[2]);
    const float3 dir = v / length(v);
    float Y[16];
    const int deg = (int)cam.degree;
    sh_basis(dir, deg, Y);
    float3 rgb = make_float3(0.f, 0.f, 0.f);
    const int nb = (deg + 1) * (deg + 1);
    for (int k = 0; k < nb; k++)
        rgb += Y[k] * make_float3(sh[48 * i + 3 * k], sh[48 * i + 3 * k + 1], sh[48 * i + 3 * k + 2]);
    colors[3 * i] = fmaxf(rgb.x + 0.5f, 0.f);
    colors[3 * i + 1] = fmaxf(rgb.y + 0.5f, 0.f);
    colors[3 * i + 2] = fmaxf(rgb.z + 0.5f, 0.f);

    const bool front = p.z > NEAR;
    const float z = front ? p.z : 1.f;
    float4 q = make_float4(quats[4 * i], quats[4 * i + 1], quats[4 * i + 2], quats[4 * i + 3]);
    q = q / sqrtf(dot4(q, q));
    const float3 s = make_float3(__expf(log_scales[3 * i]), __expf(log_scales[3 * i + 1]), __expf(log_scales[3 * i + 2]));
    const mat3 Rq = quat_rot(q);
    const mat3 M = mk(Rq.c[0] * s.x, Rq.c[1] * s.y, Rq.c[2] * s.z);
    const mat3 S = mm(M, tr(M));
    const mat3 Sc = mm(mm(Rc, S), tr(Rc));
    const float tx = fminf(fmaxf(p.x / z, -cam.limx), cam.limx) * z;
    const float ty = fminf(fmaxf(p.y / z, -cam.limy), cam.limy) * z;
    const float3 j0 = make_float3(cam.fx / z, 0.f, -cam.fx * tx / (z * z));
    const float3 j1 = make_float3(0.f, cam.fy / z, -cam.fy * ty / (z * z));
    const float a = dot(j0, mv(Sc, j0)) + 0.3f;
    const float b = dot(j0, mv(Sc, j1));
    const float c = dot(j1, mv(Sc, j1)) + 0.3f;
    const float det = a * c - b * b;
    const float mx = cam.fx * p.x / z + cam.cx - 0.5f, my = cam.fy * p.y / z + cam.cy - 0.5f;
    means2d[2 * i] = mx; means2d[2 * i + 1] = my;
    const float dd = (front && det > 0.f) ? det : 1.f;
    conics[3 * i] = c / dd; conics[3 * i + 1] = -b / dd; conics[3 * i + 2] = a / dd;
    if (!front || det <= 0.f) return;
    const float mid = 0.5f * (a + c);
    const float lam = mid + sqrtf(fmaxf(mid * mid - det, 0.1f));
    const float r = ceilf(3.f * sqrtf(lam));
    if (mx + r > 0 && mx - r < cam.W && my + r > 0 && my - r < cam.H) radii[i] = (int)r;
}

__global__ void project_backward_k(
    const float* __restrict__ means, const float* __restrict__ log_scales, const float* __restrict__ quats,
    const float* __restrict__ sh, const Cam* __restrict__ camp, const int* __restrict__ radii,
    const float* __restrict__ g_m2, const float* __restrict__ g_con, const float* __restrict__ g_col,
    float* g_means, float* g_ls, float* g_q, float* g_sh, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const Cam cam = *camp;
    const float3 m = make_float3(means[3 * i], means[3 * i + 1], means[3 * i + 2]);
    float3 gm = make_float3(0.f, 0.f, 0.f);

    // colour
    const float3 v = m - make_float3(cam.C[0], cam.C[1], cam.C[2]);
    const float vl = length(v);
    const float3 dir = v / vl;
    const int deg = (int)cam.degree;
    const int nb = (deg + 1) * (deg + 1);
    float Y[16];
    sh_basis(dir, deg, Y);
    float3 rgb = make_float3(0.5f, 0.5f, 0.5f);
    for (int k = 0; k < nb; k++)
        rgb += Y[k] * make_float3(sh[48 * i + 3 * k], sh[48 * i + 3 * k + 1], sh[48 * i + 3 * k + 2]);
    float3 gc = make_float3(rgb.x < 0.f ? 0.f : g_col[3 * i], rgb.y < 0.f ? 0.f : g_col[3 * i + 1],
                            rgb.z < 0.f ? 0.f : g_col[3 * i + 2]);
    for (int k = 0; k < 16; k++) {
        const float3 gk = k < nb ? Y[k] * gc : make_float3(0.f, 0.f, 0.f);
        g_sh[48 * i + 3 * k] = gk.x; g_sh[48 * i + 3 * k + 1] = gk.y; g_sh[48 * i + 3 * k + 2] = gk.z;
    }
    if (deg > 0) {
        float3 dY[16];
        sh_basis_grad(dir, deg, dY);
        float3 gdir = make_float3(0.f, 0.f, 0.f);
        for (int k = 1; k < nb; k++) {
            const float3 shk = make_float3(sh[48 * i + 3 * k], sh[48 * i + 3 * k + 1], sh[48 * i + 3 * k + 2]);
            gdir += dY[k] * dot(shk, gc);
        }
        gm += (gdir - dir * dot(dir, gdir)) / vl;
    }

    float3 gls = make_float3(0.f, 0.f, 0.f);
    float4 gq = make_float4(0.f, 0.f, 0.f, 0.f);
    const mat3 Rc = cam_rot(cam);
    const float3 p = mv(Rc, m) + make_float3(cam.t[0], cam.t[1], cam.t[2]);
    if (radii[i] > 0) {
        const float z = p.z;
        const float g2x = g_m2[2 * i], g2y = g_m2[2 * i + 1];
        float3 gp = make_float3(cam.fx / z * g2x, cam.fy / z * g2y,
                                -cam.fx * p.x / (z * z) * g2x - cam.fy * p.y / (z * z) * g2y);
        const float4 qraw = make_float4(quats[4 * i], quats[4 * i + 1], quats[4 * i + 2], quats[4 * i + 3]);
        const float ql = sqrtf(dot4(qraw, qraw));
        const float4 q = qraw / ql;
        const float3 s = make_float3(__expf(log_scales[3 * i]), __expf(log_scales[3 * i + 1]), __expf(log_scales[3 * i + 2]));
        const mat3 Rq = quat_rot(q);
        const mat3 M = mk(Rq.c[0] * s.x, Rq.c[1] * s.y, Rq.c[2] * s.z);
        const mat3 S = mm(M, tr(M));
        const float ux = p.x / z, uy = p.y / z;
        const bool clx = ux < -cam.limx || ux > cam.limx;
        const bool cly = uy < -cam.limy || uy > cam.limy;
        const float cux = fminf(fmaxf(ux, -cam.limx), cam.limx), cuy = fminf(fmaxf(uy, -cam.limy), cam.limy);
        const float tx = cux * z, ty = cuy * z;
        const float3 j0 = make_float3(cam.fx / z, 0.f, -cam.fx * tx / (z * z));
        const float3 j1 = make_float3(0.f, cam.fy / z, -cam.fy * ty / (z * z));
        const mat3 Rt = tr(Rc);
        const float3 t0 = mv(Rt, j0), t1 = mv(Rt, j1);
        const float a = dot(t0, mv(S, t0)) + 0.3f, b = dot(t0, mv(S, t1)), c = dot(t1, mv(S, t1)) + 0.3f;
        const float det = a * c - b * b;
        const float id2 = 1.f / (det * det);
        const float gA = g_con[3 * i], gB = g_con[3 * i + 1], gC = g_con[3 * i + 2];
        const float ga = (-c * c * gA + b * c * gB - b * b * gC) * id2;
        const float gb = (2 * b * c * gA - (a * c + b * b) * gB + 2 * a * b * gC) * id2;
        const float gcc = (-b * b * gA + a * b * gB - a * a * gC) * id2;
        const float3 Gt0 = ga * t0 + 0.5f * gb * t1;
        const float3 Gt1 = 0.5f * gb * t0 + gcc * t1;
        const mat3 gS = mk(t0 * Gt0.x + t1 * Gt1.x, t0 * Gt0.y + t1 * Gt1.y, t0 * Gt0.z + t1 * Gt1.z);
        const float3 gT0 = 2.f * mv(S, Gt0), gT1 = 2.f * mv(S, Gt1);
        const float3 gJ0 = mv(Rc, gT0), gJ1 = mv(Rc, gT1);
        const float dtx_dx = clx ? 0.f : 1.f, dtx_dz = clx ? cux : 0.f;
        const float dty_dy = cly ? 0.f : 1.f, dty_dz = cly ? cuy : 0.f;
        gp.x += gJ0.z * (-cam.fx / (z * z)) * dtx_dx;
        gp.y += gJ1.z * (-cam.fy / (z * z)) * dty_dy;
        gp.z += gJ0.x * (-cam.fx / (z * z)) + gJ1.y * (-cam.fy / (z * z))
              + gJ0.z * (-cam.fx * dtx_dz / (z * z) + 2.f * cam.fx * tx / (z * z * z))
              + gJ1.z * (-cam.fy * dty_dz / (z * z) + 2.f * cam.fy * ty / (z * z * z));
        gm += mv(Rt, gp);

        const mat3 gMm = mm(gS, M);
        const mat3 gM = mk(gMm.c[0] * 2.f, gMm.c[1] * 2.f, gMm.c[2] * 2.f);
        gls = make_float3(dot(gM.c[0], Rq.c[0]) * s.x, dot(gM.c[1], Rq.c[1]) * s.y, dot(gM.c[2], Rq.c[2]) * s.z);
        const mat3 gR = mk(gM.c[0] * s.x, gM.c[1] * s.y, gM.c[2] * s.z);
        // G_rc: row r, column c
        const float G00 = comp(gR.c[0], 0), G01 = comp(gR.c[1], 0), G02 = comp(gR.c[2], 0);
        const float G10 = comp(gR.c[0], 1), G11 = comp(gR.c[1], 1), G12 = comp(gR.c[2], 1);
        const float G20 = comp(gR.c[0], 2), G21 = comp(gR.c[1], 2), G22 = comp(gR.c[2], 2);
        const float w = q.x, x = q.y, y = q.z, zq = q.w;
        const float4 gqn = make_float4(
            2.f * (-zq * G01 + y * G02 + zq * G10 - x * G12 - y * G20 + x * G21),
            2.f * (y * G01 + zq * G02 + y * G10 - 2 * x * G11 - w * G12 + zq * G20 + w * G21 - 2 * x * G22),
            2.f * (-2 * y * G00 + x * G01 + w * G02 + x * G10 + zq * G12 - w * G20 + zq * G21 - 2 * y * G22),
            2.f * (-2 * zq * G00 - w * G01 + x * G02 + w * G10 - 2 * zq * G11 + y * G12 + x * G20 + y * G21));
        gq = (gqn - q * dot4(q, gqn)) / ql;
    }
    g_means[3 * i] = gm.x; g_means[3 * i + 1] = gm.y; g_means[3 * i + 2] = gm.z;
    g_ls[3 * i] = gls.x; g_ls[3 * i + 1] = gls.y; g_ls[3 * i + 2] = gls.z;
    g_q[4 * i] = gq.x; g_q[4 * i + 1] = gq.y; g_q[4 * i + 2] = gq.z; g_q[4 * i + 3] = gq.w;
}

// ---------------------------------------------------------------- launchers
#define F(t) (t).data_ptr<float>()
#define I(t) (t).data_ptr<int>()

void render_forward(torch::Tensor ranges, torch::Tensor point_list, torch::Tensor means, torch::Tensor conic_op,
                    torch::Tensor colors, torch::Tensor bg, torch::Tensor out, torch::Tensor final_T,
                    torch::Tensor n_contrib, int64_t W, int64_t H, int64_t tiles_x) {
    const int ty = (H + BS - 1) / BS;
    render_forward_k<<<dim3(tiles_x, ty), dim3(BS, BS), 0, at::cuda::getCurrentCUDAStream()>>>(
        I(ranges), I(point_list), reinterpret_cast<const float2*>(F(means)),
        reinterpret_cast<const float4*>(F(conic_op)), F(colors), F(bg), F(out), F(final_T), I(n_contrib),
        W, H, tiles_x);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void render_backward(torch::Tensor ranges, torch::Tensor point_list, torch::Tensor means, torch::Tensor conic_op,
                     torch::Tensor colors, torch::Tensor bg, torch::Tensor final_T, torch::Tensor n_contrib,
                     torch::Tensor dout, torch::Tensor g_means, torch::Tensor g_conic, torch::Tensor g_op,
                     torch::Tensor g_col, int64_t W, int64_t H, int64_t tiles_x, int64_t exact_clamp) {
    const int ty = (H + BS - 1) / BS;
    render_backward_k<<<dim3(tiles_x, ty), dim3(BS, BS), 0, at::cuda::getCurrentCUDAStream()>>>(
        I(ranges), I(point_list), reinterpret_cast<const float2*>(F(means)),
        reinterpret_cast<const float4*>(F(conic_op)), F(colors), F(bg), F(final_T), I(n_contrib), F(dout),
        F(g_means), F(g_conic), F(g_op), F(g_col), W, H, tiles_x, exact_clamp);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void project_forward(torch::Tensor means, torch::Tensor log_scales, torch::Tensor quats, torch::Tensor sh,
                     torch::Tensor cam, torch::Tensor m2, torch::Tensor con, torch::Tensor depth, torch::Tensor radii,
                     torch::Tensor col, int64_t n) {
    if (n == 0) return;
    project_forward_k<<<(n + 127) / 128, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        F(means), F(log_scales), F(quats), F(sh), reinterpret_cast<const Cam*>(F(cam)), F(m2), F(con), F(depth),
        I(radii), F(col), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void project_backward(torch::Tensor means, torch::Tensor log_scales, torch::Tensor quats, torch::Tensor sh,
                      torch::Tensor cam, torch::Tensor radii, torch::Tensor g_m2, torch::Tensor g_con,
                      torch::Tensor g_col, torch::Tensor g_means, torch::Tensor g_ls, torch::Tensor g_q,
                      torch::Tensor g_sh, int64_t n) {
    if (n == 0) return;
    project_backward_k<<<(n + 127) / 128, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        F(means), F(log_scales), F(quats), F(sh), reinterpret_cast<const Cam*>(F(cam)), I(radii), F(g_m2),
        F(g_con), F(g_col), F(g_means), F(g_ls), F(g_q), F(g_sh), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""
)

CPP_DECL = """
void render_forward(torch::Tensor ranges, torch::Tensor point_list, torch::Tensor means, torch::Tensor conic_op,
                    torch::Tensor colors, torch::Tensor bg, torch::Tensor out, torch::Tensor final_T,
                    torch::Tensor n_contrib, int64_t W, int64_t H, int64_t tiles_x);
void render_backward(torch::Tensor ranges, torch::Tensor point_list, torch::Tensor means, torch::Tensor conic_op,
                     torch::Tensor colors, torch::Tensor bg, torch::Tensor final_T, torch::Tensor n_contrib,
                     torch::Tensor dout, torch::Tensor g_means, torch::Tensor g_conic, torch::Tensor g_op,
                     torch::Tensor g_col, int64_t W, int64_t H, int64_t tiles_x, int64_t exact_clamp);
void project_forward(torch::Tensor means, torch::Tensor log_scales, torch::Tensor quats, torch::Tensor sh,
                     torch::Tensor cam, torch::Tensor m2, torch::Tensor con, torch::Tensor depth, torch::Tensor radii,
                     torch::Tensor col, int64_t n);
void project_backward(torch::Tensor means, torch::Tensor log_scales, torch::Tensor quats, torch::Tensor sh,
                      torch::Tensor cam, torch::Tensor radii, torch::Tensor g_m2, torch::Tensor g_con,
                      torch::Tensor g_col, torch::Tensor g_means, torch::Tensor g_ls, torch::Tensor g_q,
                      torch::Tensor g_sh, int64_t n);
"""

_mod = None


def module():
    global _mod
    if _mod is None:
        from torch.utils.cpp_extension import load_inline

        _mod = load_inline(
            "splat_cuda_ext", cpp_sources=CPP_DECL, cuda_sources=CUDA_SRC,
            functions=["render_forward", "render_backward", "project_forward", "project_backward"],
            extra_cuda_cflags=["-O3"], verbose=False,
        )
    return _mod
