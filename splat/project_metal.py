"""Projection and colour as one Metal kernel per Gaussian, forward and
backward. Computes the same as project.project + project.colors_from_sh
(the reference, which autograd differentiates in the tests), but as one
thread per Gaussian instead of ~150 small tensor ops.

Backward, per Gaussian, given dL/dmean2d, dL/dconic and dL/dcolour:

  colour:  rgb = SH(dir) + 0.5, clamped at 0; dir = (m - C) / |m - C|
           dL/dsh_k = Y_k(dir) dL/drgb, dL/ddir = sum_k grad Y_k (sh_k . dL/drgb)
           dL/dm += (I - dir dir^T) / |m - C| dL/ddir
  mean2d:  u = fx x / z + cx, v = fy y / z + cy, with p = (x, y, z) = R m + t
  conic:   (A, B, C) = (c, -b, a) / (ac - b^2), differentiated to (a, b, c)
  cov2d:   [a b; b c] = T S T^T, T = J R, S = M M^T, M = Rq diag(s)
           dL/dS = T^T G T and dL/dT = 2 G T S, with G = [ga gb/2; gb/2 gc]
           dL/dJ = dL/dT R^T, then J's entries are functions of p
  scale:   dL/dM = 2 dL/dS M, dL/ds_j = sum_i dL/dM_ij Rq_ij, log-scale times s_j
  rotation: dL/dRq_ij = dL/dM_ij s_j, through the quaternion formula and
           its normalisation
"""

import torch

from .project import NEAR, SH_C0, SH_C1, SH_C2, SH_C3

SOURCE = (
    r"""
#include <metal_stdlib>
using namespace metal;

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
constant float NEAR = {NEAR};
constant float SC0 = {SH_C0};
constant float SC1 = {SH_C1};
constant float SC2[5] = {{{", ".join(map(str, SH_C2))}}};
constant float SC3[7] = {{{", ".join(map(str, SH_C3))}}};
"""
    + r"""
// basis values Y[16] at unit direction d
static void sh_basis(float3 d, int deg, thread float* Y) {
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

// partial derivatives of each basis polynomial in x, y, z
static void sh_basis_grad(float3 d, int deg, thread float3* dY) {
    const float x = d.x, y = d.y, z = d.z;
    for (int k = 0; k < 16; k++) dY[k] = float3(0.0f);
    if (deg < 1) return;
    dY[1] = float3(0, -SC1, 0); dY[2] = float3(0, 0, SC1); dY[3] = float3(-SC1, 0, 0);
    if (deg < 2) return;
    dY[4] = SC2[0] * float3(y, x, 0);
    dY[5] = SC2[1] * float3(0, z, y);
    dY[6] = SC2[2] * float3(-2 * x, -2 * y, 4 * z);
    dY[7] = SC2[3] * float3(z, 0, x);
    dY[8] = SC2[4] * float3(2 * x, -2 * y, 0);
    if (deg < 3) return;
    const float xx = x * x, yy = y * y, zz = z * z;
    dY[9] = SC3[0] * float3(6 * x * y, 3 * xx - 3 * yy, 0);
    dY[10] = SC3[1] * float3(y * z, x * z, x * y);
    dY[11] = SC3[2] * float3(-2 * x * y, 4 * zz - xx - 3 * yy, 8 * y * z);
    dY[12] = SC3[3] * float3(-6 * x * z, -6 * y * z, 6 * zz - 3 * xx - 3 * yy);
    dY[13] = SC3[4] * float3(4 * zz - 3 * xx - yy, -2 * x * y, 8 * x * z);
    dY[14] = SC3[5] * float3(2 * x * z, -2 * y * z, xx - yy);
    dY[15] = SC3[6] * float3(3 * xx - 3 * yy, -6 * x * y, 0);
}

static float3x3 quat_rot(float4 q) {  // q = (w, x, y, z), normalised; column-major float3x3
    const float w = q.x, x = q.y, y = q.z, z = q.w;
    return float3x3(
        float3(1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)),
        float3(2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)),
        float3(2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)));
}

kernel void project_forward(
    device const float* means [[buffer(0)]],      // [n, 3]
    device const float* log_scales [[buffer(1)]], // [n, 3]
    device const float* quats [[buffer(2)]],      // [n, 4]
    device const float* sh [[buffer(3)]],         // [n, 16, 3]
    constant Cam& cam [[buffer(4)]],
    device float* means2d [[buffer(5)]],          // [n, 2]
    device float* conics [[buffer(6)]],           // [n, 3]
    device float* depths [[buffer(7)]],           // [n]
    device int* radii [[buffer(8)]],              // [n]
    device float* colors [[buffer(9)]],           // [n, 3]
    constant int& n [[buffer(10)]],
    uint i [[thread_position_in_grid]])
{
    if (int(i) >= n) return;
    const float3 m = float3(means[3 * i], means[3 * i + 1], means[3 * i + 2]);
    const float3x3 Rc = float3x3(float3(cam.R[0], cam.R[3], cam.R[6]), float3(cam.R[1], cam.R[4], cam.R[7]),
                                 float3(cam.R[2], cam.R[5], cam.R[8]));
    const float3 p = Rc * m + float3(cam.t[0], cam.t[1], cam.t[2]);
    radii[i] = 0;
    depths[i] = p.z;
    // colour is computed for every Gaussian, like the reference
    const float3 v = m - float3(cam.C[0], cam.C[1], cam.C[2]);
    const float3 dir = v / length(v);
    float Y[16];
    const int deg = int(cam.degree);
    sh_basis(dir, deg, Y);
    float3 rgb = float3(0.0f);
    const int nb = (deg + 1) * (deg + 1);
    for (int k = 0; k < nb; k++)
        rgb += Y[k] * float3(sh[48 * i + 3 * k], sh[48 * i + 3 * k + 1], sh[48 * i + 3 * k + 2]);
    rgb = max(rgb + 0.5f, 0.0f);
    colors[3 * i] = rgb.x; colors[3 * i + 1] = rgb.y; colors[3 * i + 2] = rgb.z;

    const bool front = p.z > NEAR;
    const float z = front ? p.z : 1.0f;
    float4 q = float4(quats[4 * i], quats[4 * i + 1], quats[4 * i + 2], quats[4 * i + 3]);
    q = q / length(q);
    const float3 s = exp(float3(log_scales[3 * i], log_scales[3 * i + 1], log_scales[3 * i + 2]));
    const float3x3 Rq = quat_rot(q);
    const float3x3 M = float3x3(Rq[0] * s.x, Rq[1] * s.y, Rq[2] * s.z);
    const float3x3 S = M * transpose(M);
    const float3x3 Sc = Rc * S * transpose(Rc);
    const float tx = clamp(p.x / z, -cam.limx, cam.limx) * z;
    const float ty = clamp(p.y / z, -cam.limy, cam.limy) * z;
    // J rows: (fx/z, 0, -fx tx/z^2), (0, fy/z, -fy ty/z^2)
    const float3 j0 = float3(cam.fx / z, 0, -cam.fx * tx / (z * z));
    const float3 j1 = float3(0, cam.fy / z, -cam.fy * ty / (z * z));
    const float a = dot(j0, Sc * j0) + 0.3f;
    const float b = dot(j0, Sc * j1);
    const float c = dot(j1, Sc * j1) + 0.3f;
    const float det = a * c - b * b;
    const float2 m2 = float2(cam.fx * p.x / z + cam.cx - 0.5f, cam.fy * p.y / z + cam.cy - 0.5f);
    means2d[2 * i] = m2.x; means2d[2 * i + 1] = m2.y;
    const float dd = (front && det > 0) ? det : 1.0f;  // the reference's convention for culled ones
    conics[3 * i] = c / dd; conics[3 * i + 1] = -b / dd; conics[3 * i + 2] = a / dd;
    if (!front || det <= 0) return;
    const float mid = 0.5f * (a + c);
    const float lam = mid + sqrt(max(mid * mid - det, 0.1f));
    const float r = ceil(3.0f * sqrt(lam));
    if (m2.x + r > 0 && m2.x - r < cam.W && m2.y + r > 0 && m2.y - r < cam.H) radii[i] = int(r);
}

kernel void project_backward(
    device const float* means [[buffer(0)]],
    device const float* log_scales [[buffer(1)]],
    device const float* quats [[buffer(2)]],
    device const float* sh [[buffer(3)]],
    constant Cam& cam [[buffer(4)]],
    device const int* radii [[buffer(5)]],
    device const float* g_m2 [[buffer(6)]],       // [n, 2]
    device const float* g_con [[buffer(7)]],      // [n, 3]
    device const float* g_col [[buffer(8)]],      // [n, 3]
    device float* g_means [[buffer(9)]],          // [n, 3]
    device float* g_ls [[buffer(10)]],            // [n, 3]
    device float* g_q [[buffer(11)]],             // [n, 4]
    device float* g_sh [[buffer(12)]],            // [n, 16, 3]
    constant int& n [[buffer(13)]],
    uint i [[thread_position_in_grid]])
{
    if (int(i) >= n) return;
    const float3 m = float3(means[3 * i], means[3 * i + 1], means[3 * i + 2]);
    float3 gm = float3(0.0f);

    // colour
    const float3 v = m - float3(cam.C[0], cam.C[1], cam.C[2]);
    const float vl = length(v);
    const float3 dir = v / vl;
    const int deg = int(cam.degree);
    const int nb = (deg + 1) * (deg + 1);
    float Y[16];
    sh_basis(dir, deg, Y);
    float3 rgb = float3(0.0f);
    for (int k = 0; k < nb; k++)
        rgb += Y[k] * float3(sh[48 * i + 3 * k], sh[48 * i + 3 * k + 1], sh[48 * i + 3 * k + 2]);
    rgb += 0.5f;
    float3 gc = float3(g_col[3 * i], g_col[3 * i + 1], g_col[3 * i + 2]);
    gc = select(gc, float3(0.0f), rgb < 0.0f);  // clamped channels pass no gradient
    for (int k = 0; k < 16; k++) {
        const float3 gk = k < nb ? Y[k] * gc : float3(0.0f);
        g_sh[48 * i + 3 * k] = gk.x; g_sh[48 * i + 3 * k + 1] = gk.y; g_sh[48 * i + 3 * k + 2] = gk.z;
    }
    if (deg > 0) {
        float3 dY[16];
        sh_basis_grad(dir, deg, dY);
        float3 gdir = float3(0.0f);
        for (int k = 1; k < nb; k++) {
            const float3 shk = float3(sh[48 * i + 3 * k], sh[48 * i + 3 * k + 1], sh[48 * i + 3 * k + 2]);
            gdir += dY[k] * dot(shk, gc);
        }
        gm += (gdir - dir * dot(dir, gdir)) / vl;
    }

    float3 gls = float3(0.0f);
    float4 gq = float4(0.0f);
    const float3x3 Rc = float3x3(float3(cam.R[0], cam.R[3], cam.R[6]), float3(cam.R[1], cam.R[4], cam.R[7]),
                                 float3(cam.R[2], cam.R[5], cam.R[8]));
    const float3 p = Rc * m + float3(cam.t[0], cam.t[1], cam.t[2]);
    if (radii[i] > 0) {
        const float z = p.z;
        // mean2d
        const float2 g2 = float2(g_m2[2 * i], g_m2[2 * i + 1]);
        float3 gp = float3(cam.fx / z * g2.x, cam.fy / z * g2.y,
                           -cam.fx * p.x / (z * z) * g2.x - cam.fy * p.y / (z * z) * g2.y);

        // recompute the covariance chain
        const float4 qraw = float4(quats[4 * i], quats[4 * i + 1], quats[4 * i + 2], quats[4 * i + 3]);
        const float ql = length(qraw);
        const float4 q = qraw / ql;
        const float3 s = exp(float3(log_scales[3 * i], log_scales[3 * i + 1], log_scales[3 * i + 2]));
        const float3x3 Rq = quat_rot(q);
        const float3x3 M = float3x3(Rq[0] * s.x, Rq[1] * s.y, Rq[2] * s.z);
        const float3x3 S = M * transpose(M);
        const float ux = p.x / z, uy = p.y / z;
        const bool clx = ux < -cam.limx || ux > cam.limx;
        const bool cly = uy < -cam.limy || uy > cam.limy;
        const float tx = clamp(ux, -cam.limx, cam.limx) * z;
        const float ty = clamp(uy, -cam.limy, cam.limy) * z;
        const float3 j0 = float3(cam.fx / z, 0, -cam.fx * tx / (z * z));
        const float3 j1 = float3(0, cam.fy / z, -cam.fy * ty / (z * z));
        // T = J Rc as two row vectors: row r of T = Rc^T applied to row r of J
        const float3 t0 = transpose(Rc) * j0, t1 = transpose(Rc) * j1;
        const float a = dot(t0, S * t0) + 0.3f, b = dot(t0, S * t1), c = dot(t1, S * t1) + 0.3f;
        const float det = a * c - b * b;
        const float id2 = 1.0f / (det * det);
        const float gA = g_con[3 * i], gB = g_con[3 * i + 1], gC = g_con[3 * i + 2];
        // (A, B, C) = (c, -b, a) / det
        const float ga = (-c * c * gA + b * c * gB - b * b * gC) * id2;
        const float gb = (2 * b * c * gA - (a * c + b * b) * gB + 2 * a * b * gC) * id2;
        const float gcc = (-b * b * gA + a * b * gB - a * a * gC) * id2;
        // G = [ga gb/2; gb/2 gc]; dL/dS = T^T G T
        const float3 Gt0 = ga * t0 + 0.5f * gb * t1;  // (G T) row 0
        const float3 Gt1 = 0.5f * gb * t0 + gcc * t1; // (G T) row 1
        const float3x3 gS = float3x3(t0 * Gt0.x + t1 * Gt1.x, t0 * Gt0.y + t1 * Gt1.y, t0 * Gt0.z + t1 * Gt1.z);
        // dL/dT = 2 G T S, rows; then dL/dJ rows = dL/dT rows times Rc^T, i.e. Rc * row
        const float3 gT0 = 2.0f * (S * Gt0), gT1 = 2.0f * (S * Gt1);
        const float3 gJ0 = Rc * gT0, gJ1 = Rc * gT1;
        // J00 = fx/z, J02 = -fx tx/z^2, J11 = fy/z, J12 = -fy ty/z^2
        const float dtx_dx = clx ? 0.0f : 1.0f, dtx_dz = clx ? clamp(ux, -cam.limx, cam.limx) : 0.0f;
        const float dty_dy = cly ? 0.0f : 1.0f, dty_dz = cly ? clamp(uy, -cam.limy, cam.limy) : 0.0f;
        gp.x += gJ0.z * (-cam.fx / (z * z)) * dtx_dx;
        gp.y += gJ1.z * (-cam.fy / (z * z)) * dty_dy;
        gp.z += gJ0.x * (-cam.fx / (z * z)) + gJ1.y * (-cam.fy / (z * z))
              + gJ0.z * (-cam.fx * dtx_dz / (z * z) + 2.0f * cam.fx * tx / (z * z * z))
              + gJ1.z * (-cam.fy * dty_dz / (z * z) + 2.0f * cam.fy * ty / (z * z * z));
        gm += transpose(Rc) * gp;

        // S = M M^T: dL/dM = 2 gS M (gS symmetric)
        const float3x3 gM = 2.0f * gS * M;
        // M = Rq diag(s): column j of M is column j of Rq times s_j
        gls = float3(dot(gM[0], Rq[0]) * s.x, dot(gM[1], Rq[1]) * s.y, dot(gM[2], Rq[2]) * s.z);
        const float3x3 gR = float3x3(gM[0] * s.x, gM[1] * s.y, gM[2] * s.z);
        // gR[col][row]; G_rc = gR[c][r]
        const float G00 = gR[0][0], G01 = gR[1][0], G02 = gR[2][0];
        const float G10 = gR[0][1], G11 = gR[1][1], G12 = gR[2][1];
        const float G20 = gR[0][2], G21 = gR[1][2], G22 = gR[2][2];
        const float w = q.x, x = q.y, y = q.z, zq = q.w;
        const float4 gqn = 2.0f * float4(
            -zq * G01 + y * G02 + zq * G10 - x * G12 - y * G20 + x * G21,
            y * G01 + zq * G02 + y * G10 - 2 * x * G11 - w * G12 + zq * G20 + w * G21 - 2 * x * G22,
            -2 * y * G00 + x * G01 + w * G02 + x * G10 + zq * G12 - w * G20 + zq * G21 - 2 * y * G22,
            -2 * zq * G00 - w * G01 + x * G02 + w * G10 - 2 * zq * G11 + y * G12 + x * G20 + y * G21);
        gq = (gqn - q * dot(q, gqn)) / ql;
    }
    g_means[3 * i] = gm.x; g_means[3 * i + 1] = gm.y; g_means[3 * i + 2] = gm.z;
    g_ls[3 * i] = gls.x; g_ls[3 * i + 1] = gls.y; g_ls[3 * i + 2] = gls.z;
    g_q[4 * i] = gq.x; g_q[4 * i + 1] = gq.y; g_q[4 * i + 2] = gq.z; g_q[4 * i + 3] = gq.w;
}
"""
)

_lib = None


def _kernels():
    global _lib
    if _lib is None:
        _lib = torch.mps.compile_shader(SOURCE)
    return _lib


def cam_buffer(cam, degree):
    R = cam.R.reshape(-1).float().cpu()
    t = cam.t.float().cpu()
    C = cam.center.float().cpu()
    limx = 1.3 * cam.width / (2 * cam.fx)
    limy = 1.3 * cam.height / (2 * cam.fy)
    f32 = dict(dtype=torch.float32)
    vals = torch.cat([R, t, torch.tensor([cam.fx, cam.fy, cam.cx, cam.cy, cam.width, cam.height], **f32), C,
                      torch.tensor([limx, limy, float(degree)], **f32)])
    return vals.to(cam.R.device)


class _Project(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means, log_scales, quats, sh, cambuf):
        lib = _kernels()
        n = means.shape[0]
        dev = means.device
        means, log_scales, quats, sh = (x.contiguous() for x in (means, log_scales, quats, sh))
        m2 = torch.empty(n, 2, device=dev)
        con = torch.empty(n, 3, device=dev)
        depth = torch.empty(n, device=dev)
        radii = torch.empty(n, dtype=torch.int32, device=dev)
        col = torch.empty(n, 3, device=dev)
        lib.project_forward(means, log_scales, quats, sh, cambuf, m2, con, depth, radii, col, n, threads=n)
        ctx.save_for_backward(means, log_scales, quats, sh, cambuf, radii)
        ctx.mark_non_differentiable(depth, radii)
        return m2, con, depth, radii, col

    @staticmethod
    def backward(ctx, g_m2, g_con, _gd, _gr, g_col):
        lib = _kernels()
        means, log_scales, quats, sh, cambuf, radii = ctx.saved_tensors
        n = means.shape[0]
        g_means = torch.empty_like(means)
        g_ls = torch.empty_like(log_scales)
        g_q = torch.empty_like(quats)
        g_sh = torch.empty_like(sh)
        z = torch.zeros(n, 3, device=means.device)
        lib.project_backward(
            means, log_scales, quats, sh, cambuf, radii,
            g_m2.contiguous() if g_m2 is not None else z[:, :2].contiguous(),
            g_con.contiguous() if g_con is not None else z,
            g_col.contiguous() if g_col is not None else z,
            g_means, g_ls, g_q, g_sh, n, threads=n,
        )
        return g_means, g_ls, g_q, g_sh, None


def project_and_color(means, log_scales, quats, sh, cam, degree):
    """Returns means2d, conics, depths, radii, colours, like project() plus
    colors_from_sh()."""
    return _Project.apply(means, log_scales, quats, sh, cam_buffer(cam, degree))
