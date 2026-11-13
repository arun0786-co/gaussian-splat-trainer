# How it works

## 1. The representation

3D Gaussian splatting (Kerbl et al., SIGGRAPH 2023) represents a scene as a few hundred thousand to a few million 3D Gaussians. Each one has:

| parameter | stored as | used as |
|---|---|---|
| centre | `means` [3] | world position |
| size | `log_scales` [3] | `s = exp(log_scale)`, the standard deviation along its three axes |
| orientation | `quats` [4] | normalised quaternion, rotation `Rq` |
| covariance | (derived) | `Σ = Rq diag(s)² Rq^T` |
| opacity | `opacity` [1] | `sigmoid(opacity)` |
| colour | `sh_dc` [1, 3] + `sh_rest` [15, 3] | spherical harmonics up to degree 3, so colour can change with viewing direction |

Storing log-scales, an unnormalised quaternion and an opacity logit keeps every parameter unconstrained, so plain Adam works on all of them.

Training needs photos with known camera poses, and a sparse point cloud to start from. COLMAP's structure-from-motion gives both, and the datasets ship with it (`colmap.py` reads the binary files). Every SfM point becomes one Gaussian:
- its colour is the point's colour;
- its size is the mean distance to its 3 nearest neighbours;
- its opacity starts at 0.1.

## 2. Rendering, step 1: projection (`project.py`, `project_metal.py`)

**Centre.** Each Gaussian's centre goes through the camera: `p = R m + t`, then `u = fx x/z + cx`, `v = fy y/z + cy`.

**Shape.** A 3D Gaussian seen through a perspective camera is not exactly a 2D Gaussian, so EWA splatting (Zwicker et al., 2002) linearises the projection at the centre. With J the 2x3 Jacobian of `(x, y, z) -> (fx x/z, fy y/z)`:

```
Σ2D = J R Σ R^T J^T + 0.3 I
```

The `+ 0.3` (in pixels²) is a low-pass filter. It keeps every Gaussian at least about one pixel wide, so tiny Gaussians cannot alias.

**What the rasterizer gets.** It needs the inverse, `conic = Σ2D⁻¹`, stored as its three entries (a, b, c). It also needs a radius of 3 standard deviations along the major axis, which decides the tiles the Gaussian can touch.

**Colour.** The colour is the spherical-harmonics expansion evaluated in the direction from the camera to the Gaussian, plus 0.5, clamped at 0.

**Two implementations:**
- `project.py` is plain PyTorch.
- `project_metal.py` runs one GPU thread per Gaussian, with a hand-written backward pass.

The PyTorch version is the reference. Autograd's gradients for it are what the kernel's gradients are tested against.

## 3. Rendering, step 2: tile rasterization (`raster.py`)

A pixel's colour is front-to-back alpha compositing over the Gaussians that overlap it, nearest first:

```
alpha_i = min(0.99, opacity_i · exp(-½ dᵀ conic_i d)),   d = centre_i - pixel
C = Σ_i c_i alpha_i T_i + T_final · background,        T_i = Π_{j<i} (1 - alpha_j)
```

Testing every Gaussian against every pixel would be `pixels × Gaussians` work, so the image is cut into 16x16-pixel tiles:

1. **Binning.** Every Gaussian is listed once for each tile its 3-sigma box overlaps. The list is sorted by the key `tile · N + depth rank`, which groups it by tile and orders each tile front to back. One sort orders everything; for the Truck scene that is about 1.3 million (tile, Gaussian) pairs at 226k Gaussians. A `bincount` then gives each tile's slice of the list.
2. **Forward kernel.** One threadgroup per tile, one thread per pixel.
   - The tile's Gaussians are loaded in batches of 256 into threadgroup memory: each thread loads one, then all threads read all of them.
   - Each pixel composites front to back, skipping alpha < 1/255.
   - A pixel stops once its transmittance would drop below 1e-4. The whole tile stops when all 256 pixels have.
   - Each pixel saves its final T and the index of its last contributing Gaussian, for the backward pass.

## 4. Rasterization backward

The backward kernel walks each pixel's list **back to front**. Going backwards, the transmittance in front of Gaussian i can be recovered from the one behind it: `T_i = T_{i+1} / (1 - alpha_i)`. The colour composited behind Gaussian i is accumulated as it goes:

```
behind_i = alpha_{i+1} c_{i+1} + (1 - alpha_{i+1}) behind_{i+1}
```

Then:

```
dC/dc_i     = alpha_i T_i
dC/dalpha_i = T_i (c_i - behind_i) - T_final · background / (1 - alpha_i)
```

From `dL/dalpha` the chain continues to the opacity, and through `G = exp(power)` to the conic entries and the 2D centre.

**Contention.** Every pixel of a tile adds into the same Gaussian's gradient, so done naively that is 256 threads doing atomic adds to one address. Here the 32 pixels of a simdgroup always look at the same Gaussian at the same moment, because they step through the shared batch together. The kernel therefore sums their 9 gradient values (centre xy, conic abc, opacity, colour rgb) with `simd_sum`. Lanes 0 to 8 then each do one atomic add: 9 atomics per simdgroup instead of 9 per pixel.

**The alpha clamp.** Where `opacity · G > 0.99`, alpha is clamped, so the exact gradient is zero there. The reference CUDA code passes the gradient straight through instead, as if there were no clamp. Both are implemented (`--exact-clamp` picks the exact one), and the PyTorch reference matches the kernel in both modes. On Truck they gave the same test PSNR to within 0.1 dB at 1,000 and 2,000 steps, so the default follows the reference.

**Two other differences from the reference code:**
- **The off-diagonal conic entry.** It gets its full derivative. The CUDA code stores half of it and compensates later.
- **Pixel centres** are at integer coordinates, and the projected centre includes the -0.5 that converts from COLMAP's convention.

## 5. Projection backward

Per Gaussian, given `dL/dcentre2D`, `dL/dconic` and `dL/dcolour`, everything is recomputed and differentiated by hand. With `T = J R` (2x3) and `G` the gradient with respect to the 2x2 covariance (its off-diagonal entry split in half between the two positions):

```
conic -> cov2D:  (A, B, C) = (c, -b, a) / (ac - b²), differentiated entry by entry
cov2D -> Σ:      dL/dΣ = Tᵀ G T
cov2D -> T:      dL/dT = 2 G T Σ, then dL/dJ = dL/dT Rᵀ
J -> p:          J00 = fx/z, J02 = -fx x/z², and so on. Where x/z was clamped to
                 1.3x the field of view, the clamped value carries the dependence
                 on z instead of x
Σ -> M:          Σ = M Mᵀ, so dL/dM = 2 dL/dΣ M
M -> scale:      M = Rq diag(s), so dL/ds_j = Σ_i dL/dM_ij Rq_ij
M -> rotation:   dL/dRq_ij = dL/dM_ij s_j, then through the quaternion
                 formula and its normalisation
colour -> SH:    dL/dsh_k = Y_k(dir) dL/dcolour
colour -> dir:   dL/ddir = Σ_k ∇Y_k (sh_k · dL/dcolour), then dir = v/|v|
```

The test compares all four gradient tensors with autograd on the PyTorch reference, for SH degrees 0 to 3, on 3,000 Gaussians that include culled and clamped ones. The median relative error is about 1e-7. Flipping one sign in the quaternion gradient makes the test fail.

## 6. Training (`train.py`, `gaussians.py`)

Each step picks a training photo (a shuffled pass over all of them), renders it, and takes `0.8 · L1 + 0.2 · (1 - SSIM)` as the loss.

**Optimizer.** Adam, with the learning rates of the paper:
- centres: `1.6e-4 × scene radius`, decaying log-linearly to 1/100 of that;
- colour: `2.5e-3` for the base colour, 20x less for the higher harmonics;
- opacity `0.05`, scales `5e-3`, rotations `1e-3`.

The spherical-harmonics degree goes up by one every 1,000 steps.

**Adaptive density control**, every 100 steps from step 500 to 15,000:
- **Clone.** A Gaussian whose projected centre keeps getting large gradients (average above 0.0002 in normalised device coordinates) sits in a region the current set fits badly. If it is small (largest scale under 1% of the scene radius), it is cloned.
- **Split.** If it is large, it is split into two Gaussians sampled from itself, each 1.6x smaller.
- **Prune.** Gaussians with opacity under 0.005 are removed. After step 3,000, so are ones larger than 10% of the scene radius.
- **Opacity reset.** Every 3,000 steps all opacities are pulled down to 0.01, so floaters must earn their opacity back.

**A rule the paper has but its code does not.** The paper also prunes Gaussians larger than 20 px on screen, and the first version of this trainer did. In the reference code that test never fires: `densification_postfix()` sets `max_radii2D` to zero, and `densify_and_prune()` reads it right after (`scene/gaussian_model.py`, lines 407 and 462 in the version checked in November 2026). Applying the rule for real removes, every 100 steps, the large Gaussians that cover sky, road and shadows. On Truck, test PSNR at 5,000 steps was 19.30 dB with the rule and 22.93 dB without it. The README has the full comparison, and `--screen-prune` turns the rule back on.

`torch.optim.Adam` cannot change the number of rows in its state, so `gaussians.Adam` is a small Adam over a dict of tensors. Densification rebuilds parameters and moments together, and the test checks that rows stay aligned.

## 7. How it is tested

Every hand-written GPU kernel has a PyTorch twin, and the tests compare values and gradients:

- **Rasterizer** against a dense version that builds the full `pixels × Gaussians` alpha matrix with exactly the same rules: tile culling, the 1/255 skip, the 1e-4 stop and the 0.99 clamp. Images agree to 1e-6, and all four gradients to about 1e-6 relative.
- **Binning** against a brute-force tile-by-tile list.
- **Projection and colour** against `project.py` plus autograd.
- **End to end:** a random scene is rendered from 12 views, and training from a perturbed copy must gain at least 10 dB PSNR. On a machine without an Apple GPU (CI), this test uses the PyTorch reference path.

## References

- B. Kerbl, G. Kopanas, T. Leimkühler, G. Drettakis. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. SIGGRAPH 2023. https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- M. Zwicker, H. Pfister, J. van Baar, M. Gross. EWA Splatting. IEEE TVCG 2002.
- Reference rasterizer: https://github.com/graphdeco-inria/diff-gaussian-rasterization
- A. Knapitsch et al. Tanks and Temples: Benchmarking Large-Scale Scene Reconstruction. SIGGRAPH 2017.
