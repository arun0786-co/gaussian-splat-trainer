"""Build the README figures from saved outputs. Nothing is trained or rendered here.

Inputs:
  output/<run>/renders_<step>/*.jpg   ground truth (left) and render (right), written by splat.train
  results/*_train_log*.json           training logs
  results/truck_30k.json              per-view test metrics
  results/profile*.json               step profile

    uv run --no-project --with matplotlib --with pillow python scripts/make_figures.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "images")
RENDERS = os.path.join(ROOT, "output")
RES = os.path.join(ROOT, "results")

BLUE, ORANGE, AQUA, GRAY = "#257de0", "#f47230", "#00b894", "#a3a3a3"
TEXT, DIM = "#191919", "#666666"

plt.rcParams.update({
    "font.family": "Helvetica", "font.size": 10,
    "axes.edgecolor": "#cccccc", "axes.labelcolor": TEXT,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": DIM, "ytick.color": DIM,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "savefig.dpi": 150,
})

FONT = "/System/Library/Fonts/Helvetica.ttc"


def font(size):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default()


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def halves(path):
    """Split a saved pair into (ground truth, render)."""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    return im.crop((0, 0, w // 2, h)), im.crop((w // 2, 0, w, h))


def save_photo(im, name):
    """Photo grids go out as JPEG at quality 90: a full-colour PNG of these is over 1 MB,
    and a 256-colour palette would add dithering noise to a picture about render quality."""
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    im.save(path, quality=90, optimize=True, progressive=True)
    print(path, os.path.getsize(path) // 1024, "KB")


def save_fig(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(path, os.path.getsize(path) // 1024, "KB")


def grid(tiles, cols, labels, tile_w, pad=6, label_h=26):
    """Lay out tiles in a grid with a label above each one, on white."""
    th = round(tiles[0].height * tile_w / tiles[0].width)
    rows = (len(tiles) + cols - 1) // cols
    W = cols * tile_w + (cols + 1) * pad
    H = rows * (th + label_h) + (rows + 1) * pad
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    f = font(16)
    for k, (t, lab) in enumerate(zip(tiles, labels)):
        r, c = divmod(k, cols)
        x = pad + c * (tile_w + pad)
        y = pad + r * (th + label_h + pad)
        d.text((x, y + 3), lab, fill=TEXT, font=f)
        canvas.paste(t.resize((tile_w, th), Image.LANCZOS), (x, y + label_h))
    return canvas


def truck_views():
    """Held-out views at 30k steps: ground truth next to render, with each view's PSNR."""
    per_view = {v["view"]: v for v in load("truck_30k.json")["per_view"]}
    tiles, labels = [], []
    for fn in ["000_000001.jpg", "002_000017.jpg", "004_000033.jpg"]:
        gt, rd = halves(os.path.join(RENDERS, "truck", "renders_30000", fn))
        view = fn.split("_", 1)[1]
        m = per_view[view]
        tiles += [gt, rd]
        labels += [f"ground truth, test view {view}",
                   f"render, 30k steps: {m['psnr']:.2f} dB, SSIM {m['ssim']:.3f}"]
    save_photo(grid(tiles, 2, labels, 560), "truck_test_views.jpg")


def truck_progress():
    """One held-out view after 7k, 15k and 30k steps, cropped to the truck's cab."""
    evals = load("truck_train_log.json")["evals"]
    box = (0, 60, 520, 450)   # 979x546 frame: the cab and the logo on the door
    fn = "002_000017.jpg"
    gt, _ = halves(os.path.join(RENDERS, "truck", "renders_7000", fn))
    tiles, labels = [gt.crop(box)], ["ground truth"]
    for step in ["7000", "15000", "30000"]:
        _, rd = halves(os.path.join(RENDERS, "truck", f"renders_{step}", fn))
        e = evals[step]
        tiles.append(rd.crop(box))
        labels.append(f"{int(step) // 1000}k steps, {e['gaussians'] / 1e6:.2f}M Gaussians, test {e['psnr']:.2f} dB")
    save_photo(grid(tiles, 2, labels, 560), "truck_progress.jpg")


def screen_prune():
    """The same view at 7k steps with the paper's screen-space pruning off (default) and on."""
    ab = load("ablations.json")["screen_space_prune"]
    fn = "002_000017.jpg"
    gt, off = halves(os.path.join(RENDERS, "p_default", "renders_7000", fn))
    _, on = halves(os.path.join(RENDERS, "p_screen", "renders_7000", fn))
    labels = ["ground truth",
              f"pruning off (default): test {ab['off (default)']['7000']['psnr']:.2f} dB",
              f"pruning on (--screen-prune): test {ab['on (--screen-prune)']['7000']['psnr']:.2f} dB"]
    save_photo(grid([gt, off, on], 3, labels, 480), "screen_prune.jpg")


def curves():
    runs = [("truck_train_log.json", "Truck, M4 Pro (Metal)", BLUE),
            ("truck_train_log_cuda.json", "Truck, T4 (CUDA)", ORANGE),
            ("train_train_log_cuda.json", "Train, T4 (CUDA)", AQUA)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for fname, label, c in runs:
        log = load(fname)
        st = [s["step"] for s in log["steps"]]
        axes[0].plot(st, [s["loss"] for s in log["steps"]], color=c, lw=1.6, label=label)
        axes[1].plot(st, [s["gaussians"] / 1e6 for s in log["steps"]], color=c, lw=2, label=label)
        ev = sorted(log["evals"].items(), key=lambda kv: int(kv[0]))
        axes[2].plot([int(k) for k, _ in ev], [v["psnr"] for _, v in ev], color=c, lw=2,
                     marker="o", ms=6, label=label)
    axes[0].set_title("training loss (L1 + SSIM)")
    axes[1].set_title("Gaussians (millions)")
    axes[2].set_title("held-out PSNR (dB)")
    for ax in axes:
        ax.set_xlabel("step")
        ax.grid(color="#eeeeee")
        ax.set_axisbelow(True)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:g}k"))
    axes[1].axvline(15000, color=GRAY, lw=1, ls="--")
    axes[1].text(15300, 0.15, "densification ends", color=DIM, fontsize=9)
    axes[0].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    save_fig(fig, "training_curves.png")


def profile():
    mac = load("profile.json")
    t4 = load("profile_cuda_truck.json")
    stages = [("projection + SH,\nPyTorch autograd", "projection + SH, PyTorch, fwd+bwd ms"),
              ("projection + SH,\nkernel", None),
              ("tile binning", "binning (sort) ms"),
              ("rasterize\nforward + backward", "rasterize forward+backward ms"),
              ("loss\nforward + backward", "loss (L1 + SSIM) fwd+bwd ms")]

    def get(d, key):
        if key is None:
            key = next(k for k in d if k.startswith("projection + SH") and "PyTorch" not in k)
        return d[key]

    fig, ax = plt.subplots(figsize=(8, 3.4))
    w = 0.38
    for k, (d, c, lab) in enumerate([(mac, BLUE, "M4 Pro, Metal"), (t4, ORANGE, "T4, CUDA")]):
        vals = [get(d, key) for _, key in stages]
        xs = [i + (k - 0.5) * w for i in range(len(stages))]
        ax.bar(xs, vals, width=w - 0.03, color=c, label=lab)
        for x, v in zip(xs, vals):
            ax.text(x, v * 1.12, f"{v:g}", ha="center", fontsize=8.5, color=TEXT)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_ylim(5, 3000)
    ax.set_xticks(range(len(stages)), [s for s, _ in stages])
    ax.set_ylabel("ms per step (log)")
    ax.set_title("one training step on the final Truck model (2.57M Gaussians)")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", color="#eeeeee")
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_fig(fig, "step_profile.png")


if __name__ == "__main__":
    import matplotlib.ticker  # noqa: F401
    truck_views()
    truck_progress()
    screen_prune()
    curves()
    profile()
