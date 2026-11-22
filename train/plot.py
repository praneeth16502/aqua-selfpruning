"""Plot styling shared by every figure in the repo, plus the per-run curves written by train_dense / train_prune.

One look everywhere, and one colour per method in every figure:
    fisher = blue, magnitude = orange, saliency = aqua, dense = neutral grey.
Palette: the first three slots of the categorical palette in the dataviz reference (validated for all pairs, light
and dark). Aqua is under 3:1 contrast on white, so every figure carries a legend or direct labels.
"""
from __future__ import annotations

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"      # secondary text
MUTED = "#898781"      # axis labels, reference lines
BASELINE = "#c3c2b7"   # axes, grid
RAMP_SHADE = "#efeeea"  # the pruning window

COLORS = {"fisher": "#2a78d6", "magnitude": "#eb6834", "saliency": "#1baf7a", "dense": "#898781",
          "fisher-regrow": "#6250d6"}
LABELS = {"fisher": "Fisher (method)", "magnitude": "magnitude (baseline)", "saliency": "saliency (1st order)",
          "dense": "dense", "fisher-regrow": "Fisher + regrowth"}


def setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASELINE, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": BASELINE, "grid.alpha": 0.45, "grid.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "legend.frameon": False, "legend.fontsize": 8.5, "legend.labelcolor": INK_2,
        "lines.linewidth": 2.0, "lines.markersize": 6, "font.size": 9.5,
        "figure.titlesize": 12, "figure.titleweight": "bold",
    })
    return plt


def plot_learning_curve(hist: dict, path: str, title: str = "") -> None:
    plt = setup()
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    e = hist["epoch"]
    a.plot(e, hist["train_loss"], color=MUTED, label="train (running mean over epoch)")
    a.plot(e, hist["val_loss"], color=COLORS["fisher"], label="validation")
    a.set(xlabel="epoch", ylabel="cross-entropy", title="Loss")
    b.plot(e, [100 * x for x in hist["train_acc"]], color=MUTED, label="train (running mean over epoch)")
    b.plot(e, [100 * x for x in hist["val_acc"]], color=COLORS["fisher"], label="validation")
    b.set(xlabel="epoch", ylabel="accuracy (%)", title="Accuracy")
    a.legend()
    b.legend(loc="lower right")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_prune_curve(hist: dict, path: str, title: str = "", warmup: float | None = None,
                     ramp_end: float | None = None) -> None:
    """Loss, accuracy and achieved sparsity per epoch, with the pruning ramp shaded."""
    plt = setup()
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    e = hist["epoch"]
    a, b, c = axes
    a.plot(e, hist["train_loss"], color=MUTED, label="train (running mean)")
    a.plot(e, hist["val_loss"], color=COLORS["fisher"], label="validation")
    a.set(xlabel="epoch", ylabel="cross-entropy", title="Loss")
    b.plot(e, [100 * x for x in hist["train_acc"]], color=MUTED, label="train (running mean)")
    b.plot(e, [100 * x for x in hist["val_acc"]], color=COLORS["fisher"], label="validation")
    b.set(xlabel="epoch", ylabel="accuracy (%)", title="Accuracy")
    c.plot(e, [100 * x for x in hist.get("sparsity", [0] * len(e))], marker="o", color=INK_2)
    c.set(xlabel="epoch", ylabel="sparsity (%)", title="Achieved sparsity (end of epoch)", ylim=(0, 100))
    for ax in axes:
        if warmup is not None and ramp_end is not None:
            ax.axvspan(warmup, ramp_end, color=RAMP_SHADE, zorder=0, lw=0)
    a.legend()
    b.legend(loc="lower right")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
