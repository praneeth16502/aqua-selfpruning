"""Regenerate every figure in the repo from the committed result files (no training).

    python -m train.figures

Writes:
  results/part2_mnist_seed0_curve.png            Part 2 learning curve
  results/part3/{dense,saliency_s90,fisher_s90,magnitude_s90}_seed0_curve.png   per-run pruning curves
  results/part3/pruning_dynamics.png             sparsity schedule, val accuracy, train loss (mean of 5 seeds)
  results/part3/criteria_90.png                  every seed at 90%, per criterion
  results/part4/pareto.png                       sparsity vs accuracy
  results/part4/cost.png                         measured speedup of compacted and CSR paths
  results/summary.png                            headline: accuracy vs measured speedup
  results/part4/input_pixels.png                 which input pixels each pruned model still reads
Everything is drawn from results/*.json and results/part4/models/*.npz, so a figure can never disagree with the
numbers next to it.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from train.plot import (BASELINE, COLORS, INK, INK_2, LABELS, MUTED, RAMP_SHADE, SURFACE, plot_learning_curve,
                        plot_prune_curve, setup)

R = Path("results")
P3, P4 = R / "part3", R / "part4"
SEEDS5 = [0, 1, 2, 3, 4]


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _mean_std(xs):
    m = sum(xs) / len(xs)
    s = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
    return m, s


def _save(fig, path: Path, plt):
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


# --------------------------------------------------------------------------- per-run curves
def per_run_curves():
    r = _load(R / "part2_mnist_seed0.json")
    plot_learning_curve(r["history"], str(R / "part2_mnist_seed0_curve.png"),
                        f"Part 2: dense MLP {r['dims']}, seed 0, test {r['test_acc']:.2%}")
    print(f"wrote {R / 'part2_mnist_seed0_curve.png'}")
    for name in ("dense", "saliency_s90", "fisher_s90", "magnitude_s90"):
        r = _load(P3 / f"{name}_seed0.json")
        c = r["config"]
        plot_prune_curve(r["history"], str(P3 / f"{name}_seed0_curve.png"),
                         f"{name} {r['dims']} seed 0 | test {r['test_acc']:.2%}", c["warmup_epochs"], c["ramp_end_epoch"])
        print(f"wrote {P3 / f'{name}_seed0_curve.png'}")


# --------------------------------------------------------------------------- pruning dynamics
def pruning_dynamics():
    plt = setup()
    runs = {"dense": "dense", "saliency": "saliency_s90", "fisher": "fisher_s90", "magnitude": "magnitude_s90"}
    hist = {k: [_load(P3 / f"{v}_seed{s}.json")["history"] for s in SEEDS5] for k, v in runs.items()}
    cfg = _load(P3 / "fisher_s90_seed0.json")["config"]
    e = hist["dense"][0]["epoch"]
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(14, 4))
    for ax in (a, b, c):
        ax.axvspan(cfg["warmup_epochs"], cfg["ramp_end_epoch"], color=RAMP_SHADE, zorder=0, lw=0)
    a.plot(e, [100 * x for x in hist["fisher"][0]["sparsity"]], color=INK_2, marker="o", ms=4)
    a.text(cfg["warmup_epochs"] + 0.2, 95, "cubic ramp", color=INK_2, fontsize=8.5, va="top")
    a.set(xlabel="epoch", ylabel="sparsity of weight matrices (%)", title="Schedule (same for every method)", ylim=(0, 100))
    for k in ("dense", "saliency", "fisher", "magnitude"):
        va = np.mean([[100 * x for x in h["val_acc"]] for h in hist[k]], axis=0)
        tl = np.mean([h["train_loss"] for h in hist[k]], axis=0)
        ls = "--" if k == "dense" else "-"
        b.plot(e, va, color=COLORS[k], ls=ls, label=LABELS[k])
        c.plot(e, tl, color=COLORS[k], ls=ls, label=LABELS[k])
    b.set(xlabel="epoch", ylabel="validation accuracy (%)", title="Validation accuracy (mean of 5 seeds)", ylim=(95, 98.5))
    c.set(xlabel="epoch", ylabel="training loss (log scale)", title="Training loss (mean of 5 seeds)", yscale="log")
    c.legend(loc="upper right")
    fig.suptitle("Pruning to 90% during training: magnitude keeps the weights the training loss needs best")
    fig.tight_layout()
    _save(fig, P3 / "pruning_dynamics.png", plt)


# --------------------------------------------------------------------------- 90%: every seed
def criteria_90():
    plt = setup()
    order = [("dense", "dense"), ("saliency", "saliency_s90"), ("fisher", "fisher_s90"), ("magnitude", "magnitude_s90")]
    fig, ax = plt.subplots(figsize=(7.2, 4))
    for i, (k, name) in enumerate(order):
        accs = [100 * _load(P3 / f"{name}_seed{s}.json")["test_acc"] for s in SEEDS5]
        m, sd = _mean_std(accs)
        jitter = np.linspace(-0.12, 0.12, len(accs))
        ax.scatter(i + jitter, accs, s=46, color=COLORS[k], edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.errorbar(i + 0.32, m, yerr=sd, fmt="o", color=INK, ms=6, capsize=4, lw=1.4, zorder=3)
        ax.text(i + 0.42, m, f"{m:.2f} ± {sd:.2f}", color=INK_2, fontsize=8.5, va="center")
    ax.set_xticks(range(len(order)), [LABELS[k] for k, _ in order])
    ax.set_xlim(-0.5, len(order) - 0.1)
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("90% sparsity, 5 seeds each (dots = seeds, black = mean ± std)")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, P3 / "criteria_90.png", plt)


# --------------------------------------------------------------------------- Pareto
def plot_pareto(table: dict, seeds, path: Path):
    plt = setup()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    dense = table[next(iter(table))][0]
    ax.axhline(100 * dense["mean"], color=MUTED, ls="--", lw=1.2, zorder=1)
    ax.text(20, 100 * dense["mean"] - 0.03, f"dense, same schedule ({100 * dense['mean']:.2f}%)", color=INK_2,
            fontsize=8.5, va="top")
    for meth, pts in table.items():
        x = [100 * p["achieved_sparsity"] for p in pts]
        y = [100 * p["mean"] for p in pts]
        e = [100 * p["std"] for p in pts]
        ax.errorbar(x, y, yerr=e, color=COLORS[meth], marker="o", ms=7, capsize=3, lw=2, label=LABELS[meth],
                    markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
    ax.set(xlabel="achieved sparsity of weight matrices (%)", ylabel="final-epoch test accuracy (%)",
           title=f"Sparsity vs accuracy (mean ± std over seeds {', '.join(map(str, seeds))})")
    ax.set_xticks([0, 50, 75, 90, 95])
    ax.legend(loc="lower left")
    fig.tight_layout()
    _save(fig, path, plt)


def pareto():
    d = _load(P4 / "pareto.json")
    plot_pareto(d["curves"], d["seeds"], P4 / "pareto.png")


# --------------------------------------------------------------------------- cost
def _cost_rows():
    rows = {}
    for m in _load(P4 / "cost.json")["models"]:
        name = m["model"].replace("_seed0.npz", "")
        rows[name] = m
    return rows


def _models_in_order(rows):
    out = []
    for meth in ("fisher", "magnitude"):
        for t in (50, 75, 90, 95):
            if f"{meth}_s{t}" in rows:
                out.append((meth, t, rows[f"{meth}_s{t}"]))
    return out


def cost():
    plt = setup()
    rows = _cost_rows()
    items = _models_in_order(rows)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2))
    labels = [f"{t}%" for _, t, _ in items]
    xs = np.arange(len(items)) + np.array([0 if m == "fisher" else 0.6 for m, _, _ in items])
    for ax, key, title in ((a, "compact", "Compacted dense path: smaller matrices, same BLAS"),
                           (b, "csr", "CSR path (my own kernel): only live weights touched")):
        sp = [r["seconds_per_batch"]["1000"]["dense"] / r["seconds_per_batch"]["1000"][key] for _, _, r in items]
        ax.bar(xs, sp, width=0.7, color=[COLORS[m] for m, _, _ in items], edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.axhline(1.0, color=INK_2, lw=1.2, zorder=4)
        gap = (xs[3] + xs[4]) / 2 if len(xs) > 4 else xs[-1]   # empty space between the two method groups
        ax.text(gap, 1.0, "dense\n= 1×", color=INK_2, fontsize=8, va="bottom", ha="center", linespacing=1.0)
        for x, v in zip(xs, sp):
            label = f"{v:.2f}×" if v >= 0.1 else f"{v:.3f}×"
            ax.text(x, v * (1.03 if key == "compact" else 1.15), label, ha="center", va="bottom",
                    color=INK_2, fontsize=8.5)
        ax.set_xticks(xs, labels)
        ax.set_title(title)
        ax.set_xlabel("sparsity")
        ax.grid(axis="x", visible=False)
    a.set_ylabel("wall-clock speedup vs dense (batch 1000)")
    a.set_ylim(0, max(r["seconds_per_batch"]["1000"]["dense"] / r["seconds_per_batch"]["1000"]["compact"]
                      for _, _, r in items) * 1.18)
    b.set_yscale("log")
    b.set_ylim(0.002, 3)
    from matplotlib.ticker import FixedLocator, FuncFormatter
    b.yaxis.set_major_locator(FixedLocator([0.003, 0.01, 0.03, 0.1, 0.3, 1]))
    b.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
    b.yaxis.set_minor_locator(FixedLocator([]))
    b.set_ylabel("wall-clock speedup vs dense (log scale)")
    from matplotlib.patches import Patch
    a.legend(handles=[Patch(color=COLORS["fisher"], label=LABELS["fisher"]),
                      Patch(color=COLORS["magnitude"], label=LABELS["magnitude"])], loc="upper left")
    fig.suptitle("Measured inference cost (single thread, identical predictions on every path)")
    fig.tight_layout()
    _save(fig, P4 / "cost.png", plt)


# --------------------------------------------------------------------------- headline
def summary():
    """Accuracy (mean ± std over seeds 0-2) vs measured speedup of the compacted model (seed 0)."""
    plt = setup()
    rows = _cost_rows()
    par = _load(P4 / "pareto.json")["curves"]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    dense = par["fisher"][0]
    ax.errorbar([1.0], [100 * dense["mean"]], yerr=[100 * dense["std"]], fmt="s", color=COLORS["dense"], ms=8,
                capsize=3, zorder=3, label="dense (784-256-128-10)")
    ax.annotate("dense", (1.0, 100 * dense["mean"]), xytext=(9, -12), textcoords="offset points", color=INK_2, fontsize=8)
    for meth in ("fisher", "magnitude"):
        xs, ys, es, tags = [1.0], [100 * dense["mean"]], [0], [None]
        for p in par[meth][1:]:
            r = rows.get(f"{meth}_s{p['target_pct']}")
            if r is None:
                continue
            t = r["seconds_per_batch"]["1000"]
            xs.append(t["dense"] / t["compact"])
            ys.append(100 * p["mean"])
            es.append(100 * p["std"])
            tags.append((p["target_pct"], r["compact_shapes"]))
        if meth == "fisher":  # magnitude's points all sit at ~1.0x; a connecting line there only zig-zags
            ax.plot(xs, ys, color=COLORS[meth], lw=1.5, alpha=0.6, zorder=2)
        ax.errorbar(xs[1:], ys[1:], yerr=es[1:], fmt="o", color=COLORS[meth], ms=8, capsize=3, zorder=3,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=LABELS[meth])
        # label positions picked per point so nothing collides (magnitude points all sit near 1.0x)
        offsets = {("fisher", 50): (9, -26, "left"), ("fisher", 75): (0, 16, "center"),
                   ("fisher", 90): (10, 6, "left"), ("fisher", 95): (10, 6, "left"),
                   ("magnitude", 50): (-10, 0, "right"), ("magnitude", 75): (10, 4, "left"),
                   ("magnitude", 90): (-10, 0, "right"), ("magnitude", 95): (-10, 0, "right")}
        for x, y, tag in zip(xs[1:], ys[1:], tags[1:]):
            t, shapes = tag
            dx, dy, ha = offsets.get((meth, t), (9, 4, "left"))
            if meth == "fisher":
                dims = "-".join([str(shapes[0][0])] + [str(s[1]) for s in shapes])
                text = f"{t}%: {dims}"
            else:
                text = f"{t}%"
            ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", color=INK_2, fontsize=8,
                        ha=ha, va="center")
    ax.axvline(1.0, color=BASELINE, lw=1, zorder=1)
    ax.set(xlabel="measured wall-clock speedup of the compacted model vs dense (batch 1000, single thread)",
           ylabel="test accuracy (%)")
    ax.set_xlim(0.75, max(2.7, ax.get_xlim()[1]))
    ax.set_title("Accuracy held at every sparsity; only Fisher's model actually got faster\n"
                 "(it removes whole input pixels and hidden units, so the dense matrices really shrink)", loc="left")
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, R / "summary.png", plt)


# --------------------------------------------------------------------------- input pixels
def input_pixels():
    from matplotlib.colors import LinearSegmentedColormap
    from train.data import get_dataset

    plt = setup()
    data = get_dataset("mnist")
    used = (data.x_train > 0).mean(axis=0).reshape(28, 28) * 100
    ramp = LinearSegmentedColormap.from_list("blue", [SURFACE, "#9cc3f0", COLORS["fisher"], "#123d73"])
    grey = LinearSegmentedColormap.from_list("grey", [SURFACE, "#c3c2b7", "#52514e", INK])
    maps = {}
    for meth in ("fisher", "magnitude"):
        z = np.load(P4 / "models" / f"{meth}_s90_seed0.npz")
        maps[meth] = z["mask0"].sum(axis=1).reshape(28, 28)
    vmax = max(m.max() for m in maps.values())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    im0 = axes[0].imshow(used, cmap=grey, vmin=0, vmax=100)
    axes[0].set_title("Training images:\n% that have ink in each pixel")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04).ax.tick_params(labelsize=8)
    for ax, meth in zip(axes[1:], ("fisher", "magnitude")):
        m = maps[meth]
        im = ax.imshow(m, cmap=ramp, vmin=0, vmax=vmax)
        read = int((m > 0).sum())
        ax.set_title(f"{LABELS[meth]}, 90% sparsity:\nreads {read} of 784 pixels")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=8)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("How many first-layer weights each input pixel kept (white = pixel removed entirely)")
    fig.tight_layout()
    _save(fig, P4 / "input_pixels.png", plt)


def main():
    per_run_curves()
    pruning_dynamics()
    criteria_90()
    pareto()
    cost()
    summary()
    input_pixels()


if __name__ == "__main__":
    main()
