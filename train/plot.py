from __future__ import annotations


def plot_learning_curve(hist: dict, path: str, title: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    e = hist["epoch"]
    a.plot(e, hist["train_loss"], label="train (running)")
    a.plot(e, hist["val_loss"], label="val")
    a.set(xlabel="epoch", ylabel="cross-entropy", title="loss")
    a.legend()
    b.plot(e, [100 * x for x in hist["train_acc"]], label="train (running)")
    b.plot(e, [100 * x for x in hist["val_acc"]], label="val")
    b.set(xlabel="epoch", ylabel="accuracy (%)", title="accuracy")
    b.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_prune_curve(hist: dict, path: str, title: str = "", warmup: float | None = None,
                     ramp_end: float | None = None) -> None:
    """Loss, accuracy and achieved sparsity per epoch, with the prune window shaded."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    e = hist["epoch"]
    a, b, c = axes
    a.plot(e, hist["train_loss"], label="train (running)")
    a.plot(e, hist["val_loss"], label="val")
    a.set(xlabel="epoch", ylabel="cross-entropy", title="loss")
    b.plot(e, [100 * x for x in hist["train_acc"]], label="train (running)")
    b.plot(e, [100 * x for x in hist["val_acc"]], label="val")
    b.set(xlabel="epoch", ylabel="accuracy (%)", title="accuracy")
    c.plot(e, [100 * x for x in hist.get("sparsity", [0] * len(e))], marker="o", color="tab:red")
    c.set(xlabel="epoch", ylabel="sparsity (%)", title="achieved sparsity (end of epoch)", ylim=(0, 100))
    for ax in axes:
        if warmup is not None and ramp_end is not None:
            ax.axvspan(warmup, ramp_end, color="0.9", zorder=0)
    a.legend()
    b.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
