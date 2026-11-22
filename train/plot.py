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
