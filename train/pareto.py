"""Part 4: sparsity-accuracy Pareto curve.

    python -m train.pareto --dir results/part3 --seeds 0 1 2

Uses the SAME seeds at every sparsity (so every point has the same n); target 0 is the shared dense run.
x = achieved sparsity of the weight matrices (read from each run, not the requested target), y = final-epoch test
accuracy, mean +- sample std over seeds.  Writes <out>/pareto.json (raw per-seed numbers) and <out>/pareto.png.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/part3")
    ap.add_argument("--methods", nargs="+", default=["fisher", "magnitude"])
    ap.add_argument("--targets", type=int, nargs="+", default=[0, 50, 75, 90, 95])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="results/part4")
    a = ap.parse_args(argv)
    d = Path(a.dir)

    table = {}
    for meth in a.methods:
        pts = []
        for t in a.targets:
            recs = []
            for s in a.seeds:
                f = d / (f"dense_seed{s}.json" if t == 0 else f"{meth}_s{t}_seed{s}.json")
                if not f.exists():
                    raise FileNotFoundError(f"missing {f} (run train.sweep first)")
                recs.append(json.loads(f.read_text()))
            accs = [r["test_acc"] for r in recs]
            sp = [r.get("sparsity", 0.0) for r in recs]
            m = sum(accs) / len(accs)
            sd = math.sqrt(sum((x - m) ** 2 for x in accs) / (len(accs) - 1)) if len(accs) > 1 else 0.0
            pts.append({"target_pct": t, "achieved_sparsity": sum(sp) / len(sp), "seeds": a.seeds, "test_acc": accs,
                        "mean": m, "std": sd, "active_weights": recs[0].get("active_weights", recs[0].get("prunable_weights")),
                        "layer_sparsity": [r.get("layer_sparsity") for r in recs]})
        table[meth] = pts

    print(f"{'target':>7} " + "".join(f"{m:>26}" for m in a.methods))
    for i, t in enumerate(a.targets):
        print(f"{t:>6}% " + "".join(f"{100*table[m][i]['mean']:>17.2f} +- {100*table[m][i]['std']:.2f}" for m in a.methods))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pareto.json").write_text(json.dumps({"seeds": a.seeds, "protocol": "15 epochs, final-epoch model",
                                                 "curves": table}, indent=2))

    from train.figures import plot_pareto
    plot_pareto(table, a.seeds, out / "pareto.png")
    print(f"wrote {out}/pareto.json and pareto.png")


if __name__ == "__main__":
    main()
