"""Aggregate train_prune results over seeds.

    python -m train.summarize --dir results/part3

For each (criterion, target): mean +- sample std of final test accuracy over seeds.
For saliency vs magnitude at the same target: PAIRED per-seed differences.  Same seed => same init and same batch
order up to the end of warmup, so pairing removes the init/order variance both runs share and isolates the effect
of the criterion.  Reports mean diff, std, a paired t statistic and the two-sided 5% critical value.
Writes <dir>/summary.json.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

# two-sided 5% critical values of Student's t, df = 1..10
T_CRIT_05 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def mean_std(xs):
    n = len(xs)
    m = sum(xs) / n
    s = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else float("nan")
    return m, s


def load(d: Path):
    runs = defaultdict(dict)  # (criterion, target) -> {seed: record}
    for f in sorted(d.glob("*_seed*.json")):
        m = re.match(r"(dense|saliency|magnitude)(?:_s(\d+))?_seed(\d+)\.json", f.name)
        if not m:
            continue
        r = json.loads(f.read_text())
        crit = m.group(1)
        target = 0 if crit == "dense" else int(m.group(2))
        runs[(crit, target)][int(m.group(3))] = r
    return runs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/part3")
    a = ap.parse_args(argv)
    d = Path(a.dir)
    runs = load(d)
    out = {"groups": [], "paired": []}
    print(f"{'run':<16}{'n':>3}  {'test acc mean +- std':>22}  {'min':>7} {'max':>7}  seeds")
    for (crit, t), by_seed in sorted(runs.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        accs = [by_seed[s]["test_acc"] for s in sorted(by_seed)]
        m, s = mean_std(accs)
        name = crit if crit == "dense" else f"{crit}@{t}%"
        print(f"{name:<16}{len(accs):>3}  {100*m:>12.2f} +- {100*s:.2f}  {100*min(accs):>7.2f} {100*max(accs):>7.2f}  {sorted(by_seed)}")
        out["groups"].append({"criterion": crit, "target_pct": t, "seeds": sorted(by_seed), "test_acc": accs,
                              "mean": m, "std": s})
    targets = sorted({t for (c, t) in runs if c != "dense"})
    for t in targets:
        sal, mag = runs.get(("saliency", t), {}), runs.get(("magnitude", t), {})
        seeds = sorted(set(sal) & set(mag))
        if len(seeds) < 2:
            continue
        diffs = [sal[s]["test_acc"] - mag[s]["test_acc"] for s in seeds]
        m, s = mean_std(diffs)
        tstat = m / (s / math.sqrt(len(diffs))) if s > 0 else float("inf")
        crit = T_CRIT_05.get(len(diffs) - 1)
        verdict = "significant at 5%" if crit and abs(tstat) > crit else "NOT significant at 5% (within noise)"
        print(f"\npaired saliency - magnitude @ {t}%  n={len(seeds)}  per-seed {[round(100*x, 2) for x in diffs]}")
        print(f"  mean {100*m:+.2f} pp  std {100*s:.2f} pp  t={tstat:+.2f}  (|t| crit {crit})  -> {verdict}")
        out["paired"].append({"target_pct": t, "seeds": seeds, "diff_sal_minus_mag": diffs, "mean": m, "std": s,
                              "t": tstat, "t_crit_05": crit, "significant_05": bool(crit and abs(tstat) > crit)})
    (d / "summary.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {d}/summary.json")


if __name__ == "__main__":
    main()
