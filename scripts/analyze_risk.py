"""Risk as an output instead of abstention: certified risk of each planned path.

A path with clearance c_k to the region of class k (distance minus the safety distance) can be unsafe for k
only if the scene's miss distance s_k exceeds c_k. With the capped miss distances of the n calibration scenes,
    risk_k(c) = (1 + #{i : s_i,k >= min(c, R_MAX)}) / (n + 1)
satisfies P(path unsafe for k and risk_k <= a) <= a even though the path was chosen on the test map (the
conformal p-value argument). Over two classes the budget is split evenly: a path is certified at level a
when risk_k <= a / 2 for every class, so P(unsafe and certified at a) <= a.
For every target a the planner takes the smallest margin on the grid whose path is certified.

Reads results/risk_{pb,rand}/; writes results/tables/risk_{pb,rand}.csv and results/figures/risk_curve.png.
"""

import glob
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from s2m.experiment import R_MAX
from s2m.io import read_csv, write_csv
from s2m.viz import INK, MUTED, setup

AVOID = ("indoor_plant", "bike")
TARGETS = (0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0)
LEVELS = ("L0", "L2", "L3", "L4")


def risk(cal: np.ndarray, c: float) -> float:
    return (1 + np.sum(cal >= min(c, R_MAX) - 1e-9)) / (len(cal) + 1)


def analyze(kind: str):
    paths = defaultdict(list)  # (scene, level, seed, task) -> rows sorted by margin
    cal = {}
    for f in sorted(glob.glob(f"results/risk_{kind}/*.csv")):
        name = Path(f).stem
        if name.startswith("cal_"):
            for r in read_csv(f):
                cal.setdefault((name[4:], r["level"]), defaultdict(list))
                for c in AVOID:
                    cal[(name[4:], r["level"])][c].append(float(r[f"miss_{c}"]))
            continue
        for r in read_csv(f):
            paths[(r["scene"], r["level"], r["seed"], r["task"])].append(r)
    table, amin = [], defaultdict(list)
    for lv in LEVELS:
        keys = [k for k in paths if k[1] == lv]
        stats = {a: {"cert": 0, "unsafe": 0, "coll": 0} for a in TARGETS}
        for key in keys:
            cs = {c: np.array(cal[(key[0], lv)][c]) for c in AVOID}
            rows = sorted(paths[key], key=lambda r: float(r["margin"]))
            best = np.inf
            for a in TARGETS:
                for r in rows:
                    if r["planned"] != "True":
                        continue
                    rk = max(risk(cs[c], float(r[f"clearance_{c}"])) for c in AVOID)
                    best = min(best, 2 * rk)
                    if rk <= a / 2:
                        stats[a]["cert"] += 1
                        stats[a]["unsafe"] += r["violation"] == "True"
                        stats[a]["coll"] += r["collision"] == "True"
                        break
            amin[lv].append(best)
        n = len(keys)
        for a in TARGETS:
            s = stats[a]
            table.append({"tasks": kind, "level": lv, "target_risk": a, "n": n, "certified": s["cert"] / n,
                          "unsafe_and_certified": s["unsafe"] / n,
                          "unsafe_given_certified": s["unsafe"] / s["cert"] if s["cert"] else np.nan,
                          "collision_given_certified": s["coll"] / s["cert"] if s["cert"] else np.nan})
        b = np.array(amin[lv])
        fin = b[np.isfinite(b)]
        print(f"{kind} {lv}: n={n}, min certified risk: median {np.median(fin):.2f} (of tasks with any path {len(fin) / n:.2f}); "
              + " ".join(f"a={t['target_risk']}: cert {t['certified']:.2f} unsafe|cert {t['unsafe_given_certified']:.3f}"
                         for t in table[-len(TARGETS):]))
    write_csv(f"results/tables/risk_{kind}.csv", table)
    return table


if __name__ == "__main__":
    tables = {kind: analyze(kind) for kind in ("pb", "rand") if glob.glob(f"results/risk_{kind}/*.csv")}
    setup()
    fig, axes = plt.subplots(1, len(tables), figsize=(5 * len(tables), 3.4), squeeze=False)
    for ax, (kind, table) in zip(axes[0], tables.items()):
        for lv, ls in zip(LEVELS, ("-", "--", "-.", ":")):
            t = [r for r in table if r["level"] == lv]
            ax.plot([r["target_risk"] for r in t], [r["certified"] for r in t], ls=ls, color=INK, marker="o", ms=4,
                    label=f"{lv}: path certified")
            ax.plot([r["target_risk"] for r in t], [r["unsafe_and_certified"] for r in t], ls=ls, color=MUTED, lw=1)
        ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=":")
        ax.set_xlabel("certified risk a (both classes)")
        ax.set_title("hard tasks" if kind == "pb" else "random tasks")
        ax.set_ylim(-0.02, 1.02)
    axes[0][0].set_ylabel("fraction of tasks")
    axes[0][0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig("results/figures/risk_curve.png")
    print("saved results/figures/risk_curve.png")
