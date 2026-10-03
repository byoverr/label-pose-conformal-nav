"""Our joint calibration vs the per-cell label calibration of Sundarsingh et al. on the same data.

Two settings, both on ReplicaCAD with unselected ("random") start/goal tasks:
  closed : their assumptions - closed 5-class vocabulary, 1 - alpha = 0.95, known pose (L0),
           then the same with pose drift (L2, L3), which their method does not model;
  open   : our main setting - 98-prompt open vocabulary, 1 - alpha = 0.9, drift L0-L4.
"label_cell" is our implementation of their calibration on a final (not online) map, without their
exploration mode; it plans with the most conservative label of every cell, as they do.

Reads results/tables/coverage{_closed5_a005,}.csv and results/missions_rand{_closed5_a005,}/;
writes results/tables/comparison.csv and results/figures/comparison.png.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from s2m.viz import ARM_STYLE, MUTED, setup

SETTINGS = {
    "closed": {"coverage": "results/tables/coverage_closed5_a005.csv", "missions": "results/missions_rand_closed5_a005",
               "alpha": 0.05, "levels": ("L0", "L2", "L3"),
               "title": "their setting: closed 5-class vocabulary, 1 - alpha = 0.95"},
    "open": {"coverage": "results/tables/coverage.csv", "missions": "results/missions_rand",
             "alpha": 0.1, "levels": ("L0", "L2", "L3", "L4"),
             "title": "our setting: open vocabulary (98 prompts), 1 - alpha = 0.9"},
}
LEVEL_NOTE = {"L0": "known pose", "L1": "0.6 cm", "L2": "3 cm", "L3": "9.5 cm", "L4": "63 cm"}
ARMS = ("uncalibrated", "label_cell", "joint")
CLASSES = ("indoor_plant", "bike")


def mission_rows(path):
    rows = []
    for p in sorted(Path(path).glob("*.csv")):
        if not p.name.startswith("params_"):
            rows += list(csv.DictReader(open(p)))
    return rows


def rate(rs, key):
    return float(np.mean([r[key] == "True" for r in rs])) if rs else np.nan


if __name__ == "__main__":
    table = []
    for name, cfg in SETTINGS.items():
        cov = list(csv.DictReader(open(cfg["coverage"])))
        mis = mission_rows(cfg["missions"])
        for lv in cfg["levels"]:
            for arm in ARMS + ("oracle",):
                rs = [r for r in mis if r["level"] == lv and r["arm"] == arm]
                row = {"setting": name, "alpha": cfg["alpha"], "level": lv, "arm": arm, "n_tasks": len(rs),
                       "planned": rate(rs, "planned"), "violation": rate(rs, "violation"),
                       "collision": rate(rs, "collision"), "success": rate(rs, "success")}
                for cls in CLASSES:
                    c = next((r for r in cov if r["level"] == lv and r["class"] == cls and r["arm"] == arm), None)
                    row[f"coverage_{cls}"] = float(c["coverage_mean"]) if c else np.nan
                    row[f"abstain_{cls}"] = float(c["abstain_rate"]) if c else np.nan
                table.append(row)
    with open("results/tables/comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(table)
    for t in table:
        print(f"{t['setting']:6s} {t['level']} {t['arm']:12s} cov plant {t['coverage_indoor_plant']:.2f} "
              f"bike {t['coverage_bike']:.2f} | path {t['planned']:.2f} viol {t['violation']:.3f} "
              f"coll {t['collision']:.3f} success {t['success']:.2f} (n={t['n_tasks']})")

    setup()
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4))
    for row, (name, cfg) in enumerate(SETTINGS.items()):
        lv = cfg["levels"]
        x = np.arange(len(lv))
        get = lambda arm, key: [next(t[key] for t in table if t["setting"] == name and t["level"] == l
                                     and t["arm"] == arm) for l in lv]
        for arm in ARMS:
            label, colour, marker = ARM_STYLE[arm]
            axes[row][0].plot(x, get(arm, "coverage_bike"), color=colour, marker=marker, label=label)
            axes[row][0].plot(x, get(arm, "coverage_indoor_plant"), color=colour, marker=marker, ls="--")
            axes[row][1].plot(x, get(arm, "success"), color=colour, marker=marker, label=label)
            axes[row][2].plot(x, get(arm, "violation"), color=colour, marker=marker, label=label)
        label, colour, marker = ARM_STYLE["oracle"]
        axes[row][1].plot(x, get("oracle", "success"), color=colour, marker=marker, label=label)
        axes[row][0].axhline(1 - cfg["alpha"], color=MUTED, lw=1)
        axes[row][0].set_ylabel(cfg["title"].split(":")[0] + "\n" + cfg["title"].split(": ")[1], fontsize=8)
        for ax in axes[row]:
            ax.set_xticks(x, [f"{l}\n{LEVEL_NOTE[l]}" for l in lv])
        axes[row][0].set_ylim(-0.02, 1.02)
        axes[row][1].set_ylim(-0.02, 1.02)
    axes[0][0].set_title("coverage (solid: bike, dashed: plant)")
    axes[0][1].set_title("mission success (safe path to the goal)")
    axes[0][2].set_title("unsafe missions (path closer than 0.5 m)")
    handles, labels = axes[0][1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04), fontsize=8)
    fig.suptitle("Random start/goal tasks: per-cell label calibration (Sundarsingh et al.-style) vs joint miss-distance "
                 "calibration", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig("results/figures/comparison.png")
    print("saved results/figures/comparison.png")
