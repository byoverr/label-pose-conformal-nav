"""Our joint calibration vs the per-cell label calibration of Sundarsingh et al. on the same data.

A 2 x 2 design on ReplicaCAD with unselected ("random") start/goal tasks, so that the vocabulary and
the risk level are not confounded:
  vocabulary : closed (5 classes, their assumption) or open (98 prompts, our main setting);
  1 - alpha  : 0.95 (their setting) or 0.9 (ours);
each at known pose (L0) and with pose drift (L2, L3), which their method does not model.
"label_cell" is our implementation of their calibration on a final (not online) map, without their
exploration mode; it plans with the most conservative label of every cell, as they do. "geometry"
treats every occupied cell as the class (no semantics) with a calibrated radius.

Reads results/tables/coverage{_closed5_a005,_closed5,_a005,}.csv and
results/missions_rand{_closed5_a005,_closed5,_a005,}/; writes results/tables/comparison.csv and
results/figures/comparison.png.
"""

import matplotlib.pyplot as plt
import numpy as np

from s2m.analysis import load_scores
from s2m.io import DEV_SCENES, level_labels, mission_rows, rate, read_csv, write_csv
from s2m.viz import ARM_STYLE, setup

SETTINGS = {  # (vocabulary, alpha) -> file tag
    ("closed", 0.05): "_closed5_a005",
    ("closed", 0.1): "_closed5",
    ("open", 0.05): "_a005",
    ("open", 0.1): "",
}
LEVELS = ("L0", "L2", "L3")
ARMS = ("uncalibrated", "label_cell", "geometry", "joint")
CLASSES = ("indoor_plant", "bike")


if __name__ == "__main__":
    note = level_labels(load_scores("results/scores", exclude=DEV_SCENES)) | {"L0": "known pose"}
    table = []
    for (vocab, alpha), tag in SETTINGS.items():
        cov = read_csv(f"results/tables/coverage{tag}.csv")
        mis = mission_rows(f"results/missions_rand{tag}")
        for lv in LEVELS:
            for arm in ARMS + ("oracle",):
                rs = [r for r in mis if r["level"] == lv and r["arm"] == arm]
                real = {(r["scene"], r["seed"]): r for r in rs}.values()
                row = {"vocabulary": vocab, "alpha": alpha, "level": lv, "arm": arm, "n_tasks": len(rs),
                       "planned": rate(rs, "planned"), "violation": rate(rs, "violation"),
                       "collision": rate(rs, "collision"), "success": rate(rs, "success"),
                       "fell_back": float(np.mean([int(r["classes_fell_back"]) > 0 for r in real])) if real else np.nan}
                for cls in CLASSES:
                    c = next((r for r in cov if r["level"] == lv and r["class"] == cls and r["arm"] == arm), None)
                    for key in ("coverage_mean", "coverage_cert", "abstain_rate"):
                        row[f"{key}_{cls}"] = float(c[key]) if c else np.nan
                table.append(row)
    write_csv("results/tables/comparison.csv", table)
    for t in table:
        print(f"{t['vocabulary']:6s} a={t['alpha']:<4} {t['level']} {t['arm']:12s} "
              f"cov plant {t['coverage_mean_indoor_plant']:.2f} bike {t['coverage_mean_bike']:.2f} "
              f"abst plant {t['abstain_rate_indoor_plant']:.2f} | path {t['planned']:.2f} "
              f"viol {t['violation']:.3f} success {t['success']:.2f} fallback {t['fell_back']:.2f} (n={t['n_tasks']})")

    setup()
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), sharey=True)
    x = np.arange(len(LEVELS))
    for ax, ((vocab, alpha), _) in zip(axes, SETTINGS.items()):
        for arm in ARMS + ("oracle",):
            label, colour, marker = ARM_STYLE[arm]
            y = [next(t["success"] for t in table if (t["vocabulary"], t["alpha"], t["level"], t["arm"])
                      == (vocab, alpha, lv, arm)) for lv in LEVELS]
            ax.plot(x, y, color=colour, marker=marker, label=label)
        ax.set_title(f"{vocab} vocabulary, 1 - alpha = {1 - alpha:.2f}", fontsize=9)
        ax.set_xticks(x, [f"{lv}\n{note[lv]}" for lv in LEVELS])
        ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("mission success (safe path found)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.08), fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig("results/figures/comparison.png")
    print("saved results/figures/comparison.png")
