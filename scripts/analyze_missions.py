"""Mission outcomes per arm and drift level, plus H4 (map mIoU vs mission safety).

Reads results/missions/*.csv and results/tables/scene_metrics.csv; writes
results/tables/missions.csv, results/figures/missions_vs_drift.png, results/figures/miou_vs_violations.png.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from s2m.viz import arm_line, setup

LEVELS = ("L0", "L1", "L2", "L3", "L4", "L5")
LEVEL_NOTE = {"L0": "GT", "L1": "0.6 cm", "L2": "3 cm", "L3": "9.5 cm", "L4": "63 cm", "L5": "3.5 m",
              "VO2": "VO, 2nd fr.", "VO4": "VO, 4th fr."}
ARMS = ("joint", "separate", "label_only", "pose_only", "uncalibrated", "label_cell", "oracle")


def as_bool(v):
    return v in ("True", "true", "1", True)


def load(paths):
    rows = []
    for p in paths:
        for r in csv.DictReader(open(p)):
            rows.append(r)
    return rows


def rate(rs, key):
    return float(np.mean([as_bool(r[key]) for r in rs])) if rs else np.nan


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("results/missions"))
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. _pb")
    ap.add_argument("--levels", nargs="+", default=list(LEVELS))
    ap.add_argument("--no-h4", action="store_true", help="skip the mIoU vs violations analysis")
    args = ap.parse_args()
    LEVELS = tuple(args.levels)
    rows = load(sorted(p for p in args.dir.glob("*.csv") if not p.name.startswith("params_")))
    print(f"{len(rows)} mission rows from {len({r['scene'] for r in rows})} scenes")

    table = []
    for lv in LEVELS:
        for arm in ARMS:
            rs = [r for r in rows if r["level"] == lv and r["arm"] == arm]
            planned = [r for r in rs if as_bool(r["planned"])]
            ratio = [float(r["length"]) / float(r["oracle_length"]) for r in planned
                     if r["oracle_length"] not in ("", "nan") and np.isfinite(float(r["oracle_length"]))]
            table.append({
                "level": lv, "arm": arm, "n": len(rs),
                "planned": rate(rs, "planned"), "violation": rate(rs, "violation"),
                "collision": rate(rs, "collision"), "success": rate(rs, "success"),
                "abstained": rate(rs, "abstained"),
                "violation_given_planned": rate(planned, "violation"),
                "length_ratio_median": float(np.median(ratio)) if ratio else np.nan,
            })
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    with open(f"results/tables/missions{args.tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(table)
    for t in table:
        print(f"{t['level']} {t['arm']:12s} plan {t['planned']:.2f} viol {t['violation']:.3f} "
              f"(|plan {t['violation_given_planned']:.3f}) coll {t['collision']:.3f} succ {t['success']:.2f} "
              f"abst {t['abstained']:.2f} len {t['length_ratio_median']:.2f}")

    setup()
    x = np.arange(len(LEVELS))
    xt = [f"{lv}\n{LEVEL_NOTE[lv]}" for lv in LEVELS]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for arm in ARMS:
        get = lambda k: [next(t[k] for t in table if t["level"] == lv and t["arm"] == arm) for lv in LEVELS]
        arm_line(axes[0], arm, x, get("violation_given_planned"))
        arm_line(axes[1], arm, x, get("success"))
        arm_line(axes[2], arm, x, get("planned"))
    for ax, title in zip(axes, ("violations among planned paths", "mission success", "path found")):
        ax.set_title(title)
        ax.set_xticks(x, xt)
        ax.set_xlabel("pose drift level (ATE)")
        ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("fraction of tasks")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(f"results/figures/missions_vs_drift{args.tag}.png")

    if args.no_h4:
        raise SystemExit
    # H4: map quality (mIoU at GT poses) vs safety of the uncalibrated planner at GT poses.
    metrics = {r["scene"]: r for r in csv.DictReader(open("results/tables/scene_metrics.csv"))}
    pts = []
    for scene in sorted({r["scene"] for r in rows}):
        rs = [r for r in rows if r["scene"] == scene and r["level"] == "L0" and r["arm"] == "uncalibrated"
              and as_bool(r["planned"])]
        if rs and scene in metrics:
            pts.append((float(metrics[scene]["miou"]), float(metrics[scene]["avoid_iou"]), rate(rs, "violation")))
    pts = np.array(pts)
    rho_m, p_m = spearmanr(pts[:, 0], pts[:, 2])
    rho_a, p_a = spearmanr(pts[:, 1], pts[:, 2])
    print(f"H4: Spearman(mIoU, violations) = {rho_m:.2f} (p={p_m:.2f}); "
          f"Spearman(avoid-class IoU, violations) = {rho_a:.2f} (p={p_a:.2f}); n = {len(pts)} scenes")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    for ax, col, name, rho in ((axes[0], 0, "mIoU (all classes)", rho_m), (axes[1], 1, "IoU of avoid classes", rho_a)):
        ax.scatter(pts[:, col], pts[:, 2], color="#2a78d6", s=28)
        ax.set_xlabel(f"{name} at GT poses")
        ax.set_title(f"Spearman ρ = {rho:.2f}")
    axes[0].set_ylabel("violation rate, uncalibrated planner")
    fig.suptitle("Does map quality predict mission safety? (one point per scene, L0)")
    fig.tight_layout()
    fig.savefig(f"results/figures/miou_vs_violations{args.tag}.png")
    with open(f"results/tables/h4_spearman{args.tag}.txt", "w") as f:
        f.write(f"n_scenes={len(pts)} rho_miou={rho_m:.3f} p={p_m:.3f} rho_avoid_iou={rho_a:.3f} p={p_a:.3f}\n")
