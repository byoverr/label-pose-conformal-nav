"""Risk level alpha vs coverage, keep-out radius and mission feasibility (the price of the guarantee).

Reads results/tables/coverage{_a005,,_a02,_a03}.csv (scripts/analyze_coverage.py) and
results/missions_pb{_a005,,_a02,_a03}/ (scripts/run_missions.py); writes
results/tables/alpha_sweep.csv and results/figures/alpha_sweep.png.

Example: python scripts/analyze_alpha.py --level L3
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from s2m.viz import ARM_STYLE, MUTED, setup

ALPHAS = {0.05: "_a005", 0.1: "", 0.2: "_a02", 0.3: "_a03"}
CLASSES = {"indoor_plant": "-", "bike": "--"}
ARMS = ("joint", "separate")


def coverage_rows(tag, level):
    return [r for r in csv.DictReader(open(f"results/tables/coverage{tag}.csv")) if r["level"] == level]


def mission_rows(tag, level):
    rows = []
    for p in sorted(Path(f"results/missions_pb{tag}").glob("*.csv")):
        if not p.name.startswith("params_"):
            rows += [r for r in csv.DictReader(open(p)) if r["level"] == level]
    return rows


def rate(rows, arm, key):
    v = [r[key] == "True" for r in rows if r["arm"] == arm]
    return float(np.mean(v)) if v else np.nan


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="L3")
    args = ap.parse_args()

    table = []
    for alpha, tag in ALPHAS.items():
        cov = coverage_rows(tag, args.level)
        mis = mission_rows(tag, args.level)
        for arm in ARMS + ("uncalibrated", "oracle"):
            row = {"alpha": alpha, "level": args.level, "arm": arm,
                   "planned": rate(mis, arm, "planned"), "violation": rate(mis, arm, "violation"),
                   "success": rate(mis, arm, "success")}
            for cls in CLASSES:
                c = next((r for r in cov if r["class"] == cls and r["arm"] == arm), None)
                for key in ("coverage_mean", "abstain_rate", "radius_median"):
                    row[f"{key}_{cls}"] = float(c[key]) if c else np.nan
            table.append(row)
    with open("results/tables/alpha_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(table)
    for t in table:
        print(f"alpha {t['alpha']:.2f} {t['arm']:12s} planned {t['planned']:.2f} viol {t['violation']:.3f} "
              + " ".join(f"{c}: cov {t[f'coverage_mean_{c}']:.2f} r {t[f'radius_median_{c}']:.2f} "
                         f"abst {t[f'abstain_rate_{c}']:.2f}" for c in CLASSES))

    setup()
    al = np.array(list(ALPHAS))
    get = lambda arm, key: np.array([t[key] for t in table if t["arm"] == arm])
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    ax = axes[0]
    ax.plot(al, 1 - al, color=MUTED, lw=1, label="target 1 - alpha")
    for arm in ARMS:
        label, colour, marker = ARM_STYLE[arm]
        for cls, ls in CLASSES.items():
            ax.plot(al, get(arm, f"coverage_mean_{cls}"), ls=ls, color=colour, marker=marker,
                    label=f"{arm}, {cls.replace('_', ' ')}")
    ax.set_title("coverage (abstention counted as covered)")
    ax.set_ylabel("fraction")
    ax.legend(fontsize=7, loc="lower left")

    ax = axes[1]
    for arm in ARMS:
        label, colour, marker = ARM_STYLE[arm]
        for cls, ls in CLASSES.items():
            ax.plot(al, get(arm, f"radius_median_{cls}"), ls=ls, color=colour, marker=marker)
    notes = [f"alpha {a:g}: {ab:.0%}" for a, ab in zip(al, get("joint", "abstain_rate_indoor_plant")) if ab > 0.05]
    if notes:
        ax.text(0.03, 0.04, "joint abstains on plant\n" + "\n".join(notes), transform=ax.transAxes,
                fontsize=7, color=MUTED, va="bottom")
    ax.set_ylim(0, None)
    ax.set_title("keep-out radius (median, metres)")

    ax = axes[2]
    for arm in ARMS + ("oracle",):
        label, colour, marker = ARM_STYLE[arm]
        ax.plot(al, get(arm, "planned"), color=colour, marker=marker, label=f"{arm}: path found")
    label, colour, marker = ARM_STYLE["uncalibrated"]
    ax.plot(al, get("uncalibrated", "violation"), color=colour, marker=marker, ls=":",
            label="uncalibrated: violations")
    ax.plot(al, get("joint", "violation"), color=ARM_STYLE["joint"][1], marker="o", ls=":",
            label="joint: violations")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("pass-by missions")
    ax.legend(fontsize=7, loc="center right")
    for ax in axes:
        ax.set_xlabel("risk level alpha")
        ax.set_xticks(al)
    fig.suptitle(f"The price of the guarantee, drift {args.level}", fontsize=10)
    fig.tight_layout()
    fig.savefig("results/figures/alpha_sweep.png")
    print("saved results/figures/alpha_sweep.png")
