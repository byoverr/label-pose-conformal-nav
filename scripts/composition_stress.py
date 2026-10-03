"""Stress test of the naive composition (sum of separate quantiles) under dependent failures.

Labels keep their real per-scene errors (YOLO-World on the scene). Pose error is replaced by a
scene-level mixture: k of the 21 scenes are "pose-failure" scenes whose realizations come from a
high drift level (VO lost track and dead-reckoned), the rest come from L1 (negligible drift).
Which scenes fail is the dependence structure:
  aligned  : pose fails in the k scenes where labels are worst (positive dependence),
  disjoint : pose fails in the k scenes where labels are best (failures never coincide),
  random   : a random k-subset, averaged over draws (independence).
The separate arm's guarantee is only 1 - 2*alpha by the union bound; the joint quantile is valid
whatever the dependence. Writes results/tables/composition.csv and
results/figures/composition_stress.png.

Example: python scripts/composition_stress.py --alpha 0.3 --fail-level L5
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from s2m.analysis import LAM0, load_scores, miss_at, run_splits
from s2m.viz import ARM_STYLE, MUTED, setup

CLASSES = ("indoor_plant", "bike")
DEPENDENCE = {"aligned": "--", "random": ":", "disjoint": "-"}


def mixture(rows, fail: set[str], fail_level: str) -> list[dict]:
    """L0 rows plus a 'MIX' level: fail_level realizations in `fail` scenes, L1 elsewhere."""
    out = [r for r in rows if r["level"] == "L0"]
    out += [{**r, "level": "MIX"} for r in rows
            if (r["level"] == fail_level and r["scene"] in fail) or (r["level"] == "L1" and r["scene"] not in fail)]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, default=Path("results/scores"))
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--fail-level", default="L5")
    ap.add_argument("--n-cal", type=int, default=13)
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--random-draws", type=int, default=10)
    ap.add_argument("--kmax", type=int, default=7)
    ap.add_argument("--replot", action="store_true", help="only redraw the figure from the saved table")
    args = ap.parse_args()

    tag = f"_a{str(args.alpha).replace('.', '')}_{args.fail_level}"
    table_path = Path(f"results/tables/composition{tag}.csv")
    rows = load_scores(args.scores, exclude=("apt_0",))
    scenes = sorted({r["scene"] for r in rows})
    rng = np.random.default_rng(0)
    table = []
    for cls in () if args.replot else CLASSES:
        label_miss = {r["scene"]: miss_at(r, cls, LAM0) for r in rows if r["level"] == "L0"}
        order = sorted(scenes, key=lambda s: label_miss[s])  # best labels first
        for k in range(args.kmax + 1):
            subsets = {"aligned": [order[len(order) - k:] if k else []], "disjoint": [order[:k]],
                       "random": [list(rng.choice(scenes, k, replace=False)) for _ in range(args.random_draws)]}
            for dep, draws in subsets.items():
                splits = args.splits if dep != "random" else max(args.splits // args.random_draws, 20)
                acc = {a: {"coverage": [], "radius": [], "abstain": []} for a in ("separate", "joint")}
                for fail in draws:
                    st = run_splits(mixture(rows, set(fail), args.fail_level), "MIX", args.alpha,
                                    args.n_cal, splits, classes=(cls,))[cls]
                    for a in acc:
                        for key in acc[a]:
                            acc[a][key] += st[a][key]
                for a, v in acc.items():
                    table.append({"class": cls, "k_fail": k, "frac_fail": k / len(scenes), "dependence": dep,
                                  "arm": a, "coverage_mean": np.mean(v["coverage"]),
                                  "radius_median": np.nanmedian(v["radius"]) if np.isfinite(v["radius"]).any() else np.nan,
                                  "abstain_rate": np.mean(v["abstain"])})
                print(cls, k, dep, " ".join(f"{t['arm']} {t['coverage_mean']:.3f} r {t['radius_median']:.2f}"
                                            for t in table[-2:]), flush=True)

    if args.replot:
        table = [{k: (v if k in ("class", "dependence", "arm") else float(v)) for k, v in r.items()}
                 for r in csv.DictReader(open(table_path))]
    else:
        with open(table_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(table[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(table)

    setup()
    fig, axes = plt.subplots(1, len(CLASSES), figsize=(9, 3.4), sharey=True)
    for ax, cls in zip(axes, CLASSES):
        for arm in ("separate", "joint"):
            label, colour, marker = ARM_STYLE[arm]
            for dep, ls in DEPENDENCE.items():
                t = [r for r in table if r["class"] == cls and r["arm"] == arm and r["dependence"] == dep]
                ax.plot([r["frac_fail"] for r in t], [r["coverage_mean"] for r in t], ls=ls, color=colour,
                        marker=marker, ms=4, label=f"{label.split(',')[0] if arm == 'separate' else 'joint'}, {dep}")
        ax.axhline(1 - args.alpha, color=MUTED, lw=1)
        ax.text(0, 1 - args.alpha - 0.005, f"target {1 - args.alpha:.2f}", va="top", fontsize=7, color=MUTED)
        ax.axhline(1 - 2 * args.alpha, color=MUTED, lw=1, ls=":")
        ax.text(0, 1 - 2 * args.alpha + 0.005, "union bound for separate", va="bottom", fontsize=7, color=MUTED)
        ax.set_title(cls.replace("_", " "))
        ax.set_xlabel(f"fraction of scenes with a pose failure ({args.fail_level} drift)")
    axes[0].set_ylabel("coverage on test scenes")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(f"Composition stress test, alpha = {args.alpha}: real label errors, injected scene-level pose failures",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(f"results/figures/composition_stress{tag}.png")
    print("saved")
