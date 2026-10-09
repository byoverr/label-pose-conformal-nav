"""Does the plan-time guarantee survive execution? Certified paths executed under continued drift.

The robot follows the planned path with its pose estimate still drifting (s2m.missions.execution_outcomes,
same drift model and level as the map). A path with clearance c_k to the region of class k, executed with
largest deviation delta from the commanded positions, can come closer than the safety distance to an
object of class k only if s_k + delta + eps > c_k (s_k: the scene's miss distance; eps = sqrt(2) * cell size
covers the cell-centre discretization of commanded and executed positions). Two certification rules at the
same total risk a:

  plan       risk_k(c_k) <= a / 2 for both classes (scripts/analyze_risk.py); says nothing about execution;
  execution  delta <= u_hat * L with u_hat the conformal (1 - a_e) quantile of the deviation per metre of
             executed calibration paths (other scenes, same level), a_e = a / 25, and
             risk_k(c_k - u_hat * L - eps) <= (a - a_e) / 2 for both classes; then by the union bound
             P(unsafe during execution and certified) <= a, PROVIDED the deviation of the chosen path is
             exchangeable with the pooled calibration deviations.
That proviso is only approximate: paths of one scene are dependent and the certified path is chosen by a rule
that depends on L, so the execution part is a check within the drift model, not an exact guarantee (the
scene-level part keeps the exact guarantee of the risk rule). Conformal risk control with the loss "unsafe
during execution" (scripts/analyze_crc.py --execution) gives an exact guarantee on the expected share instead.
The share a_e = a / 25 is small enough to keep a = 0.1 attainable with 20 calibration scenes (per-class map
risk can't go below 1/21); it was set by hand, not tuned. The denominator is every task realization, as in
scripts/analyze_risk.py.

Reads results/exec_{pb,rand}/; writes results/tables/execution.csv and results/figures/execution.png.
"""

import glob
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from s2m.experiment import R_MAX
from s2m.io import read_csv, write_csv
from s2m.viz import INK, MUTED, setup

AVOID = ("indoor_plant", "bike")
TARGETS = (0.1, 0.2, 0.3, 0.4)
LEVELS = ("L2", "L3", "L4")
EXEC_SHARE = 1 / 25
DISCRETIZATION = 2 ** 0.5 * 0.05  # commanded points lie within half a diagonal of a path cell centre, and danger is
                                  # judged at the centre of the cell an executed point falls in


def risk(cal: np.ndarray, c: float) -> float:
    return (1 + np.sum(cal >= min(c, R_MAX) - 1e-9)) / (len(cal) + 1)


def conformal_quantile(x: np.ndarray, alpha: float) -> float:
    m = math.ceil((len(x) + 1) * (1 - alpha))
    return math.inf if m > len(x) else float(np.sort(x)[m - 1])


def analyze(kind: str):
    paths = defaultdict(list)  # (scene, level, seed, task) -> rows of planned paths
    tasks_all = set()  # every task realization, with or without a path at some margin
    cal = {}
    for f in sorted(glob.glob(f"results/exec_{kind}/*.csv")):
        name = Path(f).stem
        if name.startswith("cal_"):
            for r in read_csv(f):
                for c in AVOID:
                    cal.setdefault((name[4:], r["level"]), defaultdict(list))[c].append(float(r[f"miss_{c}"]))
            continue
        for r in read_csv(f):
            tasks_all.add((r["scene"], r["level"], r["seed"], r["task"]))
            if r["planned"] == "True":
                paths[(r["scene"], r["level"], r["seed"], r["task"])].append(r)
    # deviation per metre of every distinct executed path, by scene and level
    dev = defaultdict(dict)
    for (s, lv, seed, task), rows in paths.items():
        for r in rows:
            dev[(s, lv)][(seed, task, r["exec_u"], r["exec_dev"])] = float(r["exec_u"])
    scenes = sorted({k[0] for k in paths})
    table = []
    for lv in LEVELS:
        keys = [k for k in paths if k[1] == lv]
        if not keys:
            continue
        n_all = sum(1 for k in tasks_all if k[1] == lv)  # all task realizations, as in analyze_risk.py
        u_all = np.concatenate([list(dev[(s, lv)].values()) for s in scenes if (s, lv) in dev])
        for a in TARGETS:
            a_e = a * EXEC_SHARE
            stats = {rule: defaultdict(float) for rule in ("plan", "execution")}
            u_hat = {}
            for key in keys:
                s = key[0]
                if s not in u_hat:
                    u_hat[s] = conformal_quantile(
                        np.concatenate([list(dev[(o, lv)].values()) for o in scenes if o != s and (o, lv) in dev]), a_e)
                cs = {c: np.array(cal[(s, lv)][c]) for c in AVOID}
                rows = sorted(paths[key], key=lambda r: float(r["margin"]))
                for rule in stats:
                    for r in rows:
                        shift = 0.0 if rule == "plan" else u_hat[s] * float(r["length"]) + DISCRETIZATION
                        budget = a / 2 if rule == "plan" else (a - a_e) / 2
                        if max(risk(cs[c], float(r[f"clearance_{c}"]) - shift) for c in AVOID) <= budget:
                            st = stats[rule]
                            st["cert"] += 1
                            st["plan_unsafe"] += r["violation"] == "True"
                            st["exec_unsafe"] += float(r["exec_unsafe"])
                            st["exec_collision"] += float(r["exec_collision"])
                            st["length"] += float(r["length"])
                            break
            for rule, st in stats.items():
                c = st["cert"]
                table.append({"tasks": kind, "level": lv, "target_risk": a, "rule": rule, "n": n_all,
                              "u_median": float(np.median(u_all)), "u_q95": float(np.quantile(u_all, 0.95)),
                              "u_hat_median": float(np.median(list(u_hat.values()))) if rule == "execution" else 0.0,
                              "certified": c / n_all,
                              "plan_unsafe_given_cert": st["plan_unsafe"] / c if c else np.nan,
                              "exec_unsafe_given_cert": st["exec_unsafe"] / c if c else np.nan,
                              "exec_unsafe_and_cert": st["exec_unsafe"] / n_all,
                              "exec_collision_given_cert": st["exec_collision"] / c if c else np.nan,
                              "length_mean": st["length"] / c if c else np.nan})
        print(f"{kind} {lv}: {n_all} task realizations; deviation per metre median {np.median(u_all):.4f}, "
              f"95% {np.quantile(u_all, 0.95):.4f}, max {u_all.max():.4f}")
        for t in table[-2 * len(TARGETS):]:
            print(f"   a={t['target_risk']:.1f} {t['rule']:9s} u_hat={t['u_hat_median']:.3f} cert {t['certified']:.2f} "
                  f"plan-unsafe|cert {t['plan_unsafe_given_cert']:.3f} exec-unsafe|cert {t['exec_unsafe_given_cert']:.3f} "
                  f"(and cert {t['exec_unsafe_and_cert']:.3f}) collision|cert {t['exec_collision_given_cert']:.2f} "
                  f"length {t['length_mean']:.1f} m")
    return table


if __name__ == "__main__":
    tables = {kind: analyze(kind) for kind in ("pb", "rand") if glob.glob(f"results/exec_{kind}/*.csv")}
    write_csv("results/tables/execution.csv", [r for t in tables.values() for r in t])
    setup()
    fig, axes = plt.subplots(1, len(tables), figsize=(5 * len(tables), 3.4), squeeze=False)
    for ax, (kind, table) in zip(axes[0], tables.items()):
        for lv, ls in zip(LEVELS, ("-", "--", ":")):
            for rule, colour in (("plan", MUTED), ("execution", INK)):
                t = [r for r in table if r["level"] == lv and r["rule"] == rule]
                ax.plot([r["target_risk"] for r in t], [r["exec_unsafe_and_cert"] for r in t], ls=ls, color=colour,
                        marker="o", ms=3, label=f"{lv}, {rule}")
        ax.plot([0, 0.4], [0, 0.4], color=MUTED, lw=0.8, ls=":")
        ax.set_xlabel("certified risk a")
        ax.set_title("hard tasks" if kind == "pb" else "random tasks")
    axes[0][0].set_ylabel("unsafe during execution and certified")
    axes[0][0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig("results/figures/execution.png")
    print("saved results/figures/execution.png")
