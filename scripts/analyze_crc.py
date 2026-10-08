"""Calibrating the planner instead of the map: conformal risk control over the margin.

The map-level guarantee covers every object of the class, also those far from any path. Conformal risk
control (Angelopoulos et al., arXiv 2208.02814) calibrates what the robot does: for a margin m on the grid,
scene i's loss L_i(m) is the fraction of its tasks (over drift realizations) whose path planned with
margin m comes closer than the safety distance to an avoid-class object (0 when no path is found). With
the monotone envelope L~_i(m) = max_{m' >= m} L_i(m') and n calibration scenes, the smallest margin with
    n / (n + 1) * mean_i L~_i(m) + 1 / (n + 1) <= a
gives E[L_test] <= a for an exchangeable test scene: the expected fraction of unsafe missions, not the
probability that a path is unsafe. Only scenes with at least one task take part. With --execution the
loss is the probability of an unsafe execution under continued drift (results/exec_*), which makes the
guarantee hold during execution without a separate deviation bound.

For comparison the per-path risk rule of scripts/analyze_risk.py (certified if risk_k <= a / 2 for both
classes) runs on the same rows. Leave-one-scene-out. Writes results/tables/crc_<root>[_execution].csv.

Example: python scripts/analyze_crc.py; python scripts/analyze_crc.py --execution
"""

import argparse
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np

from s2m.experiment import R_MAX
from s2m.io import read_csv, write_csv

AVOID = ("indoor_plant", "bike")
TARGETS = (0.05, 0.1, 0.2)


def risk(cal: np.ndarray, c: float) -> float:
    return (1 + np.sum(cal >= min(c, R_MAX) - 1e-9)) / (len(cal) + 1)


def load(kind: str, root: str):
    rows, cal = defaultdict(list), {}
    for f in sorted(glob.glob(f"results/{root}_{kind}/*.csv")):
        name = Path(f).stem
        if name.startswith("cal_"):
            for r in read_csv(f):
                for c in AVOID:
                    cal.setdefault((name[4:], r["level"]), defaultdict(list))[c].append(float(r[f"miss_{c}"]))
            continue
        for r in read_csv(f):
            rows[(r["scene"], r["level"])].append(r)
    return rows, cal


def analyze(kind: str, execution: bool, root: str):
    rows, cal = load(kind, root)
    levels = sorted({k[1] for k in rows})
    margins = sorted({float(r["margin"]) for rr in rows.values() for r in rr})
    unsafe_key = "exec_unsafe" if execution else "violation"
    table = []
    for lv in levels:
        scenes = sorted(s for (s, l) in rows if l == lv and rows[(s, l)])
        # per scene: task realizations -> per margin (planned, unsafe, success)
        per = {}
        for s in scenes:
            tasks = defaultdict(dict)
            for r in rows[(s, lv)]:
                planned = r["planned"] == "True"
                unsafe = (float(r[unsafe_key]) if execution else float(r["violation"] == "True")) if planned else 0.0
                succ = (1.0 - unsafe) if planned else 0.0  # planned paths reach the goal (risk rows keep planned only)
                tasks[(r["seed"], r["task"])][float(r["margin"])] = (planned, unsafe, succ, r)
            per[s] = tasks
        loss = {s: np.array([np.mean([per[s][t][m][1] for t in per[s]]) for m in margins]) for s in scenes}
        env = {s: np.maximum.accumulate(loss[s][::-1])[::-1] for s in scenes}
        for a in TARGETS:
            res = {"crc": [], "path": []}
            m_hat, scene_unsafe = [], {"crc": [], "path": []}
            for s in scenes:
                others = [o for o in scenes if o != s]
                n = len(others)
                bound = n / (n + 1) * np.mean([env[o] for o in others], axis=0) + 1 / (n + 1)
                ok = np.flatnonzero(bound <= a)
                if len(ok):
                    m = margins[ok[0]]
                    m_hat.append(m)
                    for t in per[s]:
                        planned, unsafe, succ, _ = per[s][t][m]
                        res["crc"].append((planned, unsafe, succ))
                    scene_unsafe["crc"].append(np.mean([per[s][t][m][1] for t in per[s]]))
                else:
                    m_hat.append(np.inf)
                    res["crc"] += [(False, 0.0, 0.0)] * len(per[s])
                    scene_unsafe["crc"].append(0.0)
                # per-path risk rule with the scene's calibration misses
                cs = {c: np.array(cal[(s, lv)][c]) for c in AVOID} if (s, lv) in cal else None
                for t in per[s]:
                    hit = (False, 0.0, 0.0)
                    for m in margins:
                        planned, unsafe, succ, r = per[s][t][m]
                        if planned and cs is not None and \
                                max(risk(cs[c], float(r[f"clearance_{c}"])) for c in AVOID) <= a / 2:
                            hit = (True, unsafe, succ)
                            break
                    res["path"].append(hit)
                scene_unsafe["path"].append(np.mean([h[1] for h in res["path"][-len(per[s]):]]))
            for rule, rr in res.items():
                rr = np.array(rr, float)
                table.append({"source": root, "tasks": kind, "loss": "execution" if execution else "plan", "level": lv,
                              "target": a, "rule": rule, "n_scenes": len(scenes), "n": len(rr),
                              "margin_median": float(np.median(m_hat)) if rule == "crc" else np.nan,
                              "planned": rr[:, 0].mean(), "unsafe": rr[:, 1].mean(), "success": rr[:, 2].mean(),
                              "unsafe_scene_mean": float(np.mean(scene_unsafe[rule])),
                              "unsafe_scene_max": float(np.max(scene_unsafe[rule]))})
            c, p = table[-2], table[-1]
            print(f"{root} {kind} {'exec' if execution else 'plan'} {lv} a={a:.2f}: CRC margin {c['margin_median']:.2f} m, "
                  f"path {c['planned']:.2f}, unsafe {c['unsafe']:.3f} (scene mean {c['unsafe_scene_mean']:.3f}, "
                  f"worst {c['unsafe_scene_max']:.2f}), success {c['success']:.2f} | per-path rule: "
                  f"path {p['planned']:.2f}, unsafe {p['unsafe']:.3f}, success {p['success']:.2f}")
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--execution", action="store_true")
    ap.add_argument("--root", default=None, help="results/<root>_{pb,rand}: risk (default), exec, risk_slam, risk_ens")
    args = ap.parse_args()
    root = args.root or ("exec" if args.execution else "risk")
    table = [r for kind in ("pb", "rand") if glob.glob(f"results/{root}_{kind}/*.csv")
             for r in analyze(kind, args.execution, root)]
    write_csv(f"results/tables/crc_{root}{'_execution' if args.execution else ''}.csv", table)
