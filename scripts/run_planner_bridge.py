"""Where the planner matters: sampling-based planners on calibrated (tightened) maps.

The guarantee of the joint arm does not depend on the planner: any path that respects the
inflated keep-out zones is safe with probability >= 1 - alpha. What the planner decides is how
often, and how fast, it finds such a path once inflation narrows the free space. This script
takes the joint arm's leave-one-scene-out radius q at one drift level, scales it by
m in {0, 0.5, 1, 1.5}, keeps only tasks that are still solvable (exact Dijkstra on the grid),
and runs RRT-Connect with uniform sampling and with the angle-limited dynamic sampling zone
(Dovgopolik's bidirectional RRT) under the same iteration budget.

Writes results/tables/planner_bridge.csv.
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from s2m.analysis import calibrate_class, load_scores
from s2m.data import load_scene
from s2m.drift import simulate, world_correction
from s2m.entities import AVOID_CLASSES, class_ids
from s2m.experiment import load_levels, prepare, stable_seed
from s2m.mapping import build_map, precompute_observations
from s2m.missions import GOAL_TOL, _disk, _to_cell, keepout, make_tasks
from s2m.perception import load_detections
from s2m.planning import ROBOT_RADIUS, dilate, path_length, plan_many
from s2m.analysis import ArmParams
from s2m.rrt import RRTConfig, path_length as rrt_length, rrt_connect

DEV_SCENES = ("apt_0",)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="L2")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--tasks", type=int, default=20)
    ap.add_argument("--rrt-seeds", type=int, default=3)
    ap.add_argument("--budget", type=int, default=1500)
    ap.add_argument("--scales", type=float, nargs="+", default=[0.0, 0.5, 1.0, 1.5])
    ap.add_argument("--part", default="0/1", help="i/n: process every n-th scene starting at i")
    ap.add_argument("--avoid", nargs="+", default=list(AVOID_CLASSES), help="avoid classes used in planning")
    ap.add_argument("--out", type=Path, default=Path("results/tables/planner_bridge.csv"))
    args = ap.parse_args()

    levels = load_levels("configs/drift_levels.yaml")
    rows = load_scores(Path("results/scores"), exclude=DEV_SCENES)  # all scenes calibrate
    all_scenes = sorted({r["scene"] for r in rows})
    part, nparts = map(int, args.part.split("/"))
    by = {}
    for r in rows:
        by.setdefault((r["scene"], r["level"]), []).append(r)

    out = []
    for name in all_scenes[part::nparts]:
        t0 = time.time()
        rng = np.random.default_rng(stable_seed(3, name))
        others = [s for s in all_scenes if s != name]  # calibrate on every other scene
        cal = [by[(s, args.level)][rng.integers(len(by[(s, args.level)]))] for s in others]
        cal_L0 = [by[(s, "L0")][0] for s in others]
        scene = load_scene(Path("data/replica_cad") / name)
        cls_ids = dict(zip(args.avoid, class_ids(scene.classes, args.avoid)))
        per_class = {cid: calibrate_class(cal, cal_L0, cname, args.alpha) for cname, cid in cls_ids.items()}
        joint = {cid: pc["joint"] for cid, pc in per_class.items()}
        fallback = {cid: None if pc["pose_only"] is None else pc["pose_only"].radius for cid, pc in per_class.items()}
        setup = prepare(scene, precompute_observations(
            scene, load_detections(Path("data/cache/detections") / f"{name}.npz")), args.avoid)
        tasks = make_tasks(setup, args.tasks, seed=0)
        est = simulate(scene.poses, levels[args.level], np.random.default_rng(stable_seed(7, 0, name)))
        D_q = world_correction(est, scene.poses)[-1]
        pred = build_map(setup.obs, est, setup.spec, setup.n_classes, "det")
        res, shape = setup.spec.res, setup.spec.shape
        base = pred.free() & ~dilate(pred.occupied(), ROBOT_RADIUS, res)
        cells = [(_to_cell(setup.spec, t.start_xz, D_q), _to_cell(setup.spec, t.goal_xz, D_q)) for t in tasks]
        cells = [c for c in cells if c[0] is not None and c[1] is not None]

        for m in args.scales:
            scaled = {cid: None if p is None else ArmParams(p.lam, m * p.radius) for cid, p in joint.items()}
            fb = {cid: None if r is None else m * r for cid, r in fallback.items()}
            keep = keepout(pred, setup.avoid_ids, scaled, fb, res)
            if keep is None:
                continue
            trav = base & ~keep
            exact = plan_many(trav, [c[0] for c in cells], [_disk(shape, c[1], 0.0, res) for c in cells], res)
            free_frac = float(trav.sum() / max(base.sum(), 1))
            for ti, ((s_rc, g_rc), ref) in enumerate(zip(cells, exact)):
                if ref is None:
                    out.append({"scene": name, "scale": m, "task": ti,
                                "solvable": False, "free_frac": free_frac})
                    continue
                ref_len = path_length(ref, res)
                for planner, cone, widen in (("rrt_connect", False, "sample"), ("angle_zone", True, "sample"),
                                             ("angle_zone_adaptive", True, "extension")):
                    for seed in range(args.rrt_seeds):
                        cfg = RRTConfig(cone=cone, widen_on=widen, max_iter=args.budget, goal_tol=GOAL_TOL)
                        t = time.time()
                        path, it = rrt_connect(trav, s_rc, g_rc, res, cfg, np.random.default_rng(stable_seed(seed, ti, name)))
                        out.append({"scene": name, "scale": m, "task": ti,
                                    "solvable": True, "free_frac": free_frac, "planner": planner,
                                    "seed": seed, "solved": path is not None, "iterations": it,
                                    "time_s": time.time() - t, "ref_length": ref_len,
                                    "length_ratio": rrt_length(path) / ref_len if path is not None else np.nan})
        radii = ", ".join(f"{scene.classes[c]}={'abstain' if p is None else f'{p.radius:.2f}'}" for c, p in joint.items())
        print(f"{name}: joint radii {radii}; {len(cells)} tasks, {time.time() - t0:.0f} s", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    keys = ["scene", "scale", "task", "solvable", "free_frac", "planner", "seed", "solved",
            "iterations", "time_s", "ref_length", "length_ratio"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        w.writerows(out)
