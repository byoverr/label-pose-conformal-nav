"""Where the planner matters: sampling-based planners on calibrated (tightened) maps.

The guarantee of the joint arm does not depend on the planner: any path that respects the
inflated keep-out zones is safe with probability >= 1 - alpha. What the planner decides is how
often, and how fast, it finds such a path once inflation narrows the free space. This script
takes the joint arm's leave-one-scene-out margin at one drift level from a missions run
(params_<scene>.csv, i.e. exactly the margin the missions used), scales it by m in
{0, 0.5, 1, 1.5}, keeps only tasks that are still solvable (exact Dijkstra on the grid), and
runs RRT-Connect with uniform sampling and with the angle-limited dynamic sampling zone
(Dovgopolik's bidirectional RRT) under the same iteration budget. The pose realization is the
missions' seed 0.

Writes results/tables/planner_bridge<tag>_<part>.csv (one file per --part).

Example: python scripts/run_planner_bridge.py --avoid indoor_plant bike --params results/missions_pb --tag _pb
"""

import argparse
import time
from pathlib import Path

import numpy as np

from s2m.analysis import ArmParams, read_params
from s2m.data import load_scene
from s2m.drift import simulate, world_correction
from s2m.entities import AVOID_CLASSES, class_ids
from s2m.experiment import load_levels, prepare, stable_seed
from s2m.io import write_csv
from s2m.mapping import build_map, precompute_observations
from s2m.missions import disk, keepout, make_tasks, to_cell
from s2m.perception import load_detections
from s2m.planning import ROBOT_RADIUS, dilate, path_length, plan_many
from s2m.rrt import RRTConfig, path_length as rrt_length, rrt_connect

FIELDS = ["scene", "scale", "task", "solvable", "free_frac", "planner", "seed", "solved",
          "iterations", "budget", "time_s", "ref_length", "length_ratio"]
PLANNERS = (("rrt_connect", False, "sample"), ("angle_zone", True, "sample"),
            ("angle_zone_adaptive", True, "extension"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="L2")
    ap.add_argument("--params", type=Path, default=Path("results/missions_pb"),
                    help="missions directory whose params_<scene>.csv hold the calibrated margins")
    ap.add_argument("--data", type=Path, default=Path("data/replica_cad"))
    ap.add_argument("--detections", type=Path, default=Path("data/cache/detections"))
    ap.add_argument("--levels", type=Path, default=Path("configs/drift_levels.yaml"))
    ap.add_argument("--tasks", type=int, default=20)
    ap.add_argument("--rrt-seeds", type=int, default=2)
    ap.add_argument("--budget", type=int, default=1500)
    ap.add_argument("--scales", type=float, nargs="+", default=[0.0, 0.5, 1.0, 1.5])
    ap.add_argument("--part", default="0/1", help="i/n: process every n-th scene starting at i")
    ap.add_argument("--avoid", nargs="+", default=list(AVOID_CLASSES), help="avoid classes used in planning")
    ap.add_argument("--tag", default="", help="output suffix, e.g. _pb")
    args = ap.parse_args()

    levels = load_levels(args.levels)
    scenes = sorted(p.stem[len("params_"):] for p in args.params.glob("params_*.csv"))
    part, nparts = map(int, args.part.split("/"))
    out = []
    for name in scenes[part::nparts]:
        t0 = time.time()
        scene = load_scene(args.data / name)
        cls_ids = dict(zip(args.avoid, class_ids(scene.classes, args.avoid)))
        params, fallback = read_params(args.params / f"params_{name}.csv", args.level, cls_ids)
        joint = params["joint"]
        setup = prepare(scene, precompute_observations(
            scene, load_detections(args.detections / f"{name}.npz")), args.avoid)
        tasks = make_tasks(setup, args.tasks, seed=0)
        est = simulate(scene.poses, levels[args.level], np.random.default_rng(stable_seed(7, 0, name)))
        D_q = world_correction(est, scene.poses)[-1]
        pred = build_map(setup.obs, est, setup.spec, setup.n_classes, "det")
        res, shape = setup.spec.res, setup.spec.shape
        base = pred.free() & ~dilate(pred.occupied(), ROBOT_RADIUS, res)
        cells = [(to_cell(setup.spec, t.start_xz, D_q), to_cell(setup.spec, t.goal_xz, D_q)) for t in tasks]
        cells = [c for c in cells if c[0] is not None and c[1] is not None]

        for m in args.scales:
            scaled = {cid: None if p is None else ArmParams(p.lam, m * p.radius) for cid, p in joint.items()}
            fb = {cid: None if r is None else m * r for cid, r in fallback.items()}
            keep = keepout(pred, setup.avoid_ids, scaled, fb, res)
            if keep is None:
                continue
            trav = base & ~keep
            exact = plan_many(trav, [c[0] for c in cells], [disk(shape, c[1], 0.0, res) for c in cells], res)
            free_frac = float(trav.sum() / max(base.sum(), 1))
            for ti, ((s_rc, g_rc), ref) in enumerate(zip(cells, exact)):
                if ref is None:
                    out.append({"scene": name, "scale": m, "task": ti, "solvable": False, "free_frac": free_frac})
                    continue
                ref_len = path_length(ref, res)
                for planner, cone, widen in PLANNERS:
                    for seed in range(args.rrt_seeds):
                        cfg = RRTConfig(cone=cone, widen_on=widen, max_iter=args.budget)
                        t = time.time()
                        path, it = rrt_connect(trav, s_rc, g_rc, res, cfg,
                                               np.random.default_rng(stable_seed(seed, ti, name)))
                        out.append({"scene": name, "scale": m, "task": ti, "solvable": True,
                                    "free_frac": free_frac, "planner": planner, "seed": seed,
                                    "solved": path is not None, "iterations": it, "budget": args.budget,
                                    "time_s": time.time() - t, "ref_length": ref_len,
                                    "length_ratio": rrt_length(path) / ref_len if path is not None else np.nan})
        radii = ", ".join(f"{scene.classes[c]}={'abstain' if p is None else f'{p.radius:.2f}'}" for c, p in joint.items())
        print(f"{name}: joint radii {radii}; {len(cells)} tasks, {time.time() - t0:.0f} s", flush=True)

    write_csv(Path(f"results/tables/planner_bridge{args.tag}_{part}.csv"), out, FIELDS)
