"""Risk as an output: plan with a grid of margins and keep every path's clearance to each avoid-class region.

For a test scene the calibration set is the other 20 scenes (one random drift realization each, as in
scripts/run_missions.py); their miss distances are saved next to the paths, so scripts/analyze_risk.py can
assign every path a conformal risk bound without re-planning. Writes results/risk_<kind>/<scene>.csv and
results/risk_<kind>/cal_<scene>.csv (resumable).

With --exec-draws N every distinct path is also executed N times under continued drift of the same level
(s2m.missions.execution_outcomes) and the output goes to results/exec_<kind>/ (scripts/analyze_execution.py).

Example: python scripts/run_risk.py --kind pass_by --part 0/2
"""

import argparse
import time
from pathlib import Path

import numpy as np

from s2m.analysis import LAM0, load_scores, miss_at
from s2m.data import load_scene
from s2m.drift import simulate
from s2m.entities import class_ids
from s2m.experiment import load_levels, prepare, stable_seed
from s2m.io import DEV_SCENES, write_csv
from s2m.mapping import precompute_observations
from s2m.odometry import vo_poses
from s2m.missions import make_tasks, run_risk_realization
from s2m.perception import load_detections

AVOID = ("indoor_plant", "bike")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/replica_cad"))
    ap.add_argument("--scores", type=Path, default=Path("results/scores"))
    ap.add_argument("--detections", type=Path, default=Path("data/cache/detections"))
    ap.add_argument("--levels", type=Path, default=Path("configs/drift_levels.yaml"))
    ap.add_argument("--only", nargs="+", default=["L0", "L2", "L3", "L4"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--tasks", type=int, default=20)
    ap.add_argument("--kind", default="pass_by", choices=["pass_by", "random"])
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--exec-draws", type=int, default=0)
    ap.add_argument("--slam", type=Path, default=None, help="SLAM pose cache: one level SLAM2 instead of synthetic drift")
    ap.add_argument("--detections-extra", type=Path, nargs="*", default=[], help="more detection caches, fused (max)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    out_dir = args.out or Path(f"results/{'exec' if args.exec_draws else 'risk'}_{'pb' if args.kind == 'pass_by' else 'rand'}")
    out_dir.mkdir(parents=True, exist_ok=True)
    levels = {lv: m for lv, m in load_levels(args.levels).items() if lv in args.only}
    if args.slam is not None:
        if args.exec_draws:
            raise SystemExit("--exec-draws needs a drift model; it cannot be combined with --slam")
        levels = {"SLAM2": None}
    rows = load_scores(args.scores, exclude=DEV_SCENES)
    by = {}
    for r in rows:
        by.setdefault((r["scene"], r["level"]), []).append(r)
    scenes = sorted({r["scene"] for r in rows})
    part, nparts = map(int, args.part.split("/"))
    for name in scenes[part::nparts]:
        out = out_dir / f"{name}.csv"
        if out.exists():
            continue
        t = time.time()
        rng = np.random.default_rng(stable_seed(0, name))
        cal_rows = []
        for lv in levels:
            for s in scenes:
                if s == name:
                    continue
                r = by[(s, lv)][rng.integers(len(by[(s, lv)]))]
                cal_rows.append({"level": lv, "scene": s, **{f"miss_{c}": miss_at(r, c, LAM0) for c in AVOID}})
        write_csv(out_dir / f"cal_{name}.csv", cal_rows)

        scene = load_scene(args.data / name)
        setup = prepare(scene, precompute_observations(scene, load_detections(args.detections / f"{name}.npz")), list(AVOID))
        setup.extra_obs = [precompute_observations(scene, load_detections(d / f"{name}.npz")) for d in args.detections_extra]
        tasks = make_tasks(setup, args.tasks, seed=0, kind=args.kind)
        res = []
        for lv, model in levels.items():
            for seed in range(1 if lv in ("L0", "SLAM2") else args.seeds):
                est = (vo_poses(args.slam / f"{name}_s2.npz", scene.poses) if lv == "SLAM2" else
                       simulate(scene.poses, model, np.random.default_rng(stable_seed(7, seed, name))))
                for r in run_risk_realization(setup, tasks, est, exec_model=model, exec_draws=args.exec_draws,
                                              rng=np.random.default_rng(stable_seed(11, seed, name))):
                    res.append({"scene": name, "level": lv, "seed": seed, **r})
        fields = ["scene", "level", "seed", "task", "margin", "planned", "violation", "collision", "length"] + \
                 [f"clearance_{c}" for c in AVOID] + \
                 (["exec_unsafe", "exec_collision", "exec_fail", "exec_u", "exec_dev"] if args.exec_draws else [])
        write_csv(out, res, fields)
        print(f"{name}: {len(tasks)} tasks, {len(res)} rows, {time.time() - t:.0f} s", flush=True)
