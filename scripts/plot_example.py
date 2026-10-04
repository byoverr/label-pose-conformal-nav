"""Qualitative example: one pass-by task planned with the uncalibrated map and with the joint arm.

Reproduces one mission realization (same seeds as scripts/run_missions.py) and draws, in the
planner frame: true obstacles, true avoid-class footprints, the keep-out zone of each arm and
both paths. Writes results/figures/example_<scene>.png.

Example: python scripts/plot_example.py v3_sc0_staging_20 --seed 1 --task 1
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from s2m.analysis import load_scores, read_params
from s2m.conformal import distance_to
from s2m.data import load_scene
from s2m.drift import simulate, world_correction
from s2m.entities import class_ids
from s2m.experiment import load_levels, prepare, stable_seed, transform_entities
from s2m.mapping import build_map, floor_class_ids, precompute_observations
from s2m.io import DEV_SCENES, level_labels
from s2m.missions import GOAL_TOL, disk, keepout, make_tasks, to_cell
from s2m.perception import load_detections
from s2m.planning import ROBOT_RADIUS, SAFETY_DISTANCE, dilate, path_length, plan_many
from s2m.viz import ARM_STYLE, setup

AVOID = ["indoor_plant", "bike"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--level", default="L2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--paper", action="store_true", help="compact version for a one-column figure (no suptitle)")
    ap.add_argument("--lang", default="en", choices=["en", "ru"], help="ru: compact figure for the Russian report")
    args = ap.parse_args()

    scene = load_scene(Path("data/replica_cad") / args.scene)
    s = prepare(scene, precompute_observations(
        scene, load_detections(Path("data/cache/detections") / f"{args.scene}.npz")), AVOID)
    task = make_tasks(s, 20, seed=0)[args.task]
    est = simulate(scene.poses, load_levels("configs/drift_levels.yaml")[args.level],
                   np.random.default_rng(stable_seed(7, args.seed, args.scene)))
    D_q = world_correction(est, scene.poses)[-1]
    res, shape = s.spec.res, s.spec.shape
    cid = dict(zip(AVOID, class_ids(scene.classes, AVOID)))
    params, fallback = read_params(Path(f"results/missions_pb/params_{args.scene}.csv"), args.level, cid)

    pred = build_map(s.obs, est, s.spec, s.n_classes, "det")
    truth = build_map(s.obs, D_q[None] @ scene.poses, s.spec, s.n_classes, "gt", floor_class_ids(scene))
    avoid_true = np.zeros(shape, bool)
    for e in transform_entities(s.entities, s.spec, D_q):
        avoid_true |= e.mask(shape)
    start, goal = to_cell(s.spec, task.start_xz, D_q), to_cell(s.spec, task.goal_xz, D_q)
    base = pred.free() & ~dilate(pred.occupied(), ROBOT_RADIUS, res)

    zones, paths = {}, {}
    for arm in ("uncalibrated", "joint"):
        zones[arm] = keepout(pred, s.avoid_ids, params[arm], fallback, res)
        paths[arm] = plan_many(base & ~zones[arm], [start], [disk(shape, goal, GOAL_TOL, res)], res)[0]

    setup()
    img = np.zeros(shape, int)  # 0 unknown, 1 free, 2 obstacle, 3 true avoid footprint
    img[truth.free()] = 1
    img[truth.occupied()] = 2
    img[avoid_true] = 3
    cmap = ListedColormap(["#ffffff", "#ecebe7", "#9a9893", "#e34948"])
    rows, cols = np.nonzero(img > 0)
    r0, r1, c0, c1 = rows.min() - 5, rows.max() + 5, cols.min() - 5, cols.max() + 5

    ru = args.lang == "ru"
    compact = args.paper or ru
    title = {"uncalibrated": "карта без калибровки", "joint": "совместный запас"} if ru else \
        {arm: ARM_STYLE[arm][0] for arm in ("uncalibrated", "joint")}
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 4.6) if compact else (10, 5.2), sharey=True)
    d_true = distance_to(avoid_true, res)
    for ax, arm in zip(axes, ("uncalibrated", "joint")):
        ax.imshow(img, cmap=cmap, origin="lower", interpolation="nearest", vmin=0, vmax=3)
        ax.contour(zones[arm].astype(float), levels=[0.5], colors=[ARM_STYLE[arm][1]], linewidths=1.2)
        path = paths[arm]
        if path is not None:
            ax.plot(path[:, 1], path[:, 0], color=ARM_STYLE[arm][1], lw=2.2)
            close = d_true[path[:, 0], path[:, 1]] < SAFETY_DISTANCE
            ax.scatter(path[close, 1], path[close, 0], color="#0b0b0b", s=10, zorder=5,
                       label="ближе 0,5 м к настоящему объекту" if ru else "closer than 0.5 m to a true plant/bike")
            length = f"{path_length(path, res):.1f}"
            if ru:
                verdict = "опасно" if close.any() else "безопасно"
                ax.set_title(f"{title[arm]}\nпуть {length.replace('.', ',')} м, {verdict}")
            else:
                verdict = "VIOLATION" if close.any() else "safe"
                ax.set_title(f"{title[arm]}\npath {length} m, {verdict}")
        else:
            ax.set_title(f"{title[arm]}\n{'пути нет' if ru else 'no path'}")
        ax.plot(start[1], start[0], "o", color="#0b0b0b", ms=7)
        ax.plot(goal[1], goal[0], "*", color="#0b0b0b", ms=12)
        ax.set_xlim(c0, c1)
        ax.set_ylim(r0, r1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="lower left", fontsize=7)
    if not compact:
        ate = level_labels([r for r in load_scores("results/scores", exclude=DEV_SCENES) if r["scene"] == args.scene])
        fig.suptitle(f"{args.scene}, drift {args.level} (ATE {ate[args.level]}): red = true plant/bike footprint, "
                     f"grey = true obstacles, line = keep-out zone of the arm (planner frame)", fontsize=9)
    fig.tight_layout()
    suffix = "_ru" if ru else "_paper" if args.paper else ""
    out = Path(f"results/figures/{'ru/' if ru else ''}example_{args.scene}{suffix}.png")
    fig.savefig(out)
    print(out)
