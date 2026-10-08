"""Extra miss distances for region variants, on the same drift realizations as scripts/compute_scores.py.

grow<cm>: the class region at LAM0 grown into adjacent occupied cells by up to <cm> (geodesic dilation
inside the occupied set), to recover truncated object edges the detector did not label. Also records the
region areas, so the price in keep-out area is visible.

Writes results/scores_extra/<scene>.csv (resumable), keyed by scene, level, seed.

Example: python scripts/extra_scores.py data/replica_cad/*
"""

import argparse
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

from s2m.analysis import LAM0
from s2m.conformal import miss_distance
from s2m.data import load_scene
from s2m.drift import simulate, world_correction
from s2m.experiment import R_MAX, load_levels, prepare, stable_seed, transform_entities
from s2m.io import write_csv
from s2m.mapping import build_map, precompute_observations
from s2m.perception import load_detections

GROW_CM = (15, 30)


def grow(region: np.ndarray, occupied: np.ndarray, steps: int) -> np.ndarray:
    """Region grown by `steps` cells into 8-connected occupied cells."""
    out = region.copy()
    for _ in range(steps):
        out = (ndimage.binary_dilation(out, structure=np.ones((3, 3), bool)) & occupied) | region
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dirs", type=Path, nargs="+")
    ap.add_argument("--levels", type=Path, default=Path("configs/drift_levels.yaml"))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--detections", type=Path, default=Path("data/cache/detections"))
    ap.add_argument("--out", type=Path, default=Path("results/scores_extra"))
    args = ap.parse_args()

    levels = load_levels(args.levels)
    args.out.mkdir(parents=True, exist_ok=True)
    for scene_dir in args.scene_dirs:
        out = args.out / f"{scene_dir.name}.csv"
        if out.exists():
            continue
        t = time.time()
        scene = load_scene(scene_dir)
        setup = prepare(scene, precompute_observations(scene, load_detections(args.detections / f"{scene.name}.npz")))
        rows = []
        for level, model in levels.items():
            for seed in range(1 if level == "L0" else args.seeds):
                est = simulate(scene.poses, model, np.random.default_rng(stable_seed(0, seed, scene.name)))
                ents = transform_entities(setup.entities, setup.spec, world_correction(est, scene.poses)[-1])
                pred = build_map(setup.obs, est, setup.spec, setup.n_classes, "det")
                occ = pred.occupied()
                row = {"scene": scene.name, "level": level, "seed": seed}
                for k in setup.avoid_ids:
                    name = scene.classes[k]
                    ek = [e for e in ents if e.cls == k]
                    base = pred.region(k, LAM0)
                    row[f"area_{name}"] = int(base.sum())
                    for cm in GROW_CM:
                        g = grow(base, occ, round(cm / 100 / setup.spec.res))
                        row[f"area_grow{cm}_{name}"] = int(g.sum())
                        row[f"miss_grow{cm}_{name}"] = miss_distance({k: g}, ek, setup.spec.res, R_MAX)
                rows.append(row)
        write_csv(out, rows)
        print(f"{scene.name}: {len(rows)} rows, {time.time() - t:.0f} s", flush=True)
