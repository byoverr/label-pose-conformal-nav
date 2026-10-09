"""Area of each avoid-class region (cells at LAM0, ground-truth poses) for every detector set, to measure what the
detector ensembles and the cascade cost in keep-out area. False area: region cells more than 20 cm from the true
objects of the class. Writes results/tables/region_areas.csv (one row per scene and detector set) and prints the
totals over the 21 evaluation scenes relative to YOLO-World v2-s.

Example: python scripts/region_areas.py
"""

import numpy as np
from pathlib import Path

from s2m.analysis import LAM0
from s2m.conformal import distance_to
from s2m.data import load_scene
from s2m.experiment import prepare
from s2m.io import DEV_SCENES, write_csv
from s2m.mapping import build_map, fuse_maps, precompute_observations
from s2m.perception import load_detections

CLASSES = ("indoor_plant", "bike", "tv_stand")
DETECTORS = {"v2-s": "data/cache/detections", "v2-x": "data/cache/detections_x", "yoloe": "data/cache/detections_yoloe"}

if __name__ == "__main__":
    scenes = [s for s in open("configs/scenes_replicacad.txt").read().split() if s not in DEV_SCENES]
    rows = []
    for name in scenes:
        scene = load_scene(Path("data/replica_cad") / name)
        obs = {k: precompute_observations(scene, load_detections(Path(d) / f"{name}.npz")) for k, d in DETECTORS.items()}
        cascade = precompute_observations(scene, load_detections(Path("data/cache/detections_cascade") / f"{name}.npz"))
        setup = prepare(scene, obs["v2-s"], list(CLASSES))
        build = lambda o: build_map(o, scene.poses, setup.spec, setup.n_classes, "det")
        maps = {k: build(o) for k, o in obs.items()}
        sets = {"v2-s": maps["v2-s"], "ens_max": fuse_maps(list(maps.values()), "max"),
                "ens_mean": fuse_maps(list(maps.values()), "mean"), "ens_vote2": fuse_maps(list(maps.values()), "vote2"),
                "cascade": build(cascade)}
        for k in setup.avoid_ids:
            cname = scene.classes[k]
            truth = np.zeros(setup.spec.shape, bool)
            for e in setup.entities:
                if e.cls == k:
                    truth |= e.mask(setup.spec.shape)
            near = distance_to(truth, setup.spec.res) <= 0.2 if truth.any() else np.zeros_like(truth)
            for tag, m in sets.items():
                region = m.region(k, LAM0)
                rows.append({"scene": name, "class": cname, "detections": tag, "area": int(region.sum()),
                             "false_area": int((region & ~near).sum())})
        print(name, flush=True)
    write_csv("results/tables/region_areas.csv", rows)
    for cname in CLASSES:
        base = sum(r["area"] for r in rows if r["class"] == cname and r["detections"] == "v2-s")
        print(cname, " ".join(f"{t} {sum(r['area'] for r in rows if r['class'] == cname and r['detections'] == t) / base:.2f}"
                              for t in ("ens_max", "ens_mean", "ens_vote2", "cascade")))
