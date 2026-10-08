"""Choose the cascade thresholds on the development scene apt_0 only (ground-truth poses).

For every candidate (tau, tau_relabel) the cascade detections of apt_0 are mapped, and for each avoid
class the label miss distance (what the guarantee pays for) and the false region area (region cells
more than 20 cm from the true objects of the class) are recorded, next to the single detector v2-s and
the plain union of the three detectors. Writes results/tables/cascade_dev.csv.

Example: python scripts/select_cascade.py
"""

import numpy as np
from pathlib import Path

from s2m.analysis import LAM0
from s2m.cascade import cascade, load_verified, verifier_prompts
from s2m.conformal import distance_to, miss_distance
from s2m.data import load_scene
from s2m.experiment import R_MAX, prepare
from s2m.io import write_csv
from s2m.mapping import build_map, precompute_observations
from s2m.perception import load_detections

DETECTORS = ("data/cache/detections", "data/cache/detections_x", "data/cache/detections_yoloe")
CLASSES = ("indoor_plant", "tv_stand", "bike")
TAUS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3)
RELABEL = (None, 0.5, 0.8)

if __name__ == "__main__":
    scene = load_scene(Path("data/replica_cad/apt_0"))
    sets = [load_detections(Path(d) / "apt_0.npz") for d in DETECTORS]
    ver = load_verified(Path("data/cache/clip/apt_0.npz"))
    ids, _ = verifier_prompts(scene.classes)
    variants = [("v2-s", None, None, sets[0])] + [
        (f"cascade tau={t} relabel={r}", t, r, cascade(sets, ver, ids, t, r)) for r in RELABEL for t in TAUS]
    rows = []
    setup = None
    for name, t, r, dets in variants:
        obs = precompute_observations(scene, dets)
        if setup is None:
            setup = prepare(scene, obs, list(CLASSES))
        pred = build_map(obs, scene.poses, setup.spec, setup.n_classes, "det")
        row = {"variant": name, "tau": "" if t is None else t, "tau_relabel": "" if r is None else r,
               "detections": sum(len(d.scores) for d in dets.values())}
        for k in setup.avoid_ids:
            cname = scene.classes[k]
            ek = [e for e in setup.entities if e.cls == k]
            region = pred.region(k, LAM0)
            truth = np.zeros(setup.spec.shape, bool)
            for e in ek:
                truth |= e.mask(setup.spec.shape)
            near = distance_to(truth, setup.spec.res) <= 0.2 if truth.any() else np.zeros_like(truth)
            row[f"miss_{cname}"] = miss_distance({k: region}, ek, setup.spec.res, R_MAX)
            row[f"area_{cname}"] = int(region.sum())
            row[f"false_area_{cname}"] = int((region & ~near).sum())
        rows.append(row)
        print("  ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
    write_csv("results/tables/cascade_dev.csv", rows)
