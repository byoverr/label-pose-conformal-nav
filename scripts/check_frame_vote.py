"""Explicit multi-frame voting on the development scene apt_0 (ground-truth poses): a cell keeps class k
only if at least m distinct frames saw class k there, on top of the usual share threshold LAM0.
Prints label miss distance, region area and false area (cells more than 20 cm from the true objects) for
the single detector, the union of the three detectors and the cascade; writes results/tables/frame_vote_dev.csv.

Example: python scripts/check_frame_vote.py
"""
import numpy as np
from pathlib import Path
from s2m.analysis import LAM0
from s2m.cascade import cascade, load_verified, verifier_prompts
from s2m.conformal import distance_to, miss_distance
from s2m.data import load_scene
from s2m.experiment import R_MAX, prepare
from s2m.io import write_csv
from s2m.mapping import build_map, precompute_observations, OBSTACLE_BAND
from s2m.perception import load_detections
DET = ("data/cache/detections", "data/cache/detections_x", "data/cache/detections_yoloe")
scene = load_scene(Path("data/replica_cad/apt_0"))
sets = [load_detections(Path(d) / "apt_0.npz") for d in DET]
ids = verifier_prompts(scene.classes)[0]
ver = load_verified(Path("data/cache/clip/apt_0.npz"))
rows = []
for name, dets in [("v2-s", sets[0]), ("union", cascade(sets, ver, ids, 0.0)), ("cascade0.2", cascade(sets, ver, ids, 0.2))]:
    obs = precompute_observations(scene, dets)
    setup = prepare(scene, obs, ["indoor_plant", "tv_stand", "bike"])
    spec = setup.spec; h, w = spec.shape
    pred = build_map(obs, scene.poses, spec, setup.n_classes, "det")
    frames = {k: np.zeros(h * w) for k in setup.avoid_ids}
    for o in obs:
        T = scene.poses[o.frame]
        p = o.pts @ T[:3, :3].T.astype(np.float32) + T[:3, 3].astype(np.float32)
        r, c, inside = spec.to_cell(p[:, 0], p[:, 2])
        band = inside & (p[:, 1] > OBSTACLE_BAND[0]) & (p[:, 1] < OBSTACLE_BAND[1])
        for k in setup.avoid_ids:
            sel = band & (o.det_label == k)
            cells = np.unique((r * w + c)[sel])
            frames[k][cells] += 1
    for m in (1, 2, 3, 5):
        out, row = [], {}
        for k in setup.avoid_ids:
            ek = [e for e in setup.entities if e.cls == k]
            reg = pred.region(k, LAM0) & (frames[k].reshape(h, w) >= m)
            truth = np.zeros(spec.shape, bool)
            for e in ek: truth |= e.mask(spec.shape)
            near = distance_to(truth, spec.res) <= 0.2
            mk = miss_distance({k: reg}, ek, spec.res, R_MAX)
            row.update({f"miss_{scene.classes[k]}": mk, f"area_{scene.classes[k]}": int(reg.sum()), f"false_area_{scene.classes[k]}": int((reg & ~near).sum())})
            out.append(f"{scene.classes[k][:5]} miss {mk:.2f} area {reg.sum():4d} false {(reg & ~near).sum():3d}")
        print(f"{name:11s} m={m}: " + " | ".join(out), flush=True)
        rows.append({"detections": name, "min_frames": m, **row})
write_csv("results/tables/frame_vote_dev.csv", rows)
