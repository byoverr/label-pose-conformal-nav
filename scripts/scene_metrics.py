"""Map-quality metrics per scene at ground-truth poses (for H1: map metric vs mission outcome).

mIoU over classes with >= 20 GT cells among occupied cells (argmax labels), and the IoU of the
avoid classes only. Writes results/tables/scene_metrics.csv.
"""

import csv
from pathlib import Path

import numpy as np

from s2m.data import load_scene
from s2m.entities import AVOID_CLASSES, class_ids
from s2m.mapping import build_map, fixed_spec, floor_class_ids, precompute_observations
from s2m.perception import load_detections


def iou_table(gt_lab, pred_lab, occ, min_cells=20):
    out = {}
    for k in np.unique(gt_lab[occ]):
        if k < 0:
            continue
        g, p = occ & (gt_lab == k), occ & (pred_lab == k)
        if g.sum() >= min_cells:
            out[int(k)] = (g & p).sum() / max((g | p).sum(), 1)
    return out


if __name__ == "__main__":
    rows = []
    for det in sorted(Path("data/cache/detections").glob("*.npz")):
        scene = load_scene(Path("data/replica_cad") / det.stem)
        obs = precompute_observations(scene, load_detections(det))
        spec = fixed_spec(scene, obs)
        k = max(scene.classes) + 1
        gt = build_map(obs, scene.poses, spec, k, "gt", floor_class_ids(scene))
        pred = build_map(obs, scene.poses, spec, k, "det")
        occ = gt.occupied()
        ious = iou_table(gt.label(), pred.label(), occ)
        avoid = [ious[c] for c in class_ids(scene.classes, AVOID_CLASSES) if c in ious]
        rows.append({"scene": scene.name, "miou": float(np.mean(list(ious.values()))),
                     "n_classes": len(ious), "avoid_iou": float(np.mean(avoid)) if avoid else np.nan,
                     "labelled_occupied": float((pred.label()[occ] >= 0).mean())})
        print(rows[-1], flush=True)
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    with open("results/tables/scene_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
