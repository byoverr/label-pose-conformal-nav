"""Per-scene score computation for every drift level and realization.

For one realization of the drifted trajectory T_hat_k = D_k T_k we build, all in the planner's
frame (the drifted world as believed at query time q = last frame):
  pred  : detector map placed with drifted poses (what the robot actually has)
  gtlab : GT-label map placed with drifted poses (isolates the effect of pose error)
  truth : true entity footprints moved by D_q (where the obstacles really are, as seen
          from the robot's believed pose)
and record the scores needed by every calibration arm. Calibration itself happens later,
over random scene splits, in the analysis step.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field

import numpy as np
import yaml

from s2m.conformal import label_score, miss_distance
from s2m.data import Scene
from s2m.drift import DriftModel, ate, endpoint_drift, simulate, world_correction
from s2m.entities import AVOID_CLASSES, Entity, class_ids, extract_entities
from s2m.grid import GridSpec
from s2m.mapping import FrameObs, Map, build_map, fixed_spec, floor_class_ids, fuse_maps

LAMBDAS = np.round(np.arange(0.0, 0.5001, 0.01), 2)  # map thresholds evaluated for the pred map
R_MAX = 3.0  # metres; a larger miss means the class is effectively missing from the map


def load_levels(path) -> dict[str, DriftModel]:
    cfg = yaml.safe_load(open(path))["levels"]
    return {name: DriftModel(**params) for name, params in cfg.items()}


def transform_entities(entities: list[Entity], spec: GridSpec, D: np.ndarray) -> list[Entity]:
    """Move entity footprints by the planar rigid transform D (4x4) and re-rasterize.

    Cells pushed off the grid are kept: the object is still there, the map just does not reach it.
    """
    out = []
    for e in entities:
        x = spec.x0 + (e.cells[:, 1] + 0.5) * spec.res
        z = spec.z0 + (e.cells[:, 0] + 0.5) * spec.res
        p = np.stack([x, np.zeros_like(x), z], 1) @ D[:3, :3].T + D[:3, 3]
        r, c, _ = spec.to_cell(p[:, 0], p[:, 2])
        out.append(Entity(e.cls, np.unique(np.stack([r, c], 1), axis=0)))
    return out


@dataclass
class SceneSetup:
    scene: Scene
    obs: list[FrameObs]
    spec: GridSpec
    n_classes: int
    avoid_ids: list[int]
    entities: list[Entity]  # avoid-class entities in the GT world frame
    truth: Map  # GT map in the GT world frame
    extra_obs: list[list[FrameObs]] = field(default_factory=list)  # other detectors on the same points
    fuse: str = "max"  # how detector maps are combined (s2m.mapping.fuse_maps)


def predicted_map(setup: SceneSetup, poses: np.ndarray) -> Map:
    """The detector map placed with `poses`; several detectors are fused cell by cell."""
    maps = [build_map(o, poses, setup.spec, setup.n_classes, "det") for o in [setup.obs, *setup.extra_obs]]
    return fuse_maps(maps, setup.fuse)


def prepare(scene: Scene, obs: list[FrameObs], avoid_names=AVOID_CLASSES) -> SceneSetup:
    spec = fixed_spec(scene, obs)
    k = max(scene.classes) + 1
    truth = build_map(obs, scene.poses, spec, k, "gt", floor_class_ids(scene))
    avoid = class_ids(scene.classes, avoid_names)
    return SceneSetup(scene, obs, spec, k, avoid, extract_entities(truth, avoid), truth)


def realization_scores(setup: SceneSetup, est_poses: np.ndarray) -> dict:
    s = setup
    gt_poses = s.scene.poses
    D = world_correction(est_poses, gt_poses)
    ents = transform_entities(s.entities, s.spec, D[-1])

    pred = predicted_map(s, est_poses)
    gtlab = build_map(s.obs, est_poses, s.spec, s.n_classes, "gt", floor_class_ids(s.scene))

    row = {"ate": ate(est_poses, gt_poses), "endpoint_drift": endpoint_drift(est_poses, gt_poses)}
    # Per avoid class (a scene without that class scores 0: nothing to miss).
    for k in s.avoid_ids:
        name = s.scene.classes[k]
        ek = [e for e in ents if e.cls == k]
        row[f"n_ent_{name}"] = len(ek)
        row[f"off_grid_{name}"] = sum(not e.inside(s.spec.shape).all() for e in ek)
        row[f"label_score_{name}"] = label_score(pred, ek)
        # same rule as the true footprints (extract_entities): occupied cells with >= 2 points of k
        gt_region = gtlab.occupied() & (gtlab.class_mass[:, :, k] >= 2)
        row[f"miss_gtlab_{name}"] = miss_distance({k: gt_region}, ek, s.spec.res, R_MAX)
        for lam in LAMBDAS:
            row[f"miss_pred_{name}_{lam:.2f}"] = miss_distance({k: pred.region(k, lam)}, ek, s.spec.res, R_MAX)
    return row


def stable_seed(*parts) -> list[int]:
    """Seed sequence independent of Python's per-process string hashing."""
    return [p if isinstance(p, int) else zlib.crc32(str(p).encode()) for p in parts]


def scene_rows(setup: SceneSetup, levels: dict[str, DriftModel], seeds: int, base_seed: int = 0):
    gt = setup.scene.poses
    for level, model in levels.items():
        n = 1 if level == "L0" else seeds  # L0 is deterministic
        for seed in range(n):
            rng = np.random.default_rng(stable_seed(base_seed, seed, setup.scene.name))
            est = simulate(gt, model, rng)
            yield {"scene": setup.scene.name, "level": level, "seed": seed, **realization_scores(setup, est)}
