"""Pose-independent frame observations and fast map building under arbitrary poses.

For every frame we back-project depth once and keep, per 3D point (camera frame), its
ground-truth class and the detector's class and score. A map for any trajectory (GT poses,
drifted poses, or GT rigidly moved into the planner frame) is then a transform + binning.

Predicted maps fuse detector evidence by averaging (not naive Bayes, which is known to make
fused semantic maps overconfident): p_k(cell) = sum of class-k scores / number of observations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from s2m.data import Scene, backproject
from s2m.grid import FLOOR_CLASSES, GridSpec
from s2m.perception import Detections, pixel_labels

FLOOR_MAX_Y = 0.05  # points below this height are floor
OBSTACLE_BAND = (0.1, 1.8)  # points in this height range are obstacles for a ground robot


@dataclass
class FrameObs:
    frame: int
    pts: np.ndarray  # (M, 3) float32, camera frame
    gt_label: np.ndarray  # (M,) int16
    det_label: np.ndarray  # (M,) int16, -1 = no detection
    det_score: np.ndarray  # (M,) float16


def precompute_observations(scene: Scene, detections: dict[int, Detections], stride: int = 4,
                            frames=None) -> list[FrameObs]:
    frames = scene.frame_ids() if frames is None else frames
    out = []
    for i in frames:
        depth = scene.depth(i)
        pts, pix = backproject(depth, scene.intrinsics, stride)
        det_lab, det_score = pixel_labels(detections[i], depth)
        out.append(FrameObs(i, pts.astype(np.float32),
                            scene.semantic(i).ravel()[pix].astype(np.int16),
                            det_lab.ravel()[pix].astype(np.int16),
                            det_score.ravel()[pix].astype(np.float16)))
    return out


@dataclass
class Map:
    """Top-down map in one frame of reference.

    n_obs:  (H, W) obstacle-band points per cell
    floor:  (H, W) floor points per cell
    class_mass: (H, W, K) per-class evidence (GT: point counts; predicted: summed scores)
    """
    spec: GridSpec
    n_obs: np.ndarray
    floor: np.ndarray
    class_mass: np.ndarray

    def occupied(self, min_points: int = 3) -> np.ndarray:
        return self.n_obs >= min_points

    def free(self) -> np.ndarray:
        return (self.floor > 0) & ~self.occupied()

    def prob(self, k: int) -> np.ndarray:
        """Fraction of a cell's observations supporting class k (score-weighted for predictions)."""
        return self.class_mass[:, :, k] / np.maximum(self.n_obs, 1)

    def region(self, k: int, lam: float) -> np.ndarray:
        """Cells the map assigns to class k at threshold lam (occupied and p_k >= lam)."""
        return self.occupied() & (self.prob(k) >= lam)

    def label(self) -> np.ndarray:
        lab = self.class_mass.argmax(axis=2)
        lab[~self.occupied() | (self.class_mass.max(axis=2) <= 0)] = -1
        return lab


def build_map(obs: list[FrameObs], poses: np.ndarray, spec: GridSpec, n_classes: int,
              source: str, floor_ids: np.ndarray | None = None) -> Map:
    """Fuse observations placed with `poses` (indexed by frame id) into a Map.

    source='gt'  : class evidence = GT label counts; floor = floor-class points at floor height
    source='det' : class evidence = detector scores; floor = any point at floor height
    """
    h, w = spec.shape
    n_obs = np.zeros(h * w, np.float64)
    floor = np.zeros(h * w, np.float64)
    mass = np.zeros(h * w * n_classes, np.float64)

    for o in obs:
        T = poses[o.frame]
        p = o.pts @ T[:3, :3].T.astype(np.float32) + T[:3, 3].astype(np.float32)
        r, c, inside = spec.to_cell(p[:, 0], p[:, 2])
        cell = r * w + c
        y = p[:, 1]

        if source == "gt":
            is_floor = inside & (y < FLOOR_MAX_Y) & np.isin(o.gt_label, floor_ids)
        else:
            is_floor = inside & (y < FLOOR_MAX_Y)
        floor += np.bincount(cell[is_floor], minlength=h * w)

        band = inside & (y > OBSTACLE_BAND[0]) & (y < OBSTACLE_BAND[1])
        n_obs += np.bincount(cell[band], minlength=h * w)

        if source == "gt":
            lab, wgt = o.gt_label[band].astype(np.int64), None
        else:
            has = o.det_label[band] >= 0
            lab = o.det_label[band][has].astype(np.int64)
            wgt = o.det_score[band][has].astype(np.float64)
            band = np.flatnonzero(band)[has]
        mass += np.bincount(cell[band] * n_classes + lab, weights=wgt, minlength=h * w * n_classes)

    return Map(spec, n_obs.reshape(h, w), floor.reshape(h, w), mass.reshape(h, w, n_classes))


def floor_class_ids(scene: Scene) -> np.ndarray:
    return np.array([k for k, name in scene.classes.items() if name in FLOOR_CLASSES])


def fixed_spec(scene: Scene, obs: list[FrameObs], res: float = 0.05, margin: float = 2.0) -> GridSpec:
    """One grid per scene, large enough to hold drifted maps (GT extent + margin)."""
    lo, hi = np.full(2, np.inf), np.full(2, -np.inf)
    for o in obs:
        T = scene.poses[o.frame]
        p = (o.pts @ T[:3, :3].T + T[:3, 3])[:, [0, 2]]
        lo, hi = np.minimum(lo, p.min(0)), np.maximum(hi, p.max(0))
    lo, hi = lo - margin, hi + margin
    shape = (int(np.ceil((hi[1] - lo[1]) / res)), int(np.ceil((hi[0] - lo[0]) / res)))
    return GridSpec(float(lo[0]), float(lo[1]), res, shape)
