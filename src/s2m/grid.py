"""Top-down 2D semantic grid built from posed RGB-D frames.

World frame is Habitat's: y up, floor at y = 0, so the grid lives in the x-z plane.
The same accumulator is used for the ground-truth map (GT semantic labels, GT poses)
and later for predicted maps (detector labels, drifted poses).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from s2m.data import Scene, backproject, transform

FLOOR_CLASSES = ("floor", "rug", "mat")  # walkable surfaces in ReplicaCAD labels


@dataclass
class GridSpec:
    x0: float  # world x of the grid's first column edge
    z0: float  # world z of the grid's first row edge
    res: float  # metres per cell
    shape: tuple[int, int]  # (rows along z, cols along x)

    def to_cell(self, x: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r = np.floor((z - self.z0) / self.res).astype(np.int64)
        c = np.floor((x - self.x0) / self.res).astype(np.int64)
        inside = (r >= 0) & (r < self.shape[0]) & (c >= 0) & (c < self.shape[1])
        return r, c, inside

    def cell_centers(self) -> tuple[np.ndarray, np.ndarray]:
        xs = self.x0 + (np.arange(self.shape[1]) + 0.5) * self.res
        zs = self.z0 + (np.arange(self.shape[0]) + 0.5) * self.res
        return xs, zs


@dataclass
class SemanticGrid:
    spec: GridSpec
    n_classes: int
    obstacle_counts: np.ndarray = field(init=False)  # (H, W, K) labelled points in the obstacle band
    floor_counts: np.ndarray = field(init=False)  # (H, W) walkable-floor points

    def __post_init__(self):
        h, w = self.spec.shape
        self.obstacle_counts = np.zeros((h, w, self.n_classes), dtype=np.int32)
        self.floor_counts = np.zeros((h, w), dtype=np.int32)

    def add(self, pts_world: np.ndarray, labels: np.ndarray, floor_ids: np.ndarray,
            band: tuple[float, float] = (0.1, 1.8), floor_max_y: float = 0.05) -> None:
        y = pts_world[:, 1]
        r, c, inside = self.spec.to_cell(pts_world[:, 0], pts_world[:, 2])

        is_floor = inside & (y < floor_max_y) & np.isin(labels, floor_ids)
        np.add.at(self.floor_counts, (r[is_floor], c[is_floor]), 1)

        is_obst = inside & (y > band[0]) & (y < band[1])
        np.add.at(self.obstacle_counts, (r[is_obst], c[is_obst], labels[is_obst]), 1)

    def occupied(self, min_points: int = 3) -> np.ndarray:
        return self.obstacle_counts.sum(axis=2) >= min_points

    def free(self, min_points: int = 1) -> np.ndarray:
        return (self.floor_counts >= min_points) & ~self.occupied()

    def label(self) -> np.ndarray:
        """Majority class per occupied cell, -1 elsewhere."""
        lab = self.obstacle_counts.argmax(axis=2)
        lab[~self.occupied()] = -1
        return lab

    def class_mask(self, k: int, min_points: int = 2) -> np.ndarray:
        """Cells where class k was observed in the obstacle band (an entity's footprint)."""
        return self.obstacle_counts[:, :, k] >= min_points


def scene_extent(scene: Scene, frame_ids, poses=None, stride: int = 8, margin: float = 0.5):
    """Bounding box (x_min, x_max, z_min, z_max) of all back-projected points."""
    poses = scene.poses if poses is None else poses
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for i in frame_ids:
        pts, _ = backproject(scene.depth(i), scene.intrinsics, stride)
        w = transform(pts, poses[i])
        lo, hi = np.minimum(lo, w.min(0)), np.maximum(hi, w.max(0))
    return lo[0] - margin, hi[0] + margin, lo[2] - margin, hi[2] + margin


def build_gt_grid(scene: Scene, frame_ids=None, res: float = 0.05, stride: int = 2,
                  poses=None, extent=None) -> SemanticGrid:
    """Ground-truth grid: GT semantic labels fused with (by default) GT poses."""
    frame_ids = scene.frame_ids() if frame_ids is None else frame_ids
    poses = scene.poses if poses is None else poses
    if extent is None:
        extent = scene_extent(scene, frame_ids, poses)
    x_min, x_max, z_min, z_max = extent
    shape = (int(np.ceil((z_max - z_min) / res)), int(np.ceil((x_max - x_min) / res)))
    grid = SemanticGrid(GridSpec(x_min, z_min, res, shape), n_classes=max(scene.classes) + 1)

    floor_ids = np.array([k for k, name in scene.classes.items() if name in FLOOR_CLASSES])
    for i in frame_ids:
        pts, pix = backproject(scene.depth(i), scene.intrinsics, stride)
        labels = scene.semantic(i).ravel()[pix]
        grid.add(transform(pts, poses[i]), labels, floor_ids)
    return grid
