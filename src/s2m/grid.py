"""Geometry of the top-down grid: world (x, z) <-> cell (row, col).

World frame is Habitat's: y up, floor at y = 0, so the grid lives in the x-z plane. Maps
themselves (ground truth and predicted, under any poses) are built by s2m.mapping.build_map.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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
