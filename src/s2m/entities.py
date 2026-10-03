"""Ground-truth entities (object instances) of selected classes in a map."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from s2m.mapping import Map

# Classes a ground robot must keep away from (fragile or expensive, may tip over).
# Chosen among classes present in every ReplicaCAD scene. "stair" was dropped after looking at
# the dev scene apt_0 only: it is a 5 m^2 region, not an object, and a box detector does not
# localise it (median class score 0.03), so it would dominate every score (see docs/journal.md).
AVOID_CLASSES = ("indoor_plant", "tv_stand", "bike")
GOAL_CLASSES = ("sofa", "refrigerator", "table", "chair", "cabinet")


@dataclass
class Entity:
    cls: int
    cells: np.ndarray  # (n, 2) int (row, col) footprint cells in the map's grid

    def mask(self, shape: tuple[int, int]) -> np.ndarray:
        m = np.zeros(shape, bool)
        m[self.cells[:, 0], self.cells[:, 1]] = True
        return m


def extract_entities(truth: Map, class_ids, min_points: int = 2, min_cells: int = 8,
                     merge_cells: int = 2) -> list[Entity]:
    """Connected components of each class's footprint.

    Occupied cells with at least `min_points` GT points of the class form the footprint; components
    closer than `merge_cells` are merged (one object seen in pieces); tiny ones are dropped.
    """
    out = []
    for k in class_ids:
        m = (truth.class_mass[:, :, k] >= min_points) & truth.occupied()
        if not m.any():
            continue
        lab, n = ndimage.label(ndimage.binary_dilation(m, iterations=merge_cells), structure=np.ones((3, 3)))
        for j in range(1, n + 1):
            cells = np.argwhere(m & (lab == j))
            if len(cells) >= min_cells:
                out.append(Entity(int(k), cells))
    return out


def class_ids(classes: dict[int, str], names) -> list[int]:
    name2id = {v: k for k, v in classes.items()}
    return [name2id[n] for n in names]
