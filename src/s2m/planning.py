"""Grid planning for reach-avoid missions on a predicted map.

Shortest paths on the 8-connected grid of traversable cells (Dijkstra via scipy.sparse.csgraph,
so the search runs in C). The planner only sees the predicted map; arms differ in how they
turn the map into keep-out zones around avoid-class regions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

ROBOT_RADIUS = 0.2  # metres (OSMa-Bench agent radius)
SAFETY_DISTANCE = 0.5  # metres to keep from avoid-class objects
GOAL_REACH = 0.6  # metres: reaching the goal = ending this close to the goal object

_NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def dilate(mask: np.ndarray, radius: float, res: float) -> np.ndarray:
    """Cells within `radius` metres of `mask` (exact Euclidean dilation)."""
    if radius <= 0 or not mask.any():
        return mask.copy()
    return ndimage.distance_transform_edt(~mask) * res <= radius


def _grid_graph(traversable: np.ndarray, res: float):
    """CSR graph over traversable cells (8-connected) and the cell <-> node index maps."""
    h, w = traversable.shape
    idx = -np.ones((h, w), np.int64)
    cells = np.argwhere(traversable)
    idx[cells[:, 0], cells[:, 1]] = np.arange(len(cells))
    rows, cols, wts = [], [], []
    for dr, dc in _NEIGHBOURS:
        r2, c2 = cells[:, 0] + dr, cells[:, 1] + dc
        ok = (r2 >= 0) & (r2 < h) & (c2 >= 0) & (c2 < w)
        ok[ok] &= traversable[r2[ok], c2[ok]]
        rows.append(idx[cells[ok, 0], cells[ok, 1]])
        cols.append(idx[r2[ok], c2[ok]])
        wts.append(np.full(ok.sum(), res * np.hypot(dr, dc)))
    graph = coo_matrix((np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(len(cells), len(cells))).tocsr()
    return graph, idx, cells


def plan_many(traversable: np.ndarray, starts: list[tuple[int, int]], goal_masks: list[np.ndarray],
              res: float) -> list[np.ndarray | None]:
    """Shortest path (cells, (n, 2)) from each start to its goal region; one graph, one search."""
    out: list[np.ndarray | None] = [None] * len(starts)
    ok = [i for i, s in enumerate(starts) if traversable[s] and (goal_masks[i] & traversable).any()]
    if not ok:
        return out
    graph, idx, cells = _grid_graph(traversable, res)
    srcs = np.array([idx[starts[i]] for i in ok])
    dist, pred = dijkstra(graph, indices=srcs, return_predecessors=True)
    for row, i in enumerate(ok):
        goal_ids = idx[goal_masks[i] & traversable]
        best = goal_ids[np.argmin(dist[row, goal_ids])]
        if not np.isfinite(dist[row, best]):
            continue
        path = [best]
        while path[-1] != srcs[row]:
            path.append(pred[row, path[-1]])
        out[i] = cells[np.array(path[::-1])]
    return out


def shortest_path(traversable: np.ndarray, start: tuple[int, int], goal_mask: np.ndarray,
                  res: float) -> np.ndarray | None:
    """Cells (n, 2) of the shortest path from start to any goal cell, or None."""
    return plan_many(traversable, [start], [goal_mask], res)[0]


def path_length(path: np.ndarray, res: float) -> float:
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum() * res)


@dataclass
class Outcome:
    planned: bool
    violation: bool = False  # came closer than SAFETY_DISTANCE to a true avoid object
    collision: bool = False  # came closer than ROBOT_RADIUS to a true obstacle
    reached: bool = False  # ended within `reach` of the true goal
    length: float = np.nan

    @property
    def success(self) -> bool:
        return self.planned and self.reached and not self.violation and not self.collision


def evaluate_path(path: np.ndarray | None, true_avoid_dist: np.ndarray, true_obst_dist: np.ndarray,
                  true_goal_dist: np.ndarray, res: float, reach: float = GOAL_REACH) -> Outcome:
    """Judge a planned path against the truth (distance fields in the planner frame)."""
    if path is None:
        return Outcome(planned=False)
    r, c = path[:, 0], path[:, 1]
    return Outcome(
        planned=True,
        violation=bool(true_avoid_dist[r, c].min() < SAFETY_DISTANCE - res),
        collision=bool(true_obst_dist[r, c].min() < ROBOT_RADIUS - res),
        reached=bool(true_goal_dist[r[-1], c[-1]] <= reach + res),
        length=path_length(path, res),
    )
