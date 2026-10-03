"""Bidirectional sampling-based planners on an occupancy grid.

RRT-Connect (Kuffner & LaValle) and its variant with the angle-limited dynamic sampling zone of
I. S. Dovgopolik's quasi-optimal bidirectional RRT: a sample q is accepted only if the angle
between (q - a) and (b - a) is below alpha, where a is the nearest node of the growing tree and
b the nearest node of the other tree to a, i.e. the sampling region is a cone pointing at the
other tree. alpha starts at alpha_d and grows by alpha_st every time a sample falls into an
obstacle, returning to alpha_d after a free sample. With probability 1 - p_cone a plain uniform
sample is used, which keeps the planner probabilistically complete.

Planners work in metric (x, y) = (col, row) * res coordinates on a boolean traversable grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RRTConfig:
    step: float = 0.3  # metres per extension
    max_iter: int = 2000
    goal_tol: float = 0.3  # metres; trees are connected when within one step
    cone: bool = False  # use the angle-limited sampling zone
    p_cone: float = 0.7  # probability of a cone sample (vs uniform) when cone=True
    alpha_d: float = np.deg2rad(8.0)
    alpha_st: float = np.deg2rad(2.0)
    # "sample": widen while cone samples fall into obstacles, reset after a free sample (as published);
    # "extension": widen after every blocked extension, shrink by one step after a successful one.
    widen_on: str = "sample"


class GridChecker:
    def __init__(self, traversable: np.ndarray, res: float):
        self.t, self.res = traversable, res
        self.h, self.w = traversable.shape

    def free(self, p: np.ndarray) -> bool:
        c, r = int(p[0] / self.res), int(p[1] / self.res)
        return 0 <= r < self.h and 0 <= c < self.w and bool(self.t[r, c])

    def segment_free(self, a: np.ndarray, b: np.ndarray) -> bool:
        n = max(2, int(np.ceil(np.linalg.norm(b - a) / (0.5 * self.res))) + 1)
        pts = a + np.linspace(0.0, 1.0, n)[:, None] * (b - a)
        c, r = (pts[:, 0] / self.res).astype(int), (pts[:, 1] / self.res).astype(int)
        ok = (r >= 0) & (r < self.h) & (c >= 0) & (c < self.w)
        return bool(ok.all() and self.t[r, c].all())


class _Tree:
    """Nodes in a growing preallocated array, so nearest-neighbour queries are one vector op."""

    def __init__(self, root, capacity: int = 1024):
        self._xy = np.empty((capacity, 2))
        self._xy[0] = root
        self.parent = [-1]
        self.n = 1

    @property
    def nodes(self) -> np.ndarray:
        return self._xy[: self.n]

    def nearest(self, p) -> int:
        d = self.nodes - p
        return int(np.argmin(np.einsum("ij,ij->i", d, d)))

    def add(self, p, parent) -> int:
        if self.n == len(self._xy):
            self._xy = np.concatenate([self._xy, np.empty_like(self._xy)])
        self._xy[self.n] = p
        self.parent.append(parent)
        self.n += 1
        return self.n - 1

    def path_to_root(self, i):
        out = []
        while i != -1:
            out.append(self._xy[i].copy())
            i = self.parent[i]
        return out


def _steer(a, b, step):
    d = b - a
    n = np.linalg.norm(d)
    return b.copy() if n <= step else a + d / n * step


def _extend(tree, q, step, chk):
    i = tree.nearest(q)
    new = _steer(tree.nodes[i], q, step)
    if chk.segment_free(tree.nodes[i], new):
        return tree.add(new, i)
    return None


def _connect(tree, q, step, chk):
    """Greedily extend `tree` towards q; returns the last node index and whether q was reached."""
    last = None
    while True:
        i = tree.nearest(q)
        new = _steer(tree.nodes[i], q, step)
        if not chk.segment_free(tree.nodes[i], new):
            return last, False
        last = tree.add(new, i)
        if np.linalg.norm(new - q) < 1e-9:
            return last, True


def _angle(u: np.ndarray, v: np.ndarray) -> float:
    cosang = u @ v / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-12)
    return float(np.arccos(np.clip(cosang, -1.0, 1.0)))


class _ConeSampler:
    """Angle-limited dynamic sampling zone; rejected samples are cheap and do not count as
    iterations (the budget counts extension attempts, which carry the collision checks)."""

    def __init__(self, cfg: RRTConfig, chk: GridChecker, lo, hi, rng):
        self.cfg, self.chk, self.lo, self.hi, self.rng = cfg, chk, lo, hi, rng
        self.alpha = cfg.alpha_d

    def sample(self, grow: _Tree, other: _Tree, tries: int = 100) -> np.ndarray:
        for _ in range(tries):
            q = self.rng.uniform(self.lo, self.hi)
            na = grow.nodes[grow.nearest(q)]
            nb = other.nodes[other.nearest(na)]
            if _angle(q - na, nb - na) <= self.alpha:
                if self.cfg.widen_on == "sample":
                    free = self.chk.free(q)
                    self.alpha = self.cfg.alpha_d if free else min(np.pi, self.alpha + self.cfg.alpha_st)
                return q
        return self.rng.uniform(self.lo, self.hi)  # cone too narrow here: fall back to uniform

    def feedback(self, extended: bool) -> None:
        """Extension outcome of a cone sample (used by widen_on='extension')."""
        if self.cfg.widen_on != "extension":
            return
        if extended:
            self.alpha = max(self.cfg.alpha_d, self.alpha - self.cfg.alpha_st)
        else:
            self.alpha = min(np.pi, self.alpha + self.cfg.alpha_st)


def rrt_connect(traversable: np.ndarray, start_rc, goal_rc, res: float, cfg: RRTConfig,
                rng: np.random.Generator):
    """Returns (path as (n, 2) metric points from start to goal, or None; iterations used)."""
    chk = GridChecker(traversable, res)
    to_xy = lambda rc: np.array([(rc[1] + 0.5) * res, (rc[0] + 0.5) * res])
    s, g = to_xy(start_rc), to_xy(goal_rc)
    if not (chk.free(s) and chk.free(g)):
        return None, 0
    start_tree, goal_tree = _Tree(s), _Tree(g)
    lo, hi = np.zeros(2), np.array([chk.w * res, chk.h * res])
    cone = _ConeSampler(cfg, chk, lo, hi, rng) if cfg.cone else None
    a, b = start_tree, goal_tree
    for it in range(1, cfg.max_iter + 1):
        from_cone = cone is not None and rng.random() < cfg.p_cone
        q = cone.sample(a, b) if from_cone else rng.uniform(lo, hi)
        new = _extend(a, q, cfg.step, chk)
        if from_cone:
            cone.feedback(new is not None)
        if new is not None:
            last, reached = _connect(b, a.nodes[new], cfg.step, chk)
            if reached:
                path = np.array(a.path_to_root(new)[::-1] + b.path_to_root(last)[1:])
                return (path if a is start_tree else path[::-1]), it
        a, b = b, a
    return None, cfg.max_iter


def path_length(path: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
