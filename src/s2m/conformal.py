"""Split conformal prediction and the nonconformity scores compared in this project.

All scores are computed once per calibration unit (a scene with one drift realization),
in the planner's frame of reference. Calibrating on one score per scene is what makes the
units exchangeable: cells, frames and objects inside a scene are strongly correlated.

Scores (lower = map is more trustworthy for this scene):

miss distance (geometric, used by the pose-only and the joint calibration)
    s = max over avoid-class entities e of  max over cells x of e's true footprint of
        dist(x, A(k_e)),  capped at r_max,
    where A(k) is the map's region for class k (footprints and regions are sets of cell
    centres; a footprint pushed off the grid by the pose error is still measured). If s <= q,
    every true footprint lies inside
    its class region dilated by q, so a planner keeping distance d + q from A(k) keeps true
    distance >= d (triangle inequality).

label score (Sundarsingh et al.-style, label-space calibration)
    s = max over true footprint cells x of (1 - p_k(x)), and UNCOVERABLE (=2) for a cell
    that is not even occupied in the predicted map. Thresholding the map at lam = 1 - q
    then covers every true footprint cell, with no geometric slack.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from s2m.entities import Entity
from s2m.mapping import Map

UNCOVERABLE = 2.0


def conformal_quantile(scores, alpha: float) -> float:
    """The ceil((n+1)(1-alpha))-th smallest score; +inf if n is too small for this alpha."""
    s = np.sort(np.asarray(scores, dtype=float))
    k = math.ceil((len(s) + 1) * (1 - alpha))
    return float(s[k - 1]) if k <= len(s) else math.inf


def coverage_beta_params(n: int, alpha: float) -> tuple[int, int]:
    """Test coverage of split CP (no ties) is Beta(k, n + 1 - k) over calibration draws."""
    k = math.ceil((n + 1) * (1 - alpha))
    return k, n + 1 - k


def distance_to(region: np.ndarray, res: float) -> np.ndarray:
    """Metric distance from every cell to the nearest cell of `region` (inf if region is empty)."""
    if not region.any():
        return np.full(region.shape, np.inf)
    return ndimage.distance_transform_edt(~region) * res


def distance_at(region: np.ndarray, cells: np.ndarray, res: float, r_max: float) -> np.ndarray:
    """Distance from cell centres (row, col; off-grid cells allowed) to the nearest cell of
    `region`, capped at r_max. Off-grid cells are measured on a grid padded by r_max."""
    h, w = region.shape
    r, c = cells[:, 0], cells[:, 1]
    if ((r >= 0) & (r < h) & (c >= 0) & (c < w)).all():
        return np.minimum(distance_to(region, res)[r, c], r_max)
    pad = int(np.ceil(r_max / res)) + 1
    field = distance_to(np.pad(region, pad), res)
    r, c = r + pad, c + pad
    ok = (r >= 0) & (r < h + 2 * pad) & (c >= 0) & (c < w + 2 * pad)
    out = np.full(len(cells), float(r_max))
    out[ok] = np.minimum(field[r[ok], c[ok]], r_max)
    return out


def distance_from(cells: np.ndarray, shape: tuple[int, int], res: float, reach: float) -> np.ndarray:
    """Distance from every grid cell to the nearest of `cells` (off-grid cells allowed; those
    farther than `reach` outside the grid are ignored)."""
    pad = int(np.ceil(reach / res)) + 1
    m = np.zeros((shape[0] + 2 * pad, shape[1] + 2 * pad), bool)
    r, c = cells[:, 0] + pad, cells[:, 1] + pad
    ok = (r >= 0) & (r < m.shape[0]) & (c >= 0) & (c < m.shape[1])
    m[r[ok], c[ok]] = True
    return distance_to(m, res)[pad:pad + shape[0], pad:pad + shape[1]]


def miss_distance(regions: dict[int, np.ndarray], entities: list[Entity], res: float,
                  r_max: float) -> float:
    """Scene-level miss distance of true footprints w.r.t. per-class map regions: the directed
    Hausdorff distance from footprint cell centres to region cell centres, capped at r_max."""
    if not entities:
        return 0.0
    worst = 0.0
    for k in {e.cls for e in entities}:
        cells = np.concatenate([e.cells for e in entities if e.cls == k])
        worst = max(worst, float(distance_at(regions[k], cells, res, r_max).max()))
    return min(worst, r_max)


def label_score(pred: Map, entities: list[Entity]) -> float:
    if not entities:
        return 0.0
    occ = pred.occupied()
    worst = 0.0
    for e in entities:
        if not e.inside(occ.shape).all():
            return UNCOVERABLE  # part of the object lies outside the map: no label set holds it
        r, c = e.cells[:, 0], e.cells[:, 1]
        s = np.where(occ[r, c], 1.0 - pred.prob(e.cls)[r, c], UNCOVERABLE)
        worst = max(worst, float(s.max()))
    return worst
