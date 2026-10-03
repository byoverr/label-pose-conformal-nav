"""Split conformal prediction and the nonconformity scores compared in this project.

All scores are computed once per calibration unit (a scene with one drift realization),
in the planner's frame of reference. Calibrating on one score per scene is what makes the
units exchangeable: cells, frames and objects inside a scene are strongly correlated.

Scores (lower = map is more trustworthy for this scene):

miss distance (geometric, used by the pose-only and the joint calibration)
    s = max over avoid-class entities e of  max over cells x of e's true footprint of
        dist(x, A(k_e)),  capped at r_max,
    where A(k) is the map's region for class k. If s <= q, every true footprint lies inside
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


def miss_distance(regions: dict[int, np.ndarray], entities: list[Entity], res: float,
                  r_max: float) -> float:
    """Scene-level miss distance of true footprints w.r.t. per-class map regions."""
    if not entities:
        return 0.0
    fields = {k: distance_to(regions[k], res) for k in {e.cls for e in entities}}
    worst = max(float(fields[e.cls][e.cells[:, 0], e.cells[:, 1]].max()) for e in entities)
    return min(worst, r_max)


def label_score(pred: Map, entities: list[Entity]) -> float:
    if not entities:
        return 0.0
    occ = pred.occupied()
    worst = 0.0
    for e in entities:
        r, c = e.cells[:, 0], e.cells[:, 1]
        s = np.where(occ[r, c], 1.0 - pred.prob(e.cls)[r, c], UNCOVERABLE)
        worst = max(worst, float(s.max()))
    return worst
