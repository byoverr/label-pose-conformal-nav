"""Calibration arms evaluated over random calibration/test splits of scenes.

Unit of calibration = one scene with one drift realization. For every split, calibration scenes
contribute one realization each (drawn at random); test scenes contribute all their other
realizations. An arm "covers" a test realization if every true avoid-class footprint lies inside
the arm's region (map threshold lam) dilated by the arm's radius, in the planner frame.

Arms (alpha = target miscoverage):
  uncalibrated : region at lam0, no inflation (what a planner does by default)
  label_L0     : label-space CP calibrated without drift (Sundarsingh et al.-style), no inflation
  label_drift  : label-space CP calibrated on drifted maps, no inflation
  separate     : label-space CP (no drift) + pose radius calibrated separately, each at alpha
  bonferroni   : same as separate, each at alpha / 2 (valid for any dependence)
  joint        : one miss-distance CP at lam0 (this work)
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from s2m.conformal import UNCOVERABLE, conformal_quantile
from s2m.experiment import LAMBDAS, R_MAX

TOL = 0.05  # one grid cell: "inside the region" for arms without inflation
LAM0 = 0.05  # region threshold of the joint and uncalibrated arms, chosen on the dev scene
ARMS = ("uncalibrated", "label_L0", "label_drift", "separate", "bonferroni", "joint")


@dataclass(frozen=True)
class ArmParams:
    lam: float  # map threshold defining the avoid-class region
    radius: float  # extra keep-out distance around the region (metres)


def load_scores(score_dir: Path, exclude=()) -> list[dict]:
    rows = []
    for p in sorted(Path(score_dir).glob("*.csv")):
        if p.stem in exclude:
            continue
        for r in csv.DictReader(open(p)):
            rows.append({k: (v if k in ("scene", "level") else float(v)) for k, v in r.items()})
    return rows


def miss_at(row: dict, lam: float) -> float:
    """Miss distance of the predicted map at threshold lam (nearest stored lambda not above lam,
    i.e. never a smaller region than the arm asked for)."""
    i = np.searchsorted(LAMBDAS, lam + 1e-9) - 1
    return row[f"miss_pred_{LAMBDAS[max(i, 0)]:.2f}"]


def _label_lam(rows: list[dict], alpha: float) -> float | None:
    q = conformal_quantile([r["label_score"] for r in rows], alpha)
    return None if q >= UNCOVERABLE - 1e-9 else max(0.0, 1.0 - q)


def _radius(scores, alpha: float) -> float | None:
    q = conformal_quantile(scores, alpha)
    return None if not math.isfinite(q) or q >= R_MAX else q


def calibrate_arms(cal: list[dict], cal_L0: list[dict], alpha: float,
                   lam0: float = LAM0) -> dict[str, ArmParams | None]:
    """Parameters of every arm from calibration rows (None = abstains: no certificate)."""
    out: dict[str, ArmParams | None] = {"uncalibrated": ArmParams(lam0, 0.0)}
    for name, rows in (("label_L0", cal_L0), ("label_drift", cal)):
        lam = _label_lam(rows, alpha)
        out[name] = None if lam is None else ArmParams(lam, 0.0)
    for name, a in (("separate", alpha), ("bonferroni", alpha / 2)):
        lam, r = _label_lam(cal_L0, a), _radius([c["miss_gtlab"] for c in cal], a)
        out[name] = None if lam is None or r is None else ArmParams(lam, r)
    r = _radius([miss_at(c, lam0) for c in cal], alpha)
    out["joint"] = None if r is None else ArmParams(lam0, r)
    return out


def covers(row: dict, p: ArmParams) -> bool:
    return miss_at(row, p.lam) <= max(p.radius, TOL) + 1e-9


def run_splits(rows: list[dict], level: str, alpha: float, n_cal: int, n_splits: int,
               lam0: float = LAM0, seed: int = 0) -> dict[str, dict]:
    """Coverage / radius / threshold / abstention of every arm at one drift level."""
    rng = np.random.default_rng(seed)
    by_scene: dict[str, dict[str, list]] = {}
    for r in rows:
        by_scene.setdefault(r["scene"], {}).setdefault(r["level"], []).append(r)
    scenes = sorted(by_scene)
    stats = {a: {"coverage": [], "radius": [], "lam": [], "abstain": []} for a in ARMS}
    for _ in range(n_splits):
        perm = rng.permutation(scenes)
        cal_s, test_s = perm[:n_cal], perm[n_cal:]
        pick = lambda s, lv: by_scene[s][lv][rng.integers(len(by_scene[s][lv]))]
        cal = [pick(s, level) for s in cal_s]
        cal_L0 = [pick(s, "L0") for s in cal_s]
        test = [r for s in test_s for r in by_scene[s][level]]
        for arm, p in calibrate_arms(cal, cal_L0, alpha, lam0).items():
            st = stats[arm]
            st["abstain"].append(float(p is None))
            st["coverage"].append(np.nan if p is None else float(np.mean([covers(t, p) for t in test])))
            st["radius"].append(np.nan if p is None else p.radius)
            st["lam"].append(np.nan if p is None else p.lam)
    return stats
