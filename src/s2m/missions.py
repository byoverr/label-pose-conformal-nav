"""Reach-avoid missions: plan on the predicted map, judge against the truth.

A task = start point + goal point on the floor. Two task families:
  pass_by  (default): start and goal lie far from every avoid-class object (>= START_GOAL_CLEAR),
           but the shortest obstacle-free path between them passes within SAFETY_DISTANCE of an
           avoid object. A safe detour exists; a planner that under-estimates the object's
           footprint cuts the corner and violates. This is the situation the guarantee is for.
  approach: goals 0.7-1.2 m from an avoid object ("drive to the sofa next to the plant"). Kept
           for reference: once the calibrated keep-out exceeds 1.2 m the goal itself becomes
           forbidden, so these tasks mostly measure how close a certificate lets the robot get.
Goals are geometric on purpose: the guarantee studied here is about avoid classes, and
semantic goal errors would confound it.

The planner sees only the drifted predicted map; every calibration arm turns that map into
keep-out zones (avoid-class region at threshold lam, dilated by SAFETY_DISTANCE + arm radius).
Outcomes are judged with the true geometry expressed in the planner frame, which is what the
robot actually experiences when it follows the plan.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from s2m.analysis import ArmParams
from s2m.conformal import distance_to
from s2m.drift import world_correction
from s2m.experiment import SceneSetup, transform_entities
from s2m.mapping import build_map, floor_class_ids
from s2m.planning import ROBOT_RADIUS, SAFETY_DISTANCE, dilate, evaluate_path, plan_many

GOAL_TOL = 0.3  # metres: reaching the goal = ending this close to the goal point
GOAL_BAND = (SAFETY_DISTANCE + 0.2, SAFETY_DISTANCE + 0.7)  # goal distance to an avoid object
MIN_START_GOAL = 3.0  # metres between start and goal
CLEAR = ROBOT_RADIUS + 0.1  # start/goal clearance from any obstacle
START_GOAL_CLEAR = 1.5  # pass_by: start/goal distance to every avoid object (> d + typical radius)


@dataclass
class Task:
    start_xz: tuple[float, float]  # GT world frame
    goal_xz: tuple[float, float]  # GT world frame


def make_tasks(setup: SceneSetup, n: int, seed: int = 0, kind: str = "pass_by") -> list[Task]:
    s, rng, res = setup, np.random.default_rng(seed), setup.spec.res
    avoid = np.zeros(s.spec.shape, bool)
    for e in s.entities:
        avoid |= e.mask(s.spec.shape)
    d_avoid = distance_to(avoid, res)
    roomy = s.truth.free() & (distance_to(s.truth.occupied(), res) > CLEAR)
    xs, zs = s.spec.cell_centers()
    to_xz = lambda rc: (float(xs[rc[1]]), float(zs[rc[0]]))

    if kind == "approach":
        goals = np.argwhere(roomy & (d_avoid >= GOAL_BAND[0]) & (d_avoid <= GOAL_BAND[1]))
        starts = np.argwhere(roomy & (d_avoid > GOAL_BAND[0]))
    else:
        goals = starts = np.argwhere(roomy & (d_avoid >= START_GOAL_CLEAR))
    if len(goals) == 0 or len(starts) == 0:
        return []

    # pass_by: keep pairs whose shortest obstacle-only path grazes an avoid object
    trav = s.truth.free() & ~dilate(s.truth.occupied(), ROBOT_RADIUS, res)
    tasks = []
    for _ in range(80):
        if len(tasks) == n:
            break
        batch = []
        while len(batch) < 2 * n:
            g, st = goals[rng.integers(len(goals))], starts[rng.integers(len(starts))]
            if np.hypot(*(g - st)) * res >= MIN_START_GOAL:
                batch.append((tuple(st), tuple(g)))
        if kind == "approach":
            tasks += [Task(to_xz(a), to_xz(b)) for a, b in batch][: n - len(tasks)]
            continue
        paths = plan_many(trav, [a for a, _ in batch], [_disk(s.spec.shape, b, 0.0, res) for _, b in batch], res)
        for (a, b), path in zip(batch, paths):
            if path is not None and d_avoid[path[:, 0], path[:, 1]].min() < SAFETY_DISTANCE and len(tasks) < n:
                tasks.append(Task(to_xz(a), to_xz(b)))
    return tasks


def _to_cell(spec, xz, D):
    """GT-world point -> cell of the planner frame (moved by the query-time correction D)."""
    p = D[:3, :3] @ np.array([xz[0], 0.0, xz[1]]) + D[:3, 3]
    r, c, inside = spec.to_cell(np.array([p[0]]), np.array([p[2]]))
    return (int(r[0]), int(c[0])) if inside[0] else None


def _disk(shape, cell, radius, res):
    m = np.zeros(shape, bool)
    m[cell] = True
    return dilate(m, radius, res)


def keepout(pred, avoid_ids, params: dict[int, ArmParams | None], fallback: dict[int, float | None],
            res: float) -> np.ndarray | None:
    """Keep-out cells of one arm, or None if the arm cannot certify the mission.

    A class whose certificate abstains falls back to geometry: every occupied cell may be that
    class, inflated by the class's pose-only radius. Same rule for every arm.
    """
    keep = np.zeros(pred.spec.shape, bool)
    for k in avoid_ids:
        p = params[k]
        if p is not None:
            keep |= dilate(pred.region(k, p.lam), SAFETY_DISTANCE + p.radius, res)
        elif fallback.get(k) is not None:
            keep |= dilate(pred.occupied(), SAFETY_DISTANCE + fallback[k], res)
        else:
            return None
    return keep


def run_realization(setup: SceneSetup, tasks: list[Task], est_poses: np.ndarray,
                    arms: dict[str, dict[int, ArmParams | None]],
                    fallback: dict[int, float | None]) -> list[dict]:
    """Plan every task with every arm (plus an oracle on the true map) for one drift realization.

    arms[arm][class_id] = per-class parameters (None = that class's certificate abstains).
    """
    s, res, shape = setup, setup.spec.res, setup.spec.shape
    D_q = world_correction(est_poses, s.scene.poses)[-1]

    # Truth in the planner frame: the GT map moved rigidly by the query-time pose error.
    truth = build_map(s.obs, D_q[None] @ s.scene.poses, s.spec, s.n_classes, "gt",
                      floor_class_ids(s.scene))
    avoid_true = np.zeros(shape, bool)
    for e in transform_entities(s.entities, s.spec, D_q):
        avoid_true |= e.mask(shape)
    d_avoid, d_obst = distance_to(avoid_true, res), distance_to(truth.occupied(), res)

    cells = [(_to_cell(s.spec, t.start_xz, D_q), _to_cell(s.spec, t.goal_xz, D_q)) for t in tasks]
    valid = [i for i, (a, b) in enumerate(cells) if a is not None and b is not None]
    starts = [cells[i][0] if i in valid else None for i in range(len(tasks))]
    goal_masks = [_disk(shape, cells[i][1], GOAL_TOL, res) if i in valid else None for i in range(len(tasks))]
    d_goal = {i: distance_to(_disk(shape, cells[i][1], 0.0, res), res) for i in valid}

    pred = build_map(s.obs, est_poses, s.spec, s.n_classes, "det")
    base = pred.free() & ~dilate(pred.occupied(), ROBOT_RADIUS, res)

    plans, abstained, fell_back = {}, {}, {}
    for arm, params in arms.items():
        keep = keepout(pred, s.avoid_ids, params, fallback, res)
        abstained[arm] = keep is None
        fell_back[arm] = sum(params[k] is None for k in s.avoid_ids)
        plans[arm] = [None] * len(tasks) if keep is None else _plan_subset(base & ~keep, starts, goal_masks, valid, res)

    oracle_trav = (truth.free() & ~dilate(truth.occupied(), ROBOT_RADIUS, res)
                   & ~dilate(avoid_true, SAFETY_DISTANCE, res))
    plans["oracle"] = _plan_subset(oracle_trav, starts, goal_masks, valid, res)

    rows = []
    for i in valid:
        judge = lambda path: evaluate_path(path, d_avoid, d_obst, d_goal[i], res, reach=GOAL_TOL)
        oracle_len = judge(plans["oracle"][i]).length
        for arm in plans:
            o = judge(plans[arm][i])
            rows.append({"task": i, "arm": arm,
                         "abstained": abstained.get(arm, False),
                         "classes_fell_back": fell_back.get(arm, 0),
                         **asdict(o), "success": o.success, "oracle_length": oracle_len})
    return rows


def _plan_subset(trav, starts, goal_masks, valid, res):
    out = [None] * len(starts)
    sub = plan_many(trav, [starts[i] for i in valid], [goal_masks[i] for i in valid], res)
    for i, path in zip(valid, sub):
        out[i] = path
    return out
