"""Reach-avoid missions: plan on the predicted map, judge against the truth.

A task = start point + goal point on the floor. Two task families:
  pass_by  (default): start and goal lie far from every avoid-class object (>= START_GOAL_CLEAR),
           but the shortest obstacle-free path between them passes within SAFETY_DISTANCE of an
           avoid object. A safe detour exists; a planner that under-estimates the object's
           footprint cuts the corner and violates. This is the situation the guarantee is for.
  random:  same start/goal rules as pass_by, any pair connected in the true map; no requirement that
           the path comes near an avoid object. The unselected task mix, for comparison with
           published mission success rates.
Goals are geometric on purpose: the guarantee studied here is about avoid classes, and
semantic goal errors would confound it. The goal is handed to the planner in its own frame, so a
found path always ends at the goal: "success" means a path was found and it is safe
(no violation of the avoid-class distance, no collision with a true obstacle).

The planner sees only the drifted predicted map; every calibration arm turns that map into
keep-out zones (avoid-class region at threshold lam, dilated by SAFETY_DISTANCE + arm radius).
Outcomes are judged with the true geometry expressed in the planner frame, which is what the
robot actually experiences when it follows the plan.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from scipy import ndimage

from s2m.analysis import ArmParams
from s2m.conformal import distance_from, distance_to
from s2m.drift import world_correction
from s2m.experiment import R_MAX, SceneSetup, predicted_map, transform_entities
from s2m.mapping import build_map, floor_class_ids
from s2m.planning import ROBOT_RADIUS, SAFETY_DISTANCE, dilate, evaluate_path, plan_many

GOAL_TOL = 0.3  # metres: reaching the goal = ending this close to the goal point
MIN_START_GOAL = 3.0  # metres between start and goal
CLEAR = ROBOT_RADIUS + 0.1  # start/goal clearance from any obstacle
START_GOAL_CLEAR = 1.5  # start/goal distance to every avoid object (> d + typical radius)
MAX_BATCHES = 80  # task sampling gives up after this many batches (small or cluttered scenes)
MAX_DRAWS_PER_BATCH = 100_000


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

    if kind not in ("pass_by", "random"):
        raise ValueError(f"unknown task kind {kind!r}")
    goals = starts = np.argwhere(roomy & (d_avoid >= START_GOAL_CLEAR))
    if len(goals) < 2:
        return []

    # random: any pair connected in the true map; pass_by: only pairs whose shortest
    # obstacle-only path grazes an avoid object
    trav = s.truth.free() & ~dilate(s.truth.occupied(), ROBOT_RADIUS, res)
    tasks = []
    for _ in range(MAX_BATCHES):
        if len(tasks) == n:
            break
        batch = []
        for _ in range(MAX_DRAWS_PER_BATCH):  # bounded: a tiny scene may have no pair MIN_START_GOAL apart
            if len(batch) == 2 * n:
                break
            g, st = goals[rng.integers(len(goals))], starts[rng.integers(len(starts))]
            if np.hypot(*(g - st)) * res >= MIN_START_GOAL:
                batch.append((tuple(st), tuple(g)))
        if not batch:
            break
        paths = plan_many(trav, [a for a, _ in batch], [disk(s.spec.shape, b, 0.0, res) for _, b in batch], res)
        for (a, b), path in zip(batch, paths):
            grazes = path is not None and d_avoid[path[:, 0], path[:, 1]].min() < SAFETY_DISTANCE
            if path is not None and (grazes or kind == "random") and len(tasks) < n:
                tasks.append(Task(to_xz(a), to_xz(b)))
    return tasks


def to_cell(spec, xz, D):
    """GT-world point -> cell of the planner frame (moved by the query-time correction D)."""
    p = D[:3, :3] @ np.array([xz[0], 0.0, xz[1]]) + D[:3, 3]
    r, c, inside = spec.to_cell(np.array([p[0]]), np.array([p[2]]))
    return (int(r[0]), int(c[0])) if inside[0] else None


def disk(shape, cell, radius, res):
    """Cells within `radius` metres of one cell."""
    m = np.zeros(shape, bool)
    m[cell] = True
    return dilate(m, radius, res)


def keepout(pred, avoid_ids, params: dict[int, ArmParams | None], fallback: dict[int, float | None],
            res: float) -> np.ndarray | None:
    """Keep-out cells of one arm, or None if the arm cannot certify the mission.

    A class whose certificate abstains falls back to the class-agnostic certificate: every occupied
    cell may be that class, inflated by the calibrated geometry-only radius (fallback[k]; valid on
    its own, see s2m.analysis). Same rule for every arm.
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


def _exposed(trav: np.ndarray, starts, d_avoid: np.ndarray) -> bool:
    """Worst case over all plans: can the robot reach, from some task start, a cell closer than
    SAFETY_DISTANCE to a true avoid object without leaving the arm's traversable set?"""
    labels, _ = ndimage.label(trav, structure=np.ones((3, 3)))
    ids = {int(labels[st]) for st in starts if st is not None} - {0}
    return bool(ids) and bool((np.isin(labels, list(ids)) & (d_avoid < SAFETY_DISTANCE)).any())


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
    moved = [e.cells for e in transform_entities(s.entities, s.spec, D_q)]
    d_avoid = distance_from(np.concatenate(moved) if moved else np.zeros((0, 2), int), shape, res, R_MAX)
    d_obst = distance_to(truth.occupied(), res)

    cells = [(to_cell(s.spec, t.start_xz, D_q), to_cell(s.spec, t.goal_xz, D_q)) for t in tasks]
    valid = [i for i, (a, b) in enumerate(cells) if a is not None and b is not None]
    starts = [cells[i][0] if i in valid else None for i in range(len(tasks))]
    goal_masks = [disk(shape, cells[i][1], GOAL_TOL, res) if i in valid else None for i in range(len(tasks))]
    d_goal = {i: distance_to(disk(shape, cells[i][1], 0.0, res), res) for i in valid}

    pred = predicted_map(s, est_poses)
    base = pred.free() & ~dilate(pred.occupied(), ROBOT_RADIUS, res)

    plans, abstained, fell_back, exposed = {}, {}, {}, {}
    for arm, params in arms.items():
        keep = keepout(pred, s.avoid_ids, params, fallback, res)
        abstained[arm] = keep is None
        fell_back[arm] = sum(params[k] is None for k in s.avoid_ids)
        exposed[arm] = keep is not None and _exposed(base & ~keep, starts, d_avoid)
        plans[arm] = [None] * len(tasks) if keep is None else _plan_subset(base & ~keep, starts, goal_masks, valid, res)

    oracle_trav = (truth.free() & ~dilate(truth.occupied(), ROBOT_RADIUS, res)
                   & (d_avoid > SAFETY_DISTANCE))
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
                         "exposed": exposed.get(arm, False),
                         **asdict(o), "success": o.success, "oracle_length": oracle_len})
    return rows


def _plan_subset(trav, starts, goal_masks, valid, res):
    out = [None] * len(starts)
    sub = plan_many(trav, [starts[i] for i in valid], [goal_masks[i] for i in valid], res)
    for i, path in zip(valid, sub):
        out[i] = path
    return out


RISK_MARGINS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0)


EXEC_STEP, EXEC_TURN = 0.03, np.radians(2.0)  # motion primitives of the mapping trajectories (drift levels are per step)


def path_poses(cells: np.ndarray, spec, height: float = 1.5) -> np.ndarray:
    """Level camera poses (camera-to-world, camera y = world -y) of a robot that follows a grid path cell by
    cell: it turns in place and drives straight between adjacent cell centres with the motion primitives of
    the mapping trajectories (3 cm, 2 deg). Without drift every pose lies in a path cell, so a zero-drift
    execution is evaluated exactly like the plan."""
    from s2m.drift import rot_y

    xs, zs = spec.cell_centers()
    pts = np.stack([xs[cells[:, 1]], zs[cells[:, 0]]], 1)
    way = pts

    def pose(xz, yaw):
        T = np.eye(4)
        T[:3, :3] = rot_y(yaw) @ np.diag([-1.0, -1.0, 1.0])  # level camera looking along the heading
        T[:3, 3] = [xz[0], height, xz[1]]
        return T

    poses, heading = [], None
    for a, b in zip(way[:-1], way[1:]):
        d = b - a
        h = float(np.arctan2(d[0], d[1]))  # yaw of the camera z axis (x = sin, z = cos)
        if heading is None:
            heading = h
            poses.append(pose(a, h))
        turn = (h - heading + np.pi) % (2 * np.pi) - np.pi
        for f in np.linspace(0, 1, int(np.ceil(abs(turn) / EXEC_TURN)) + 1)[1:]:
            poses.append(pose(a, heading + f * turn))
        heading = h
        for f in np.linspace(0, 1, int(np.ceil(np.linalg.norm(d) / EXEC_STEP)) + 1)[1:]:
            poses.append(pose(a + f * d, h))
    return np.array(poses) if poses else np.zeros((0, 4, 4))


def execution_outcomes(cells: np.ndarray, spec, d_avoid: np.ndarray, d_obst: np.ndarray, model,
                       rng: np.random.Generator, draws: int) -> dict:
    """Simulated executions of a planned path under continued drift. exec_unsafe / exec_collision / exec_fail:
    fraction of draws that come closer than the safety distance to an avoid-class object / hit an obstacle /
    do either; exec_u: the largest deviation from the commanded position divided by the path length, first
    draw (the score an execution log would give); exec_dev: mean over draws of the largest deviation."""
    from s2m.drift import path_length, simulate

    P = path_poses(cells, spec)
    if len(P) < 2:
        return {"exec_unsafe": 0.0, "exec_collision": 0.0, "exec_fail": 0.0, "exec_u": 0.0, "exec_dev": 0.0}
    cmd = P[:, [0, 2], 3]
    length = max(path_length(P), 1e-9)
    unsafe = coll = fail = 0
    devs = []
    for _ in range(draws):
        q = simulate(P, model, rng)[:, [0, 2], 3]
        r, c, inside = spec.to_cell(q[:, 0], q[:, 1])
        r, c = np.clip(r, 0, spec.shape[0] - 1), np.clip(c, 0, spec.shape[1] - 1)
        u = bool(np.where(inside, d_avoid[r, c], np.inf).min() < SAFETY_DISTANCE)  # off the map: no avoid object
        k = bool(np.where(inside, d_obst[r, c], 0.0).min() < ROBOT_RADIUS - spec.res)  # off the map: a wall
        unsafe, coll, fail = unsafe + u, coll + k, fail + (u or k)
        devs.append(float(np.linalg.norm(q - cmd, axis=1).max()))
    return {"exec_unsafe": unsafe / draws, "exec_collision": coll / draws, "exec_fail": fail / draws,
            "exec_u": devs[0] / length, "exec_dev": float(np.mean(devs))}


def run_risk_realization(setup: SceneSetup, tasks: list[Task], est_poses: np.ndarray,
                         margins=RISK_MARGINS, lam: float = 0.05, exec_model=None, exec_draws: int = 0,
                         rng: np.random.Generator | None = None) -> list[dict]:
    """Plan every task with a grid of uncalibrated margins and record each path's clearance to every
    avoid-class region (risk is assigned later from calibration scores, see scripts/analyze_risk.py).

    clearance_<class> = min over path cells of dist(cell, region of the class) - SAFETY_DISTANCE (>= margin);
    inf when the class region is empty.
    """
    s, res, shape = setup, setup.spec.res, setup.spec.shape
    D_q = world_correction(est_poses, s.scene.poses)[-1]
    truth = build_map(s.obs, D_q[None] @ s.scene.poses, s.spec, s.n_classes, "gt", floor_class_ids(s.scene))
    moved = [e.cells for e in transform_entities(s.entities, s.spec, D_q)]
    d_avoid = distance_from(np.concatenate(moved) if moved else np.zeros((0, 2), int), shape, res, R_MAX)
    d_obst = distance_to(truth.occupied(), res)
    cells = [(to_cell(s.spec, t.start_xz, D_q), to_cell(s.spec, t.goal_xz, D_q)) for t in tasks]
    valid = [i for i, (a, b) in enumerate(cells) if a is not None and b is not None]
    starts = [cells[i][0] if i in valid else None for i in range(len(tasks))]
    goal_masks = [disk(shape, cells[i][1], GOAL_TOL, res) if i in valid else None for i in range(len(tasks))]
    d_goal = {i: distance_to(disk(shape, cells[i][1], 0.0, res), res) for i in valid}
    pred = predicted_map(s, est_poses)
    base = pred.free() & ~dilate(pred.occupied(), ROBOT_RADIUS, res)
    regions = {k: pred.region(k, lam) for k in s.avoid_ids}
    fields = {k: distance_to(regions[k], res) for k in s.avoid_ids}
    rows, exec_cache = [], {}
    for m in margins:
        keep = np.zeros(shape, bool)
        for k in s.avoid_ids:
            keep |= dilate(regions[k], SAFETY_DISTANCE + m, res)
        paths = _plan_subset(base & ~keep, starts, goal_masks, valid, res)
        for i in valid:
            o = evaluate_path(paths[i], d_avoid, d_obst, d_goal[i], res, reach=GOAL_TOL)
            row = {"task": i, "margin": m, "planned": o.planned, "violation": o.violation,
                   "collision": o.collision, "length": o.length}
            for k in s.avoid_ids:
                name = s.scene.classes[k]
                row[f"clearance_{name}"] = (float(fields[k][paths[i][:, 0], paths[i][:, 1]].min()) - SAFETY_DISTANCE
                                            if paths[i] is not None else np.nan)
            if exec_draws and paths[i] is not None:
                key = paths[i].tobytes()
                if key not in exec_cache:
                    exec_cache[key] = execution_outcomes(paths[i], s.spec, d_avoid, d_obst, exec_model, rng, exec_draws)
                row.update(exec_cache[key])
            rows.append(row)
    return rows
