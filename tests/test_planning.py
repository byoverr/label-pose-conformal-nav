import numpy as np
import pytest

from s2m.planning import dilate, evaluate_path, path_length, shortest_path


def test_straight_corridor_and_wall_detour():
    free = np.ones((20, 40), bool)
    goal = np.zeros_like(free)
    goal[10, 35] = True
    path = shortest_path(free, (10, 2), goal, res=0.1)
    assert path_length(path, 0.1) == pytest.approx(3.3)  # 33 straight steps

    free[:, 20] = False  # wall with a gap at the top row
    free[0, 20] = True
    detour = shortest_path(free, (10, 2), goal, res=0.1)
    assert path_length(detour, 0.1) > 3.3
    assert free[detour[:, 0], detour[:, 1]].all()


def test_unreachable_goal_returns_none():
    free = np.ones((10, 10), bool)
    free[:, 5] = False
    goal = np.zeros_like(free)
    goal[5, 8] = True
    assert shortest_path(free, (5, 1), goal, res=0.1) is None


def test_dilate_is_euclidean():
    m = np.zeros((21, 21), bool)
    m[10, 10] = True
    d = dilate(m, 0.5, res=0.1)
    assert d[10, 15] and not d[10, 16]
    assert d[13, 14] and not d[14, 14]  # sqrt(4^2+4^2)*0.1 = 0.566 > 0.5


def test_evaluate_path_flags_violation():
    path = np.array([[5, c] for c in range(10)])
    avoid = np.full((10, 10), 1.0)
    avoid[5, 4] = 0.2  # passes 0.2 m from an avoid object
    far = np.full((10, 10), 5.0)
    goal = np.full((10, 10), 5.0)
    goal[5, 9] = 0.0
    out = evaluate_path(path, avoid, far, goal, res=0.05, reach=0.6)
    assert out.planned and out.violation and out.reached and not out.success
