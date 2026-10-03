import numpy as np

from s2m.rrt import GridChecker, RRTConfig, path_length, rrt_connect


def corridor_map():
    """20 x 60 free room with a wall that has a narrow gap."""
    t = np.ones((20, 60), bool)
    t[:, 30] = False
    t[9:11, 30] = True
    return t


def check_path(t, path, res):
    chk = GridChecker(t, res)
    assert all(chk.segment_free(a, b) for a, b in zip(path[:-1], path[1:]))


def test_open_space_path_is_valid_and_near_straight():
    t = np.ones((20, 60), bool)
    path, it = rrt_connect(t, (10, 2), (10, 57), 0.1, RRTConfig(), np.random.default_rng(0))
    assert path is not None
    check_path(t, path, 0.1)
    assert np.allclose(path[0], [0.25, 1.05]) and np.allclose(path[-1], [5.75, 1.05])
    assert path_length(path) < 1.2 * 5.5


def test_both_variants_pass_a_narrow_gap():
    t = corridor_map()
    for cone in (False, True):
        ok = 0
        for seed in range(10):
            path, _ = rrt_connect(t, (2, 5), (17, 55), 0.1, RRTConfig(cone=cone, max_iter=3000),
                                  np.random.default_rng(seed))
            if path is not None:
                check_path(t, path, 0.1)
                ok += 1
        assert ok >= 8


def test_blocked_returns_none():
    t = corridor_map()
    t[:, 30] = False
    path, it = rrt_connect(t, (2, 5), (17, 55), 0.1, RRTConfig(max_iter=300), np.random.default_rng(0))
    assert path is None and it == 300
