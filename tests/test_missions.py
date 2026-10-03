import numpy as np

from s2m.analysis import ArmParams
from s2m.grid import GridSpec
from s2m.mapping import Map
from s2m.missions import keepout
from s2m.planning import SAFETY_DISTANCE


def two_object_map(k=3):
    """Class 1 at (10, 10), an unlabelled obstacle at (10, 30); 5 cm cells."""
    shape = (20, 40)
    m = Map(GridSpec(0.0, 0.0, 0.05, shape), np.zeros(shape), np.zeros(shape), np.zeros((*shape, k)))
    m.n_obs[10, 10] = m.n_obs[10, 30] = 10
    m.class_mass[10, 10, 1] = 10.0
    return m


def test_certified_class_keeps_out_only_its_region():
    m = two_object_map()
    keep = keepout(m, [1], {1: ArmParams(0.05, 0.1)}, {1: 0.0}, 0.05)
    assert keep[10, 10] and keep[10, 10 + int((SAFETY_DISTANCE + 0.1) / 0.05)]
    assert not keep[10, 30]  # the unlabelled obstacle is not treated as class 1


def test_abstaining_class_falls_back_to_all_occupied_cells():
    m = two_object_map()
    keep = keepout(m, [1], {1: None}, {1: 0.0}, 0.05)
    assert keep[10, 10] and keep[10, 30]


def test_no_fallback_means_no_certificate():
    m = two_object_map()
    assert keepout(m, [1], {1: None}, {1: None}, 0.05) is None


def test_make_tasks_respects_the_pass_by_rules():
    from s2m.conformal import distance_to
    from s2m.entities import Entity
    from s2m.experiment import SceneSetup
    from s2m.missions import MIN_START_GOAL, START_GOAL_CLEAR, make_tasks

    res, shape = 0.05, (60, 200)  # a 10 m x 3 m room
    m = Map(GridSpec(0.0, 0.0, res, shape), np.zeros(shape), np.ones(shape), np.zeros((*shape, 2)))
    m.n_obs[[0, -1], :] = m.n_obs[:, [0, -1]] = 10  # walls
    plant = np.argwhere(np.pad(np.ones((6, 6), bool), ((27, 27), (97, 97))))  # 30 cm object mid-room
    m.n_obs[plant[:, 0], plant[:, 1]] = 10
    setup = SceneSetup(None, [], m.spec, 2, [1], [Entity(1, plant)], m)
    tasks = make_tasks(setup, 5, seed=0, kind="pass_by")
    assert tasks
    obj = np.zeros(shape, bool)
    obj[plant[:, 0], plant[:, 1]] = True
    d = distance_to(obj, res)
    for t in tasks:
        (sr, sc), (gr, gc) = [m.spec.to_cell(np.array([x]), np.array([z]))[:2] for x, z in (t.start_xz, t.goal_xz)]
        assert d[sr[0], sc[0]] >= START_GOAL_CLEAR and d[gr[0], gc[0]] >= START_GOAL_CLEAR
        assert np.hypot(t.start_xz[0] - t.goal_xz[0], t.start_xz[1] - t.goal_xz[1]) >= MIN_START_GOAL - res
    assert make_tasks(setup, 5, seed=0, kind="random")
