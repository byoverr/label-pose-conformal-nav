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
