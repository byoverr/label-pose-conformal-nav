import numpy as np
import pytest

from s2m.analysis import ArmParams, calibrate_class, covers, miss_at
from s2m.conformal import distance_to, miss_distance
from s2m.data import map_class_names
from s2m.entities import Entity
from s2m.experiment import LAMBDAS, R_MAX, transform_entities
from s2m.grid import GridSpec
from s2m.planning import SAFETY_DISTANCE, dilate, evaluate_path, plan_many


def score_row(miss: float, gtlab: float, label: float = 0.5, cls: str = "bike") -> dict:
    """A score row with the same miss distance at every stored lambda."""
    row = {f"label_score_{cls}": label, f"miss_gtlab_{cls}": gtlab}
    row.update({f"miss_pred_{cls}_{lam:.2f}": miss for lam in LAMBDAS})
    return row


def test_calibrate_class_joint_is_the_max_and_separate_the_sum():
    # n = 9, alpha = 0.1: rank ceil(10 * 0.9) = 9 -> every quantile is the maximum
    cal = [score_row(miss=0.1 * (i + 1), gtlab=0.05 * i) for i in range(9)]
    cal_L0 = [score_row(miss=0.05 * (i + 1), gtlab=0.0) for i in range(9)]
    p = calibrate_class(cal, cal_L0, "bike", alpha=0.1)
    assert p["joint"].radius == pytest.approx(0.9)
    assert p["label_only"].radius == pytest.approx(0.45)
    assert p["pose_only"].radius == pytest.approx(0.4)
    assert p["separate"].radius == pytest.approx(0.45 + 0.4)
    assert p["uncalibrated"].radius == 0.0


def test_too_few_scenes_or_huge_scores_abstain_and_count_as_covered():
    few = [score_row(0.1, 0.1) for _ in range(8)]  # n = 8 < 1/alpha - 1
    p = calibrate_class(few, few, "bike", alpha=0.1)
    assert all(p[a] is None for a in ("label_only", "pose_only", "separate", "joint"))
    huge = [score_row(R_MAX, 0.1) for _ in range(9)]
    assert calibrate_class(huge, huge, "bike", alpha=0.1)["joint"] is None
    assert covers(score_row(2.0, 0.0), "bike", None)  # a vacuous certificate is valid
    assert not covers(score_row(0.5, 0.0), "bike", ArmParams(0.05, 0.4))


def test_miss_at_reads_the_nearest_stored_threshold_below_and_rejects_large_lambdas():
    row = {f"miss_pred_bike_{lam:.2f}": lam for lam in LAMBDAS}
    assert miss_at(row, "bike", 0.055) == pytest.approx(0.05)
    with pytest.raises(ValueError):
        miss_at(row, "bike", 0.6)


def test_transform_entities_moves_footprints_rigidly():
    spec = GridSpec(0.0, 0.0, 0.05, (40, 40))
    e = Entity(1, np.array([[10, 10], [10, 11], [11, 10]]))
    assert np.array_equal(transform_entities([e], spec, np.eye(4))[0].cells, e.cells)
    D = np.eye(4)
    D[0, 3] = 0.25  # 25 cm along world x = 5 columns
    moved = transform_entities([e], spec, D)[0].cells
    assert np.array_equal(moved, e.cells + [0, 5])


def test_margin_from_miss_distance_keeps_true_distance():
    """Triangle inequality end to end: plan around the inflated map region, judge against the truth."""
    res, shape = 0.05, (60, 120)
    true_obj = np.zeros(shape, bool)
    true_obj[25:35, 55:65] = True  # the real object
    region = np.zeros(shape, bool)
    region[25:35, 63:69] = True  # the map sees only part of it, shifted
    q = miss_distance({1: region}, [Entity(1, np.argwhere(true_obj))], res, R_MAX)
    assert q > 0
    trav = ~dilate(region, SAFETY_DISTANCE + q, res)
    goal = np.zeros(shape, bool)
    goal[30, 115] = True
    path = plan_many(trav, [(30, 2)], [goal], res)[0]
    assert path is not None
    out = evaluate_path(path, distance_to(true_obj, res), np.full(shape, np.inf), distance_to(goal, res), res, reach=0.1)
    assert not out.violation and out.success


def test_map_class_names_precedence():
    target = {0: "other", 1: "indoor_plant", 2: "wall", 3: "ceiling", 4: "bike"}
    names = {0: "plant", 1: "shower wall", 2: "ceiling lamp", 3: "bicycle helmets", 4: "bicycle", 5: "wall plant"}
    rules = {"exact": {"plant": "indoor_plant", "bicycle": "bike"}, "last_word": {"wall": "wall"},
             "first_word": {"ceiling": "ceiling"}}
    lut = map_class_names(names, target, rules)
    assert [target[lut[i]] for i in range(6)] == ["indoor_plant", "wall", "ceiling", "other", "bike", "other"]


def test_transform_entities_keeps_cells_pushed_off_the_grid():
    spec = GridSpec(0.0, 0.0, 0.05, (40, 40))
    e = Entity(1, np.array([[10, 1], [10, 2]]))
    D = np.eye(4)
    D[0, 3] = -0.25  # 5 columns to the left: both cells leave the grid
    moved = transform_entities([e], spec, D)[0]
    assert np.array_equal(moved.cells, e.cells - [0, 5])
    assert not moved.inside(spec.shape).any() and not moved.mask(spec.shape).any()
