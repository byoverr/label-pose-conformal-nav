import math

import numpy as np
import pytest

from s2m.conformal import (conformal_quantile, coverage_beta_params, label_score,
                           miss_distance)
from s2m.entities import Entity
from s2m.grid import GridSpec
from s2m.mapping import Map


def test_quantile_rank_and_small_n():
    s = np.arange(1, 10)  # n = 9
    assert conformal_quantile(s, 0.1) == 9  # ceil(10 * 0.9) = 9 -> the maximum
    assert conformal_quantile(s, 0.05) == math.inf  # needs n >= 19
    assert conformal_quantile(np.arange(1, 20), 0.05) == 19


def test_split_conformal_coverage_matches_theory():
    rng = np.random.default_rng(0)
    n, alpha, trials = 11, 0.1, 4000
    hits = []
    for _ in range(trials):
        cal, test = rng.exponential(size=n), rng.exponential()
        hits.append(test <= conformal_quantile(cal, alpha))
    a, b = coverage_beta_params(n, alpha)
    assert np.mean(hits) == pytest.approx(a / (a + b), abs=0.015)  # E[coverage] = 11/12


def empty_map(shape=(40, 40), k=3):
    spec = GridSpec(0.0, 0.0, 0.05, shape)
    return Map(spec, np.zeros(shape), np.zeros(shape), np.zeros((*shape, k)))


def test_miss_distance_of_shifted_region_equals_shift():
    footprint = np.argwhere(np.ones((5, 5), bool)) + 10  # cells 10..14
    e = Entity(1, footprint)
    region = np.zeros((40, 40), bool)
    region[10:15, 16:21] = True  # same block shifted by 6 cells along columns
    d = miss_distance({1: region}, [e], res=0.05, r_max=3.0)
    assert d == pytest.approx(6 * 0.05)
    assert miss_distance({1: np.zeros((40, 40), bool)}, [e], 0.05, r_max=3.0) == 3.0  # missed class


def test_label_score_uses_one_minus_probability_and_flags_unobserved():
    m = empty_map()
    m.n_obs[10:15, 10:15] = 10
    m.class_mass[10:15, 10:15, 1] = 7.0  # p = 0.7 on the whole footprint
    e = Entity(1, np.argwhere(np.ones((5, 5), bool)) + 10)
    assert label_score(m, [e]) == pytest.approx(0.3)
    m.n_obs[12, 12] = 0  # one footprint cell unobserved in the prediction
    assert label_score(m, [e]) == 2.0
