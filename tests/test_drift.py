import numpy as np
import pytest

from s2m.drift import (DriftModel, ate, calibrate_level, endpoint_drift, rot_y, simulate,
                       world_correction)


def straight_then_turn(n=200, step=0.1):
    """Level camera moving forward along -z, then turning in place."""
    poses = []
    for k in range(n):
        T = np.eye(4)
        T[:3, :3] = rot_y(0.0 if k < n // 2 else 0.02 * (k - n // 2)) @ np.diag([1.0, -1.0, -1.0])
        T[:3, 3] = [0.0, 1.5, -step * min(k, n // 2)]
        poses.append(T)
    return np.array(poses)


def test_zero_noise_reproduces_ground_truth():
    gt = straight_then_turn()
    est = simulate(gt, DriftModel(), np.random.default_rng(0))
    assert np.allclose(est, gt, atol=1e-9)
    assert ate(est, gt) == pytest.approx(0.0, abs=1e-9)


def test_drift_is_planar_and_rigid():
    gt = straight_then_turn()
    est = simulate(gt, DriftModel(0.1, 0.1, 0.05), np.random.default_rng(1))
    assert np.allclose(est[:, 1, 3], gt[:, 1, 3])  # height untouched
    assert np.allclose(est[:, :3, 1], gt[:, :3, 1])  # cameras stay level
    D = world_correction(est, gt)
    assert np.allclose(D[:, 1, :], [0, 1, 0, 0], atol=1e-9)  # correction keeps y
    for R in D[:, :3, :3]:
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)


def test_scale_bias_shortens_path():
    gt = straight_then_turn()
    est = simulate(gt, DriftModel(scale_bias=-0.1), np.random.default_rng(0))
    assert endpoint_drift(est, gt) == pytest.approx(0.1, rel=1e-6)  # straight segment only


def test_calibrate_level_hits_target_ate():
    gt = straight_then_turn()
    model = calibrate_level(gt, target=0.05, metric="ate", seeds=64)
    errs = [ate(simulate(gt, model, np.random.default_rng(100 + s)), gt) for s in range(64)]
    assert np.mean(errs) == pytest.approx(0.05, rel=0.15)
