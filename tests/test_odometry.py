import numpy as np

from s2m.drift import DriftModel, rot_y, simulate, world_correction
from s2m.odometry import VOResult, planarize
from tests.test_drift import straight_then_turn


def tilt(angle: float) -> np.ndarray:
    """Rotation about the world x axis (a pitch error a free 6-DoF estimate can have)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def test_planarize_keeps_planar_errors_exactly():
    gt = straight_then_turn()
    est = simulate(gt, DriftModel(0.1, 0.1, 0.05), np.random.default_rng(3))
    assert np.allclose(planarize(est, gt), est, atol=1e-9)


def test_planarize_removes_height_and_tilt_but_keeps_position_and_heading():
    gt = straight_then_turn()
    planar = simulate(gt, DriftModel(0.1, 0.1, 0.05), np.random.default_rng(4))
    E = np.eye(4)
    E[:3, :3] = tilt(0.05)
    est = E @ planar
    est[:, 1, 3] += 0.2  # height error
    out = planarize(est, gt)
    assert np.allclose(out[:, 1, 3], gt[:, 1, 3])  # true height back
    assert np.allclose(out[:, :3, 1], gt[:, :3, 1], atol=1e-9)  # level again
    assert np.allclose(out[:, [0, 2], 3], est[:, [0, 2], 3])  # estimated floor position kept
    D = world_correction(out, gt)
    assert np.allclose(D[:, 1, :], [0, 1, 0, 0], atol=1e-9)  # a planar correction


def test_full_trajectory_holds_last_keyframe_correction():
    from s2m.odometry import full_trajectory

    gt = straight_then_turn(n=20)
    frames = np.array([0, 4, 8, 12, 16])
    est_kf = gt[frames].copy()
    est_kf[2:, 0, 3] += 0.5  # VO jumps by 0.5 m in x from keyframe 8 on
    full = full_trajectory(gt, VOResult(frames, est_kf, np.ones(5, bool), np.zeros((5, 6, 6))))
    assert np.allclose(full[:8], gt[:8])
    assert np.allclose(full[8:, 0, 3], gt[8:, 0, 3] + 0.5)


def test_rot_y_yaw_extraction_used_by_planarize():
    R = rot_y(0.3)
    assert np.isclose(np.arctan2(-R[2, 0], R[0, 0]), 0.3)


def test_planar_motion_keeps_yaw_and_floor_translation():
    from s2m.odometry import planar_motion

    M = np.eye(4)
    M[:3, :3] = rot_y(0.2)
    M[:3, 3] = [0.1, 0.0, -0.3]
    assert np.allclose(planar_motion(M), M)
    N = M.copy()
    N[:3, :3] = tilt(0.1) @ M[:3, :3]
    N[1, 3] = 0.05
    out = planar_motion(N)
    assert np.allclose(out[:3, 1], [0, 1, 0]) and out[1, 3] == 0.0
