"""Synthetic odometry drift on the floor plane.

The estimated trajectory is built like dead reckoning: every relative motion step of the
ground-truth trajectory is replayed with multiplicative noise,
    T_hat_k = T_hat_{k-1} @ Delta_k @ N_k,   Delta_k = T_{k-1}^{-1} T_k,   T_hat_0 = T_0,
where N_k is a small rotation about the vertical axis plus a translation in the floor plane,
both proportional to the step size (as in odometry motion models, e.g. AMCL's alpha model).
Height, roll and pitch stay exact, so the error is planar: x, z and yaw.

Because ReplicaCAD cameras are level (camera y axis = world -y), "vertical" in the camera
frame is the camera y axis and the noise can be applied directly in the camera frame.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class DriftModel:
    sigma_t: float = 0.0  # translation noise std per metre travelled (per axis)
    sigma_yaw_rot: float = 0.0  # yaw noise std per radian turned
    sigma_yaw_trans: float = 0.0  # yaw noise std per metre travelled (rad / m)
    scale_bias: float = 0.0  # systematic under/over-estimation of distance travelled

    def scaled(self, m: float) -> "DriftModel":
        return replace(self, sigma_t=self.sigma_t * m, sigma_yaw_rot=self.sigma_yaw_rot * m,
                       sigma_yaw_trans=self.sigma_yaw_trans * m)


# Noise shape only; the overall magnitude is set per level by `calibrate_level`.
BASE_MODEL = DriftModel(sigma_t=0.05, sigma_yaw_rot=0.05, sigma_yaw_trans=0.01)


def rot_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def yaw_of(R: np.ndarray) -> float:
    """Rotation angle about the y axis of a (near) pure y-rotation."""
    return float(np.arctan2(R[0, 2], R[0, 0]))


def check_level(poses: np.ndarray, tol: float = 1e-3) -> None:
    """Assert that every camera is level (its y axis is world -y)."""
    dev = np.abs(poses[:, :3, 1] - np.array([0.0, -1.0, 0.0])).max()
    if dev > tol:
        raise ValueError(f"cameras are not level (max deviation {dev:.2e}); planar drift model invalid")


def simulate(poses: np.ndarray, model: DriftModel, rng: np.random.Generator) -> np.ndarray:
    """Return drifted camera-to-world poses with the same shape as `poses` (N, 4, 4)."""
    check_level(poses)
    est = np.empty_like(poses)
    est[0] = poses[0]
    for k in range(1, len(poses)):
        delta = np.linalg.inv(poses[k - 1]) @ poses[k]
        delta[:3, 3] *= 1.0 + model.scale_bias
        dist = float(np.linalg.norm(delta[:3, 3]))
        dyaw = abs(yaw_of(delta[:3, :3]))
        noise = np.eye(4)
        # In the camera frame the vertical axis is y, so a yaw is a rotation about y.
        noise[:3, :3] = rot_y(rng.normal(0.0, model.sigma_yaw_rot * dyaw + model.sigma_yaw_trans * dist))
        noise[[0, 2], 3] = rng.normal(0.0, model.sigma_t * dist, size=2)  # camera x and z
        est[k] = est[k - 1] @ delta @ noise
    return est


def world_correction(est: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """D_k = T_hat_k T_k^{-1}: maps the true world position of a point seen at frame k
    to where the drifted map puts it. Planar rigid transforms, shape (N, 4, 4)."""
    return est @ np.linalg.inv(gt)


def ate(est: np.ndarray, gt: np.ndarray) -> float:
    """Absolute trajectory error (RMSE of floor-plane position), first-frame aligned."""
    d = est[:, [0, 2], 3] - gt[:, [0, 2], 3]
    return float(np.sqrt((d ** 2).sum(axis=1).mean()))


def path_length(poses: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(poses[:, [0, 2], 3], axis=0), axis=1).sum())


def endpoint_drift(est: np.ndarray, gt: np.ndarray) -> float:
    """Final floor-plane position error as a fraction of the distance travelled."""
    return float(np.linalg.norm(est[-1, [0, 2], 3] - gt[-1, [0, 2], 3]) / max(path_length(gt), 1e-9))


def calibrate_level(poses: np.ndarray, target: float, metric: str = "ate",
                    base: DriftModel = BASE_MODEL, seeds: int = 32) -> DriftModel:
    """Scale `base` so that the mean of `metric` ('ate' in metres or 'endpoint' as a fraction
    of path length) over `seeds` simulations matches `target`.

    For small noise both metrics grow linearly with the noise scale, so one probe run plus a
    single correction step is enough; we refine once more to absorb the residual non-linearity.
    """
    fn = {"ate": ate, "endpoint": endpoint_drift}[metric]

    def mean_metric(m: DriftModel) -> float:
        return float(np.mean([fn(simulate(poses, m, np.random.default_rng(s)), poses) for s in range(seeds)]))

    model = base
    for _ in range(2):
        model = model.scaled(target / mean_metric(model))
    return model
