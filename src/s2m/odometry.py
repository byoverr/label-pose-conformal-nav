"""Real pose-estimator error: frame-to-frame RGB-D visual odometry (Open3D) on the same frames.

The estimated trajectory starts at the ground-truth pose of the first frame and chains the
relative motions estimated between consecutive frames (no loop closure, no global optimisation).
When the solver reports failure the previous relative motion is reused (constant-velocity guess),
as a real dead-reckoning front end would do.

With planar=True every relative motion is projected onto the motions a ground robot with a level
camera can make (yaw about the camera's vertical axis, translation in its horizontal plane), the
standard planar constraint of wheeled-robot VO. Without it, a single bad step (e.g. a close-up
of a wall during an in-place turn) can tilt the whole remaining trajectory.

Open3D's compute_rgbd_odometry(source, target) returns T with p_target = T @ p_source in camera
coordinates, i.e. T = T_target^{-1} T_source, hence T_target = T_source @ T^{-1}.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from s2m.data import Scene


@dataclass
class VOResult:
    frames: np.ndarray  # (K,) frame ids the trajectory is defined on
    poses: np.ndarray  # (K, 4, 4) estimated camera-to-world poses
    success: np.ndarray  # (K,) bool, solver success for the step ending at this frame (True for the first)
    info: np.ndarray  # (K, 6, 6) information matrix reported for that step (zeros if failed)


def _rgbd(scene: Scene, i: int):
    import open3d as o3d

    color = o3d.geometry.Image(np.ascontiguousarray(scene.rgb(i)))
    depth = o3d.geometry.Image((scene.depth(i) * 1000.0).astype(np.uint16))  # millimetres
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, depth_scale=1000.0, depth_trunc=8.0, convert_rgb_to_intensity=True)


def planar_motion(M: np.ndarray) -> np.ndarray:
    """Project a camera-frame relative motion onto yaw about the camera y axis + x/z translation."""
    from s2m.drift import rot_y

    P = np.eye(4)
    P[:3, :3] = rot_y(np.arctan2(M[0, 2], M[0, 0]))
    P[[0, 2], 3] = M[[0, 2], 3]
    return P


def rgbd_odometry(scene: Scene, frames=None, planar: bool = True) -> VOResult:
    import open3d as o3d

    frames = np.array(scene.frame_ids() if frames is None else frames)
    intr = scene.intrinsics
    pinhole = o3d.camera.PinholeCameraIntrinsic(intr.width, intr.height, intr.fx, intr.fy, intr.cx, intr.cy)
    option = o3d.pipelines.odometry.OdometryOption()
    jacobian = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()

    poses = np.empty((len(frames), 4, 4))
    poses[0] = scene.poses[frames[0]]
    success = np.ones(len(frames), bool)
    info = np.zeros((len(frames), 6, 6))
    prev = _rgbd(scene, frames[0])
    last_motion = np.eye(4)
    for j in range(1, len(frames)):
        cur = _rgbd(scene, frames[j])
        ok, T, inf = o3d.pipelines.odometry.compute_rgbd_odometry(
            prev, cur, pinhole, np.linalg.inv(last_motion), jacobian, option)
        if ok:
            last_motion = np.linalg.inv(np.asarray(T))  # camera motion source -> target
            if planar:
                last_motion = planar_motion(last_motion)
            info[j] = np.asarray(inf)
        success[j] = ok
        poses[j] = poses[j - 1] @ last_motion
        prev = cur
    return VOResult(frames, poses, success, info)


def full_trajectory(gt: np.ndarray, vo: VOResult) -> np.ndarray:
    """(N, 4, 4) poses indexed by frame id: VO estimates on the VO frames, and for every other
    frame the ground-truth pose moved by the correction of the most recent VO frame."""
    D = vo.poses @ np.linalg.inv(gt[vo.frames])
    idx = np.searchsorted(vo.frames, np.arange(len(gt)), side="right") - 1
    idx = np.clip(idx, 0, len(vo.frames) - 1)
    return D[idx] @ gt


def save_vo(path, vo: VOResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, frames=vo.frames, poses=vo.poses, success=vo.success, info=vo.info)


def load_vo(path) -> VOResult:
    z = np.load(path)
    return VOResult(z["frames"], z["poses"], z["success"], z["info"])


def vo_poses(path, gt: np.ndarray) -> np.ndarray:
    """Estimated poses the robot maps with: cached VO, all frames, floor-plane error only."""
    return planarize(full_trajectory(gt, load_vo(path)), gt)


def planarize(est: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Keep only the floor-plane part of the estimator's error: camera x, z and heading.

    A ground robot knows gravity (IMU) and its camera height, so it would not map with the roll,
    pitch and height errors of a free 6-DoF VO; the planar part is what stays unobservable.
    Returns poses P_k T_k with P_k a rotation about the vertical plus a floor-plane shift, chosen
    so that the camera lands at the estimated (x, z) with the estimated heading.
    """
    from s2m.drift import rot_y

    out = np.empty_like(gt)
    for k, (E, T) in enumerate(zip(est, gt)):
        R = E[:3, :3] @ T[:3, :3].T  # world-frame rotation error
        yaw = np.arctan2(-R[2, 0], R[0, 0])  # its component about the vertical (world y)
        P = np.eye(4)
        P[:3, :3] = rot_y(yaw)
        P[:3, 3] = -P[:3, :3] @ T[:3, 3]
        P[[0, 2], 3] += E[[0, 2], 3]
        P[1, 3] += T[1, 3]
        out[k] = P @ T
    return out
