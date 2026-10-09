"""RGB-D SLAM with loop closures on top of the frame-to-frame odometry (Open3D pose graph).

The odometry chain (s2m.odometry) gives the initial poses. Keyframes are taken every `key_every`
odometry frames; thresholds were chosen on the development scene apt_0 only. Loop-closure candidates are
keyframe pairs far apart in time whose images match (ORB features, PnP on depth); each candidate is
verified by RGB-D odometry started from the PnP pose and by the overlap of the two point clouds after
alignment.
Verified loops enter the pose graph as certain edges (with trust_loops=False as uncertain ones that
Open3D's line process may prune). Every odometry frame then follows the
correction of its keyframe. No ground truth is used except the first pose (the map origin), as in
the odometry.

Open3D conventions: a PoseGraphNode holds a camera-to-world pose; an edge (s, t, T) states
p_t = T p_s in camera coordinates, i.e. T = pose_t^{-1} pose_s.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from s2m.data import Scene
from s2m.odometry import VOResult, _rgbd, planar_motion


@dataclass
class SlamStats:
    keyframes: int
    candidates: int
    verified: int
    kept: int


def _orb_features(scene: Scene, frame: int, intr, n: int = 1500):
    """ORB keypoints with valid depth: (descriptors, 2D pixels, 3D points in the camera frame)."""
    import cv2

    gray = cv2.cvtColor(np.ascontiguousarray(scene.rgb(frame)), cv2.COLOR_RGB2GRAY)
    kps, desc = cv2.ORB_create(n).detectAndCompute(gray, None)
    if desc is None:
        return None
    depth = scene.depth(frame)
    uv = np.array([k.pt for k in kps], np.float32)
    z = depth[np.clip(uv[:, 1].round().astype(int), 0, depth.shape[0] - 1),
              np.clip(uv[:, 0].round().astype(int), 0, depth.shape[1] - 1)]
    ok = (z > 0.2) & (z < 8.0)
    xyz = np.stack([(uv[:, 0] - intr.cx) * z / intr.fx, (uv[:, 1] - intr.cy) * z / intr.fy, z], 1)
    return desc[ok], uv[ok], xyz[ok].astype(np.float32)


def _pnp(fa, fb, intr, min_matches: int, min_inliers: int):
    """Relative pose T (p_b = T p_a) from ORB matches of keyframe a (3D) to keyframe b (2D), or None."""
    import cv2

    if fa is None or fb is None or len(fa[0]) < min_matches or len(fb[0]) < min_matches:
        return None
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(fa[0], fb[0])
    if len(matches) < min_matches:
        return None
    ia = np.array([m.queryIdx for m in matches])
    ib = np.array([m.trainIdx for m in matches])
    K = np.array([[intr.fx, 0, intr.cx], [0, intr.fy, intr.cy], [0, 0, 1]], np.float64)
    ok, rvec, tvec, inl = cv2.solvePnPRansac(fa[2][ia], fb[1][ib], K, None, reprojectionError=3.0,
                                             iterationsCount=200, flags=cv2.SOLVEPNP_EPNP)
    if not ok or inl is None or len(inl) < min_inliers:
        return None
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.ravel()
    return T, len(inl)


def chain_information(infos: np.ndarray) -> np.ndarray:
    """Information of a chain of odometry steps: the step covariances add up (a failed step, reported
    with zero information, counts with a weak prior of 0.1 rad / 0.1 m)."""
    cov = sum(np.linalg.inv(0.5 * (s + s.T) + np.eye(6) * 1e-6) if np.any(s) else np.eye(6) * 1e-2 for s in infos)
    info = np.linalg.inv(cov)
    return 0.5 * (info + info.T)


def _cloud(rgbd, pinhole, voxel: float):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole)
    return pcd.voxel_down_sample(voxel)


def loop_closure_slam(scene: Scene, vo: VOResult, key_every: int = 5, min_gap: int = 15, per_key: int = 2,
                      min_matches: int = 40, min_inliers: int = 25, min_fitness: float = 0.4, max_rmse: float = 0.03,
                      trust_loops: bool = True):
    """Pose-graph SLAM over the odometry `vo` (frames of `scene`).
    Returns (VOResult, SlamStats, accepted loops as (vo index a, vo index b, T_ab))."""
    import open3d as o3d

    reg = o3d.pipelines.registration
    intr = scene.intrinsics
    pinhole = o3d.camera.PinholeCameraIntrinsic(intr.width, intr.height, intr.fx, intr.fy, intr.cx, intr.cy)
    option = o3d.pipelines.odometry.OdometryOption()
    jacobian = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()

    keys = np.arange(0, len(vo.frames), key_every)
    P = vo.poses[keys]
    graph = reg.PoseGraph()
    for p in P:
        graph.nodes.append(reg.PoseGraphNode(p))

    # odometry edges: the chained motion between consecutive keyframes
    for a in range(len(keys) - 1):
        info = chain_information(vo.info[keys[a] + 1:keys[a + 1] + 1])
        graph.edges.append(reg.PoseGraphEdge(a, a + 1, np.linalg.inv(P[a + 1]) @ P[a], info, uncertain=False))

    rgbd_cache, cloud_cache = {}, {}

    def rgbd(k):
        if k not in rgbd_cache:
            rgbd_cache[k] = _rgbd(scene, int(vo.frames[keys[k]]))
        return rgbd_cache[k]

    def cloud(k):
        if k not in cloud_cache:
            cloud_cache[k] = _cloud(rgbd(k), pinhole, 0.03)
        return cloud_cache[k]

    # loop-closure candidates by appearance (ORB + PnP on depth), independent of the drifting estimate
    feats = [_orb_features(scene, int(vo.frames[k]), intr) for k in keys]
    cand = []
    for b in range(len(keys)):
        scored = []
        for a in range(0, b - min_gap + 1):
            T = _pnp(feats[a], feats[b], intr, min_matches, min_inliers)
            if T is not None:
                scored.append((T[1], a, T[0]))
        scored.sort(key=lambda x: -x[0])
        cand += [(a, b, T) for _, a, T in scored[:per_key]]

    verified = 0
    loops = []
    for a, b, guess in cand:
        ok, T, info = o3d.pipelines.odometry.compute_rgbd_odometry(rgbd(a), rgbd(b), pinhole, guess, jacobian, option)
        if not ok:
            continue
        T = np.linalg.inv(planar_motion(np.linalg.inv(np.asarray(T))))  # planar camera motion, as in the odometry
        ev = reg.evaluate_registration(cloud(a), cloud(b), 0.05, T)
        if ev.fitness < min_fitness or ev.inlier_rmse > max_rmse:
            continue
        verified += 1
        loops.append((a, b, T))
        info = np.asarray(info)
        graph.edges.append(reg.PoseGraphEdge(a, b, T, 0.5 * (info + info.T) + np.eye(6) * 1e-6, uncertain=not trust_loops))

    if verified:
        reg.global_optimization(
            graph, reg.GlobalOptimizationLevenbergMarquardt(), reg.GlobalOptimizationConvergenceCriteria(),
            reg.GlobalOptimizationOption(max_correspondence_distance=0.05, edge_prune_threshold=0.25,
                                         preference_loop_closure=1.0, reference_node=0))
    kept = sum(1 for e in graph.edges if e.source_node_id + 1 != e.target_node_id)
    newP = np.stack([np.asarray(n.pose) for n in graph.nodes])

    # every odometry frame follows the correction of its keyframe
    owner = np.minimum(np.arange(len(vo.frames)) // key_every, len(keys) - 1)
    corr = newP @ np.linalg.inv(P)
    poses = corr[owner] @ vo.poses
    return (VOResult(vo.frames, poses, vo.success, vo.info),
            SlamStats(len(keys), len(cand), verified, kept), [(int(keys[a]), int(keys[b]), T) for a, b, T in loops])
