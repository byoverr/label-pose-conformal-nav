# Pose-drift protocol

OSMa-Bench publishes ground-truth poses only, so pose error is injected synthetically.
The model and its parameters are fixed before any calibration result is seen.

## Model (`src/s2m/drift.py`)

Dead reckoning with noise proportional to motion, as in odometry motion models
(e.g. AMCL's `alpha` parameters):

```
T_hat_k = T_hat_{k-1} · Δ_k · N_k,    Δ_k = T_{k-1}^{-1} T_k,    T_hat_0 = T_0
N_k = Rot_y(δψ) · Trans(δx, 0, δz)
δx, δz ~ N(0, (σ_t · |Δt_k|)²)
δψ     ~ N(0, (σ_ψr · |Δψ_k| + σ_ψt · |Δt_k|)²)
```

ReplicaCAD cameras are level (camera y axis = world −y, checked for every frame), so the noise is
applied in the camera frame and the error stays planar: x, z and yaw; height, roll and pitch are exact.

The drifted map places a point seen at frame k at `D_k p` with `D_k = T_hat_k T_k^{-1}`; the planner
works in the drifted frame at the query time q (last frame), where the true point is at `D_q p`.

## Levels (`configs/drift_levels.yaml`)

The noise *shape* is fixed (σ_t : σ_ψr : σ_ψt = 0.05 : 0.05 : 0.01); one multiplier per level is
calibrated once on the dev scene `apt_0` (path 33.7 m, 32 seeds) and then applied unchanged to every
scene, because it describes the pose estimator, not the scene.

| Level | Target | Achieved on apt_0 (mean of 16 seeds) | What it imitates |
|---|---|---|---|
| L0 | 0 | 0 | ground-truth poses |
| L1 | ATE ≈ 0.7 cm | 0.6 cm | dense RGB-D SLAM on synthetic scenes (Replica ATE 0.4–1 cm) |
| L2 | ATE ≈ 3.5 cm | 3.0 cm | real RGB-D SLAM (TUM ATE ≈ 2–5 cm) |
| L3 | ATE ≈ 11 cm | 9.5 cm | hard scenes (ScanNet ATE ≈ 10–12 cm) |
| L4 | end drift ≈ 3 % of path | 2.8 % (ATE 63 cm) | odometry without loop closure |
| L5 | end drift ≈ 17 % of path | 16.4 % (ATE 3.5 m) | stress test (stereo VIO without loop closure) |

Reference magnitudes come from the literature survey in the companion report (SplaTAM Table 1 for
Replica/TUM/ScanNet ATE; ZED VIO and quadruped odometry drift for L4–L5).

## Not modelled (limitations)

- loop closures and re-anchoring of map objects;
- correlation between pose error and scene appearance (e.g. more drift in texture-poor areas);
- execution-time localization error while following the plan (the guarantee is in the planner frame
  at query time).
