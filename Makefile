# Reproduce every number in the article and the paper. Run from the repository root.
# Heavy steps are resumable: finished scenes are skipped when a target is re-run.
#
#   make setup         virtual environment with pinned dependencies
#   make main          data, detector, calibration scores, coverage, missions   (~3 h on a laptop)
#   make comparison    closed-vocabulary comparison with per-cell label calibration
#   make robustness    risk-level sweep, composition stress test, SAM, visual odometry, HM3D, grid
#   make docs          tables, paper_ru.pdf (Russian article) and paper.pdf (English) (needs tectonic)
#   make test          unit tests

PY ?= .venv/bin/python
SCENES := $(shell cat configs/scenes_replicacad.txt)
HM3D := 00800-TEEsavR23oF 00802-wcojb4TFT35 00808-y9hTuugGdiq 00813-svBbv1Pavdk 00814-p53SfW6mjZe 00815-h1zeeAwLh9Z
PB := --avoid indoor_plant bike
CLOSED5 := indoor_plant bike tv_stand sofa table

.PHONY: setup data scores coverage missions main comparison alpha composition sam vo hm3d grid \
        detector yoloe regions risk ensemble slam execution cascade mission_risk robustness tables docs test all

setup:
	python3.12 -m venv .venv
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install -e .

# --- main experiment ------------------------------------------------------------------------
data:
	$(PY) scripts/download_subset.py $(SCENES) --step 8

scores: data
	$(PY) scripts/run_detector.py data/replica_cad/*
	$(PY) scripts/compute_scores.py data/replica_cad/* --seeds 10

coverage:
	$(PY) scripts/analyze_coverage.py
	$(PY) scripts/coverage_checks.py

missions:
	$(PY) scripts/scene_metrics.py
	$(PY) scripts/run_missions.py $(PB) --out results/missions_pb
	$(PY) scripts/analyze_missions.py --dir results/missions_pb --tag _pb
	$(PY) scripts/run_missions.py --kind random $(PB) --out results/missions_rand
	$(PY) scripts/analyze_missions.py --dir results/missions_rand --tag _rand --no-h4
	$(PY) scripts/run_missions.py --out results/missions          # all three avoid classes
	$(PY) scripts/analyze_missions.py --dir results/missions
	$(PY) scripts/plot_example.py v3_sc0_staging_20 --seed 1 --task 1
	$(PY) scripts/plot_example.py v3_sc0_staging_20 --seed 1 --task 1 --paper

main: scores coverage missions

# --- comparison with per-cell label calibration in its setting --------------------------------
comparison:
	$(PY) scripts/run_detector.py data/replica_cad/* --closed $(CLOSED5) --detections data/cache/detections_closed5
	$(PY) scripts/compute_scores.py data/replica_cad/* --detections data/cache/detections_closed5 --out results/scores_closed5
	$(PY) scripts/analyze_coverage.py --scores results/scores_closed5 --alpha 0.05 --tag _closed5_a005
	$(PY) scripts/analyze_coverage.py --scores results/scores_closed5 --tag _closed5
	$(PY) scripts/analyze_coverage.py --alpha 0.05 --tag _a005
	$(PY) scripts/run_missions.py --kind random --alpha 0.05 --scores results/scores_closed5 \
		--detections data/cache/detections_closed5 --only L0 L2 L3 $(PB) --out results/missions_rand_closed5_a005
	$(PY) scripts/run_missions.py --kind random --scores results/scores_closed5 \
		--detections data/cache/detections_closed5 --only L0 L2 L3 $(PB) --out results/missions_rand_closed5
	$(PY) scripts/run_missions.py --kind random --alpha 0.05 --only L0 L2 L3 $(PB) --out results/missions_rand_a005
	$(PY) scripts/run_missions.py --scores results/scores_closed5 --detections data/cache/detections_closed5 \
		--only L0 L2 L3 --out results/missions_closed5     # all three avoid classes, closed vocabulary
	$(PY) scripts/analyze_missions.py --dir results/missions_closed5 --tag _closed5 --levels L0 L2 L3 --no-h4
	$(PY) scripts/compare_known_pose.py

# --- robustness checks ------------------------------------------------------------------------
alpha:
	for a in 05 2 3; do \
		$(PY) scripts/analyze_coverage.py --alpha 0.$$a --tag _a0$$a && \
		$(PY) scripts/run_missions.py --alpha 0.$$a $(PB) --out results/missions_pb_a0$$a ; \
	done
	$(PY) scripts/run_missions.py --kind random --alpha 0.3 $(PB) --out results/missions_rand_a03
	$(PY) scripts/analyze_alpha.py --level L3

composition:
	$(PY) scripts/composition_stress.py --alpha 0.3 --fail-level L5 --splits 1000 --random-draws 20

sam:
	$(PY) scripts/run_segmenter.py data/replica_cad/*
	$(PY) scripts/compute_scores.py data/replica_cad/* --masks data/cache/masks_sam --out results/scores_sam
	$(PY) scripts/run_missions.py --scores results/scores_sam --masks data/cache/masks_sam $(PB) --out results/missions_pb_sam
	$(PY) scripts/analyze_missions.py --dir results/missions_pb_sam --tag _pb_sam --no-h4

vo:
	$(PY) scripts/download_subset.py $(SCENES) --step 2 --rgbd-only --workers 24 --out data/replica_cad_vo
	$(PY) scripts/run_vo.py data/replica_cad/*
	$(PY) scripts/run_missions.py --scores results/scores results/scores_vo --vo 2 4 --only L0 VO2 VO4 \
		$(PB) --out results/missions_pb_vo
	$(PY) scripts/analyze_missions.py --dir results/missions_pb_vo --tag _pb_vo --levels L0 VO2 VO4 \
		--scores results/scores results/scores_vo --no-h4

hm3d:
	$(PY) scripts/download_subset.py $(HM3D) --step 8 --root data/hm3d --out data/hm3d
	$(PY) scripts/run_hm3d.py data/hm3d/*

grid:
	$(PY) scripts/grid_resolution.py --res 0.05 0.25 0.5 1.0

detector:  # the largest YOLO-World v2 model instead of v2-s
	$(PY) scripts/run_detector.py data/replica_cad/* --weights models/yolov8x-worldv2.pt --detections data/cache/detections_x
	$(PY) scripts/compute_scores.py data/replica_cad/* --detections data/cache/detections_x --out results/scores_x
	$(PY) scripts/analyze_coverage.py --scores results/scores_x --tag _x
	$(PY) scripts/run_missions.py --scores results/scores_x --detections data/cache/detections_x $(PB) --out results/missions_pb_x
	$(PY) scripts/analyze_missions.py --dir results/missions_pb_x --tag _pb_x --no-h4
	$(PY) scripts/run_missions.py --kind random --scores results/scores_x --detections data/cache/detections_x $(PB) \
		--out results/missions_rand_x
	$(PY) scripts/analyze_missions.py --dir results/missions_rand_x --tag _rand_x --no-h4

yoloe:  # YOLOE-v8-S instead of YOLO-World v2-s (needs models/mobileclip_blt.ts once to embed the prompts)
	$(PY) scripts/run_detector.py data/replica_cad/* --weights models/yoloe-v8s-seg.pt --detections data/cache/detections_yoloe
	$(PY) scripts/compute_scores.py data/replica_cad/* --detections data/cache/detections_yoloe --out results/scores_yoloe
	$(PY) scripts/analyze_coverage.py --scores results/scores_yoloe --tag _yoloe
	$(PY) scripts/run_missions.py --scores results/scores_yoloe --detections data/cache/detections_yoloe $(PB) --out results/missions_pb_yoloe
	$(PY) scripts/analyze_missions.py --dir results/missions_pb_yoloe --tag _pb_yoloe --no-h4
	$(PY) scripts/run_missions.py --kind random --scores results/scores_yoloe --detections data/cache/detections_yoloe $(PB) \
		--out results/missions_rand_yoloe
	$(PY) scripts/analyze_missions.py --dir results/missions_rand_yoloe --tag _rand_yoloe --no-h4

regions:  # class regions grown into adjacent occupied cells
	$(PY) scripts/extra_scores.py data/replica_cad/*
	$(PY) scripts/coverage_checks.py --only regions

risk:  # risk as an output instead of abstention
	$(PY) scripts/run_risk.py --kind pass_by
	$(PY) scripts/run_risk.py --kind random
	$(PY) scripts/analyze_risk.py

ENS := --detections data/cache/detections data/cache/detections_x data/cache/detections_yoloe

ensemble:  # the three detectors fused into one map (needs make detector yoloe); every fusion rule is reported
	for rule in max mean vote2; do \
		$(PY) scripts/compute_scores.py data/replica_cad/* $(ENS) --fuse $$rule --out results/scores_ens_$$rule && \
		$(PY) scripts/analyze_coverage.py --scores results/scores_ens_$$rule --tag _ens_$$rule && \
		$(PY) scripts/run_missions.py --scores results/scores_ens_$$rule $(ENS) --fuse $$rule $(PB) --out results/missions_pb_ens_$$rule && \
		$(PY) scripts/analyze_missions.py --dir results/missions_pb_ens_$$rule --tag _pb_ens_$$rule --no-h4 && \
		$(PY) scripts/run_missions.py --kind random --scores results/scores_ens_$$rule $(ENS) --fuse $$rule $(PB) \
			--out results/missions_rand_ens_$$rule && \
		$(PY) scripts/analyze_missions.py --dir results/missions_rand_ens_$$rule --tag _rand_ens_$$rule --no-h4 || exit 1; \
	done

slam:  # loop-closure SLAM over the cached odometry (needs make vo)
	$(PY) scripts/run_slam.py data/replica_cad/*
	$(PY) scripts/run_missions.py --scores results/scores results/scores_vo results/scores_slam --vo 2 --slam 2 \
		--only L0 VO2 SLAM2 $(PB) --out results/missions_pb_slam
	$(PY) scripts/analyze_missions.py --dir results/missions_pb_slam --tag _pb_slam --levels L0 VO2 SLAM2 \
		--scores results/scores results/scores_vo results/scores_slam --no-h4
	$(PY) scripts/run_slam.py data/replica_cad/* $(ENS) --fuse max --out results/scores_slam_ens_max
	$(PY) scripts/run_missions.py --scores results/scores_ens_max results/scores_slam_ens_max --slam 2 --only L0 SLAM2 \
		$(ENS) --fuse max $(PB) --out results/missions_pb_slam_ens_max
	$(PY) scripts/analyze_missions.py --dir results/missions_pb_slam_ens_max --tag _pb_slam_ens_max --levels L0 SLAM2 \
		--scores results/scores_ens_max results/scores_slam_ens_max --no-h4

execution:  # certified paths executed under continued drift
	$(PY) scripts/run_risk.py --kind pass_by --only L2 L3 L4 --exec-draws 20
	$(PY) scripts/run_risk.py --kind random --only L2 L3 L4 --exec-draws 20
	$(PY) scripts/analyze_execution.py

cascade:  # union of the three detectors checked by CLIP ViT-B/16 (models/clip/, downloaded on first use)
	$(PY) scripts/run_verifier.py data/replica_cad/*
	$(PY) scripts/select_cascade.py
	$(PY) scripts/check_frame_vote.py
	$(PY) scripts/make_cascade.py data/replica_cad/*
	$(PY) scripts/compute_scores.py data/replica_cad/* --detections data/cache/detections_cascade --out results/scores_cascade
	$(PY) scripts/analyze_coverage.py --scores results/scores_cascade --tag _cascade
	$(PY) scripts/run_missions.py --scores results/scores_cascade --detections data/cache/detections_cascade $(PB) \
		--out results/missions_pb_cascade
	$(PY) scripts/analyze_missions.py --dir results/missions_pb_cascade --tag _pb_cascade --no-h4
	$(PY) scripts/run_missions.py --kind random --scores results/scores_cascade --detections data/cache/detections_cascade \
		$(PB) --out results/missions_rand_cascade
	$(PY) scripts/analyze_missions.py --dir results/missions_rand_cascade --tag _rand_cascade --no-h4

mission_risk: risk execution slam ensemble  # conformal risk control on the planner (mission-level guarantee)
	for kind in pass_by random; do \
		$(PY) scripts/run_risk.py --kind $$kind --scores results/scores_slam --slam data/cache/slam \
			--out results/risk_slam_$$(test $$kind = pass_by && echo pb || echo rand) && \
		$(PY) scripts/run_risk.py --kind $$kind --scores results/scores_ens_max \
			--detections-extra data/cache/detections_x data/cache/detections_yoloe \
			--out results/risk_ens_$$(test $$kind = pass_by && echo pb || echo rand) || exit 1; \
	done
	$(PY) scripts/analyze_crc.py
	$(PY) scripts/analyze_crc.py --execution
	$(PY) scripts/analyze_crc.py --root risk_slam
	$(PY) scripts/analyze_crc.py --root risk_ens

robustness: alpha composition sam vo hm3d grid detector yoloe regions risk ensemble slam execution cascade mission_risk
	$(PY) scripts/analyze_variants.py

# --- documents ---------------------------------------------------------------------------------
tables:
	$(PY) scripts/make_tables.py
	$(PY) scripts/plot_figures_ru.py
	$(PY) scripts/plot_example.py v3_sc0_staging_20 --seed 1 --task 1 --paper --lang ru

docs: tables
	cd paper && tectonic paper_ru.tex
	cd paper && tectonic paper.tex

test:
	$(PY) -m pytest -q

all: main comparison robustness docs
