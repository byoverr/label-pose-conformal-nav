# Reproduce every number in the paper and the report. Run from the repository root.
# Heavy steps are resumable: finished scenes are skipped when a target is re-run.
#
#   make setup         virtual environment with pinned dependencies
#   make main          data, detector, calibration scores, coverage, missions   (~3 h on a laptop)
#   make comparison    closed-vocabulary comparison with per-cell label calibration
#   make robustness    risk-level sweep, composition stress test, SAM, visual odometry, HM3D, grid
#   make docs          report tables, report.pdf and paper.pdf (needs tectonic)
#   make test          unit tests

PY ?= .venv/bin/python
SCENES := $(shell cat configs/scenes_replicacad.txt)
HM3D := 00800-TEEsavR23oF 00802-wcojb4TFT35 00808-y9hTuugGdiq 00813-svBbv1Pavdk 00814-p53SfW6mjZe 00815-h1zeeAwLh9Z
PB := --avoid indoor_plant bike
CLOSED5 := indoor_plant bike tv_stand sofa table

.PHONY: setup data scores coverage missions main comparison alpha composition sam vo hm3d grid \
        robustness tables docs test all

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
	$(PY) scripts/run_missions.py --kind random --alpha 0.05 --scores results/scores_closed5 \
		--detections data/cache/detections_closed5 --only L0 L2 L3 $(PB) --out results/missions_rand_closed5_a005
	$(PY) scripts/compare_known_pose.py

# --- robustness checks ------------------------------------------------------------------------
alpha:
	for a in 05 2 3; do \
		$(PY) scripts/analyze_coverage.py --alpha 0.$$a --tag _a0$$a && \
		$(PY) scripts/run_missions.py --alpha 0.$$a $(PB) --out results/missions_pb_a0$$a ; \
	done
	$(PY) scripts/analyze_alpha.py --level L3

composition:
	$(PY) scripts/composition_stress.py --alpha 0.3 --fail-level L5

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

robustness: alpha composition sam vo hm3d grid
	$(PY) scripts/analyze_variants.py

# --- documents ---------------------------------------------------------------------------------
tables:
	$(PY) scripts/make_report_tables.py

docs: tables
	cd report && tectonic report.tex
	cd paper && tectonic paper.tex

test:
	$(PY) -m pytest -q

all: main comparison robustness docs
