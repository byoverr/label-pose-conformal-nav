# calibrated-semantic-nav

**Work in progress (BE2R test task, Oct 2026).**

How do errors of open-vocabulary semantic maps and of robot pose turn into navigation failures —
and can a planner get **one calibrated guarantee** that covers both?

- Data: [OSMa-Bench](https://github.com/be2rlab/OSMa-Bench) (ReplicaCAD scenes, CC BY 4.0).
- Plan: map building → pose drift injection → reach-avoid planning → conformal calibration
  (label-only vs separate label+pose vs joint "miss distance").

Decision log: [`docs/journal.md`](docs/journal.md).

## Setup

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .
.venv/bin/python scripts/download_subset.py apt_0 --step 8
.venv/bin/python scripts/check_poses.py data/replica_cad/apt_0
```
