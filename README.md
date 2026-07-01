# Biohub Cell Tracking During Development

This repository tracks experiments for the Kaggle competition
Biohub - Cell Tracking During Development.

The task is to detect cells in 3D time-lapse microscopy volumes and link them
across time into lineage graphs. Submissions write a single `submission.csv`
with node rows for cell centers and edge rows for temporal links.

## Repository Contents

- `kaggle/`: Kaggle notebook entries used for submitted experiments.
- `scripts/validate_submission.py`: structural validation for generated
  submission files.
- `docs/experiment_log.md`: submitted runs and leaderboard results.

Large competition data, generated outputs, and downloaded reference notebooks
are intentionally excluded from the public repository.

## Current Experiments

The first batch uses classical 3D peak detection and physical-distance linking:

| Experiment | Method | Main change |
|---|---|---|
| exp001 | Classical baseline | Moderate local maxima with two-pass Hungarian linking |
| exp002 | DoG band-pass | Difference-of-Gaussians candidate detection |
| exp003 | More nodes | Lower threshold and smaller non-maximum suppression radius |
| exp004 | Conservative | Higher threshold and larger suppression radius |
| exp005 | Link12 | Wider physical linking gate with gap closing enabled |

## Validation

```bash
python scripts/validate_submission.py /path/to/submission.csv --test-dir /path/to/test
```

The validator checks column order, row invariants, integer fields, node-id
references, and dataset coverage when a test directory is supplied.
