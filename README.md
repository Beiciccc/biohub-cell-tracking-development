# Biohub Cell Tracking During Development

This repository tracks experiments for the Kaggle competition
Biohub - Cell Tracking During Development.

The task is to detect cells in 3D time-lapse microscopy volumes and link them
across time into lineage graphs. Submissions write a single `submission.csv`
with node rows for cell centers and edge rows for temporal links.

## Repository Contents

- `kaggle/`: Kaggle notebook and script entries used for submitted experiments.
- `scripts/validate_submission.py`: structural validation for generated
  submission files.
- `docs/experiment_log.md`: submitted runs and leaderboard results.

Large competition data, generated outputs, and downloaded reference notebooks
are intentionally excluded from the public repository.

## Current Experiments

Submitted experiments cover classical, UNet-based, rule-based, and learned
graph tracking:

| Experiment | Method | Main change |
|---|---|---|
| exp001 | Classical baseline | Moderate local maxima with two-pass Hungarian linking |
| exp002 | UNet + ILP | Public artifact model with threshold 0.99 and ILP linking |
| exp003 | UNet + ILP geometry prune | Threshold 0.995, division weight 0.8, geometry cleanup |
| exp004 | UNet + ILP edge -0.8 | Threshold 0.995, division weight 0.8, geometry cleanup |
| exp005 | UNet + ILP edge -0.5 | Threshold 0.995, division weight 0.8, geometry cleanup |
| exp006-rule | Rule-based safe division recovery | Blob detector baseline with guarded high-confidence division edges |
| exp006-edge065 | UNet + ILP edge -0.65 | Threshold 0.995, division weight 0.8, edge weight -0.65 |
| exp007 | Rule-based safe-plus | Runtime exceeded during submission scoring |
| exp008 | Rule-based safe-tiny | Runtime exceeded during submission scoring |
| exp010 | Rule-based fast safe-plus | Runtime-safe safe division widening |
| exp016 | Public CV6 rule baseline | Two-pass velocity linking with gap closing and short-track filtering |
| exp017 | Public CV6 link 8.5 | Single-parameter wider loose link gate |
| exp018 | Public CV6 gap 5.5 | Tighter one-frame gap-closing distance |
| exp021 | Public CV6 gap 5.25 | Further tightened one-frame gap-closing distance |
| exp022 | Public CV6 gap 5.0 | Most restrictive tested one-frame gap-closing distance |
| exp023 | Rule-based v10 | Multi-scale DoG tracking with line smoothing, two-frame gap recovery, and safe division edges |
| exp024 | Rule-based v10 threshold 0.032 | Slightly higher DoG relative threshold to reduce weak false peaks |
| exp025 | Rule-based v10 threshold 0.032 gap 5.5 | Tighter one-frame gap-closing distance on the threshold 0.032 v10 tracker |
| exp026 | Rule-based v10 threshold 0.032 gap 5.25 | Further tightened one-frame gap-closing distance on the threshold 0.032 v10 tracker |
| exp027 | Rule-based v10 threshold 0.033 gap 5.5 | Higher DoG relative threshold on the 5.5 um gap-distance v10 tracker |
| exp028 | Learned graph tracker | GPU learned-edge model with ILP linking and conservative edge-consensus pruning |
| exp029 | Learned graph tracker prune50 | More conservative edge-consensus preset on the exp028 learned graph pipeline |
| exp030 | Learned graph tracker prune40 | Latest conservative edge-consensus preset from the learned graph pipeline |
| exp031 | Pilkwang precision repair | Independent learned graph precision-repair branch with threshold 0.992 |
| exp032 | Pilkwang recall clean | Recall-oriented learned graph branch with threshold 0.985 and two-step gap recovery |
| exp033 | Yusuke score push | Learned graph branch with threshold 0.99, gap recovery, and tightened division caps |

## Validation

```bash
python scripts/validate_submission.py /path/to/submission.csv --test-dir /path/to/test
```

The validator checks column order, row invariants, integer fields, node-id
references, and dataset coverage when a test directory is supplied.
