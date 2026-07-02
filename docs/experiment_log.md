# Experiment Log

Competition: Biohub - Cell Tracking During Development

Daily submission budget checked on 2026-07-01: 5 submissions. The official
submission list was empty before the first batch. The 2026-07-02 batch started
with five remaining daily submissions in the official submission table.

| ID | Kernel | Method | Status | Public score | Notes |
|---|---|---|---|---|---|
| exp001 | `beicicc/biohub-cell-tracking-exp001-classical` | Classical local maxima + Hungarian linking | complete | 0.750 | Stable baseline |
| exp002 | `beicicc/biohub-cell-tracking-exp002-unet-ilp-099` | UNet + transformer + ILP, threshold 0.99 | complete | 0.810 | Public artifact model anchor |
| exp003 | `beicicc/biohub-cell-tracking-exp003-unet-geom-prune` | UNet + ILP, threshold 0.995, division weight 0.8, geometry cleanup | complete | 0.812 | Tests false-edge pruning |
| exp004 | `beicicc/biohub-cell-tracking-exp004-unet-raw081` | UNet + ILP, threshold 0.995, division weight 0.8, edge weight -0.8, geometry cleanup | complete | 0.818 | Tests lower edge reward |
| exp005 | `beicicc/biohub-cell-tracking-exp005-unet-mild-prune` | UNet + ILP, threshold 0.995, division weight 0.8, edge weight -0.5, geometry cleanup | complete | 0.818 | Tests still lower edge reward |
| exp006-rule | `beicicc/biohub-cell-tracking-exp006-rule-safe-div` | Rule-based blob detector with conservative safe division recovery | complete | 0.839 | Ref 54251148; new best public score |
| exp006-edge065 | `beicicc/biohub-cell-tracking-exp006-unet-edge065` | UNet + ILP, threshold 0.995, division weight 0.8, edge weight -0.65, geometry cleanup | complete | 0.817 | Ref 54251540; below edge -0.5/-0.8 anchors |
| exp007 | `beicicc/biohub-cell-tracking-exp007-rule-safe-plus` | Rule-based safe division variant with gap closing | complete | runtime exceeded | Ref 54255298; no public score returned |
| exp008 | `beicicc/biohub-cell-tracking-exp008-rule-safe-tiny` | Rule-based tighter safe division variant with gap closing | complete | runtime exceeded | Ref 54255299; no public score returned |

Runtime rule update: CPU rule-based variants that take about 900 seconds on the
public test set are not safe for submission. Further variants should either use
the faster exp006-rule structure or a GPU-backed learned pipeline.
