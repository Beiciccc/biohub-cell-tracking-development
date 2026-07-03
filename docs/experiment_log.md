# Experiment Log

Competition: Biohub - Cell Tracking During Development

Daily submission budget checked on 2026-07-01: 5 submissions. The official
submission list was empty before the first batch. The 2026-07-02 batch started
with five remaining daily submissions in the official submission table. The
2026-07-03 batch started with zero same-day rows in the official submission
table.

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
| exp010 | `beicicc/biohub-cell-tracking-exp010-rule-fast-safe-plus` | Fast rule-based safe division widening without slow gap-closing variant | complete | 0.839 | Ref 54274155; ties current best |
| exp016 | `beicicc/biohub-cell-tracking-exp016-hosen-cv6` | Public CV6 rule baseline with two-pass velocity linking, one-frame gap closing, and short-track filtering | complete | 0.842 | Ref 54292435; new best public score; visible run 225,701 rows in 84.9 seconds |
| exp017 | `beicicc/biohub-cell-tracking-exp017-hosen-cv6-link85` | Public CV6 with loose link gate widened from 8.0 to 8.5 um | complete | 0.842 | Ref 54294439; ties current best; visible run 225,574 rows in 94.9 seconds |
| exp018 | `beicicc/biohub-cell-tracking-exp018-hosen-cv6-gap55` | Public CV6 with one-frame gap-closing distance tightened from 6.0 to 5.5 um | complete | 0.843 | Ref 54296135; new best public score; visible run 223,906 rows in 89.5 seconds |
| exp021 | `beicicc/biohub-cell-tracking-exp021-hosen-cv6-gap525` | Public CV6 with one-frame gap-closing distance tightened further to 5.25 um | complete | 0.844 | Ref 54298117; new best public score; visible run 223,084 rows in 94.8 seconds |

Runtime rule update: CPU rule-based variants that take about 900 seconds on the
public test set are not safe for submission. Further variants should either use
the faster exp006-rule structure or a GPU-backed learned pipeline.
