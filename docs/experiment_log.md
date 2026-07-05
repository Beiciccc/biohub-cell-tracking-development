# Experiment Log

Competition: Biohub - Cell Tracking During Development

Daily submission budget checked on 2026-07-01: 5 submissions. The official
submission list was empty before the first batch. The 2026-07-02 batch started
with five remaining daily submissions in the official submission table. The
2026-07-03, 2026-07-04, and 2026-07-05 batches started with zero same-day rows
in the official submission table.

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
| exp022 | `beicicc/biohub-cell-tracking-exp022-hosen-cv6-gap50` | Public CV6 with one-frame gap-closing distance tightened further to 5.0 um | complete | 0.844 | Ref 54299703; ties current best; visible run 222,098 rows in 104.1 seconds |
| exp023 | `beicicc/biohub-cell-tracking-exp023-yunus-v10` | Rule-based v10 with multi-scale DoG detection, line smoothing, two-frame gap recovery, and safe division edges | complete | 0.858 | Ref 54317389; new best public score; visible run 244,948 rows in 170 seconds |
| exp024 | `beicicc/biohub-cell-tracking-exp024-yunus-v10-thr032` | Rule-based v10 with relative DoG threshold raised from 0.030 to 0.032 | complete | 0.859 | Ref 54320504; new best public score; visible run 243,716 rows in 130 seconds |
| exp025 | `beicicc/biohub-cell-tracking-exp025-yunus-v10-thr032-gap55` | Rule-based v10 with relative DoG threshold 0.032 and one-frame gap-closing distance tightened to 5.5 um | complete | 0.860 | Ref 54324365; new best public score; visible run 242,455 rows in 157 seconds |
| exp026 | `beicicc/biohub-exp026-v10-thr032-gap525` | Rule-based v10 with relative DoG threshold 0.032 and one-frame gap-closing distance tightened to 5.25 um | complete | 0.860 | Ref 54327166; ties current best public score; visible run 241,955 rows in 159 seconds |
| exp027 | `beicicc/biohub-exp027-v10-thr033-gap55` | Rule-based v10 with relative DoG threshold raised to 0.033 and one-frame gap-closing distance kept at 5.5 um | complete | 0.860 | Ref 54329789; ties current best public score; visible run 241,712 rows in 150 seconds |
| exp028 | `beicicc/biohub-exp028-yusuke-lb873-repro` | GPU learned-edge tracker with ILP linking and conservative edge-consensus pruning | complete | 0.884 | Ref 54356057; new best public score; visible run 283,675 rows with 145,367 nodes and 138,308 edges; prediction stage 6.71 minutes |

Runtime rule update: CPU rule-based variants that take about 900 seconds on the
public test set are not safe for submission. Further variants should either use
the faster exp006-rule structure or a GPU-backed learned pipeline.
