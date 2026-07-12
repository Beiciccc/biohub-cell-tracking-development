# Experiment Log

Competition: Biohub - Cell Tracking During Development

Daily submission budget checked on 2026-07-01: 5 submissions. The official
submission list was empty before the first batch. The 2026-07-02 batch started
with five remaining daily submissions in the official submission table. The
2026-07-03, 2026-07-04, and 2026-07-05 batches started with zero same-day rows
in the official submission table. The 2026-07-06 batch started with one
same-day row already present in the official submission table. The 2026-07-07
batch started with zero same-day rows in the official submission table.
The 2026-07-08 batch started with one same-day row already present in the
official submission table.
The 2026-07-09 batch started with zero same-day rows in the official
submission table.
The 2026-07-10 batch started with zero same-day rows in the official
submission table.
The 2026-07-11 and 2026-07-12 batches each started with zero same-day rows in
the official submission table.

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
| exp029 | `beicicc/biohub-exp029-lb884-prune50` | GPU learned-edge tracker with the more conservative ultra-tiny prune50 preset | complete | 0.885 | Ref 54363595; new best public score; visible run 270,404 rows with 138,478 nodes and 131,926 edges; prediction stage 6.75 minutes |
| exp030 | `beicicc/biohub-exp030-lb884-prune40` | GPU learned-edge tracker with the latest conservative prune40 preset | complete | 0.885 | Ref 54370950; ties current best public score; visible run 270,404 rows with 138,478 nodes and 131,926 edges; prediction stage 6.17 minutes |
| exp031 | `beicicc/biohub-exp031-pilkwang-precision-repair` | Pilkwang learned graph precision-repair branch with detection threshold 0.992 | complete | 0.884 | Ref 54375624; below current best; visible run 266,902 rows with 136,859 nodes and 130,043 edges; prediction stage 6.61 minutes |
| exp032 | `beicicc/biohub-exp032-pilkwang-recall-clean` | Pilkwang learned graph recall-clean branch with detection threshold 0.985 and two-step gap recovery | complete | 0.888 | Ref 54380857; new best public score; visible run 276,082 rows with 141,328 nodes and 134,754 edges; prediction stage 6.13 minutes |
| exp033 | `beicicc/biohub-exp033-yusuke-score-push` | Yusuke learned graph score-push branch with detection threshold 0.99, gap recovery, and tightened division caps | complete | 0.893 | Ref 54387185; new best public score; visible run 262,385 rows with 134,239 nodes and 128,146 edges; prediction stage 6.43 minutes |
| exp034 | `beicicc/biohub-exp034-yusuke-safe-div-precision` | Yusuke score-push branch with tighter safe-division geometry caps | complete | 0.893 | Ref 54393983; ties current best public score; visible run 262,359 rows with 134,238 nodes and 128,121 edges; prediction stage 6.30 minutes |
| exp035 | `beicicc/biohub-exp035-vmerckle-pass-rescue` | Vmerckle learned graph branch with learned motion relinking prepass and rescue edges | complete | 0.893 | Ref 54401907; ties current best public score; visible run 262,385 rows with 134,220 nodes and 128,165 edges; prediction stage 6.32 minutes |
| exp036 | `beicicc/biohub-exp036-vmerckle-altunet-divseed` | Vmerckle pass-rescue branch with motion relinking division-seed candidates | complete | 0.893 | Ref 54407572; ties current best public score; visible run 262,381 rows with 134,217 nodes and 128,164 edges; prediction stage 6.51 minutes; 30 motion division-seed edges |
| exp037 | `beicicc/biohub-exp037-yusuke-lb897-min7-short-track` | Yusuke learned graph branch with short-track filtering increased to min length 7 | complete | 0.897 | Ref 54432797; new best public score; visible run 248,558 rows with 126,379 nodes and 122,179 edges; prediction stage 6.25 minutes; removed 7,465 short-track nodes and 5,307 short-track edges |
| exp038 | `beicicc/biohub-exp038-pilkwang-blend-default` | Pilkwang learned graph branch with DeepCenter full-frame center prior and min6 short-track filtering | complete | 0.897 | Ref 54439274; ties current best public score; visible run 251,664 rows with 128,072 nodes and 123,592 edges; prediction stage 6.32 minutes; added 12 full-frame center nodes, 2,119 gap nodes, and 423 safe-division edges |
| exp039 | `beicicc/biohub-exp039-yusuke-min8-short-track` | Yusuke learned graph branch with short-track filtering increased to min length 8 | complete | 0.896 | Ref 54444702; below current best; visible run 245,386 rows with 124,671 nodes and 120,715 edges; prediction stage 7.06 minutes; removed 9,173 short-track nodes and 6,771 short-track edges |
| exp040 | `beicicc/biohub-exp040-vmerckle-linefit-skip-div-edges` | Vmerckle learned graph branch with linefit smoothing, two-frame gap recovery, and conservative division-edge filtering | complete | 0.893 | Ref 54458011; below current best; visible run 261,385 rows with 134,051 nodes and 127,334 edges; prediction stage 6.80 minutes; added 406 two-frame gap nodes and 445 safe-division edges |
| exp043 | `beicicc/biohub-exp043-yusuke-dataset-mintrack-recall` | Yusuke learned graph branch with dataset-specific short-track recall restoration | complete | 0.892 | Ref 54466209; below current best; visible run 269,435 rows with 137,079 nodes and 132,356 edges; prediction stage 6.86 minutes; removed 7,583 short-track nodes and 5,448 short-track edges |
| exp044 | `beicicc/biohub-exp044-pilkwang-conservative-min8` | Pilkwang latest blend branch with stricter min8 short-track pruning and gap2 disabled | complete | 0.890 | Ref 54482138; below current best; visible run 235,824 rows with 119,872 nodes and 115,952 edges; prediction stage 6.52 minutes; removed 9,906 short-track nodes and 7,309 short-track edges |
| exp046c | `beicicc/biohub-exp046c-rahul-rj4-pathfix` | Rahul DeepCenter sparse-gate branch with corrected Kaggle checkpoint mount | complete | 0.889 | Ref 54488437; below current best; visible run 239,697 rows with 121,923 nodes and 117,774 edges; prediction stage 6.69 minutes; added 4 full-frame center nodes and removed 7,939 short-track nodes |
| exp048 | `beicicc/biohub-exp048-tamerlan-gap-recovery` | Learned graph min6 pruning branch with additional gap recovery | complete | 0.892 | Ref 54496234; below current best; visible run 257,351 rows with 131,022 nodes and 126,329 edges; prediction stage 6.61 minutes; added 2,208 gap nodes and 413 safe-division edges |
| exp049 | `beicicc/biohub-exp049-khj-vel0-min6` | High-confidence min6 branch with velocity-free motion relinking | complete | 0.889 | Ref 54508141; below current best; visible run 242,870 rows with 123,666 nodes and 119,204 edges; prediction stage 6.89 minutes; added 2,113 gap nodes and 376 safe-division edges |
| exp050 | `beicicc/biohub-exp050-boris-min6` | Learned graph min6 branch with tighter division guards | complete | 0.887 | Ref 54503491; below current best; visible run 242,895 rows with 123,689 nodes and 119,206 edges; prediction stage 7.06 minutes; added 2,270 gap nodes and 340 safe-division edges |
| exp052 | `beicicc/biohub-exp052-abhijith-v74-d4` | 400-epoch learned graph branch with full D4 detection TTA, threshold 0.97, and min6 pruning | complete | 0.900 | Ref 54522943; new best public score; visible run 252,523 rows with 128,535 nodes and 123,988 edges; prediction stage 10.47 minutes; added 2,068 gap nodes and 392 safe-division edges |
| exp053 | `beicicc/biohub-exp053-pilkwang-v21-rescue` | Exp052 graph anchor with conditional high-confidence short-track recovery when pruning is unusually severe | complete | 0.900 | Ref 54533859; ties current best; visible run 252,842 rows with 128,715 nodes and 124,127 edges; rescue triggered on one dataset and restored 41 components with 180 nodes |
| exp054 | `beicicc/biohub-exp054-pilkwang-v23-precision-9725` | Exp052 graph anchor with detection threshold raised from 0.9700 to 0.9725 while graph constraints remain fixed | complete | 0.899 | Ref 54553760; 0.001 below current best; visible run 251,670 rows with 128,106 nodes and 123,564 edges; prediction stage 10.85 minutes; removed 429 nodes and 424 edges relative to exp052 |
| exp056 | `beicicc/biohub-exp056-division-prior09` | Exp052 graph anchor with the ILP division prior lowered from 1.0 to 0.9 while detection and post-processing remain fixed | complete | 0.900 | Ref 54563539; ties current best; visible run 252,526 rows with 128,537 nodes and 123,989 edges; prediction stage 10.39 minutes; added 2 nodes and 1 edge relative to exp052 |
| exp057 | `beicicc/biohub-exp057-center-confirmed-marginal-gap` | Exp052 graph anchor with an immutable epoch-400 DeepCenter checkpoint confirming only one-frame gap repairs spanning at least 8 um | complete | 0.901 | Ref 54595520; new best public score; visible run 247,815 rows with 126,279 nodes and 121,536 edges; 733 marginal repairs checked and rejected, 1,335 strong-motion repairs retained; submission SHA-256 `37c01034d31d63087e2e4433e4c7a7c4e1eb2fd91b115aae78da3a100de5f901` |

Runtime rule update: CPU rule-based variants that take about 900 seconds on the
public test set are not safe for submission. Further variants should either use
the faster exp006-rule structure or a GPU-backed learned pipeline.
