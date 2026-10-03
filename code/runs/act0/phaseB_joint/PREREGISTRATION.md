# Act-0 Part 2 (Phase B) preregistration: healthy vs stroke (Zenodo 19599466)
Written before any LOSO run; `stroke_phaseB.py --loso` refuses to run without it and stores its sha256.

## Cohorts (y: stroke = 1; POST-intervention files never used)
- **primary** (pure rest block, before any task marker / segment break), fixed 285 s from onset for every subject
  (shortest rest block = PAC05 285.0 s, so no duration confound): stroke PAC02-PAC10 (9) vs controls C03,
  CONTROL04-08 (6) = the authors' own control list. PAC01, 02, C01 excluded (<= 52 s of rest before the task).
- **sensitivity**: authors' definition = first 300 s of every file (qEEG_5min_previos.m), PAC01-10 (10) vs
  02, C01, C03, CONTROL04-08 (8). For PAC01/02/C01 this includes task data -- reported as sensitivity only.

## Preprocessing
- Authors' bad-channel lists (1-based file-order indices) -> spherical-spline interpolation (EEG ch only). No ICA
  (deviation from preprint; stated).
- BSI gate: recorded A2 reference, notch 50 Hz, 0.5-100 Hz, 2 s non-overlapping epochs, reject epochs > +-100 uV
  on the 21 central channels, Welch (1 s Hann, 50%), bands delta 1-4 theta 4-8 alpha 8-12 beta 12-30.
  Relative power = band / sum of 4 (channel-averaged); DAR = d/a; DTABR = (d+t)/(a+b).
  pdBSI = mean over 6 pairs (FC1/FC2, FC3/FC4, C1/C2, C3/C4, C5/C6, CP1/CP2) and bins of |(R-L)/(R+L)|; broadband
  1-30 Hz + per band.
- Encoder inputs: CAR over EEG channels, 19 ch (T7/T8/P7/P8 -> T3/T4/T5/T6).
  ViT-M: paper §4.1 (1-40 Hz, 100 Hz, per-channel z, 5 s frames stride 2.5 s).
  FEI+C: NMT-matched (0.5-40 Hz = NMT device band, 200 Hz, per-channel z, 2.5 s non-overlapping frames).

## Gate (pre-declared)
Pipeline soundness: broadband pdBSI higher in stroke, two-sided Mann-Whitney p < 0.05 (primary cohort).
Direction agreement with the preprint also reported for DAR, DTABR, rel-delta (up), rel-alpha, rel-beta (down).
LOSO runs regardless; a failed gate is reported as a caveat, never used to change the design.

## Arms (features)
- BSI = [pdBSI 1-30, delta, theta, alpha, beta] (5).
- VJEPA = released ViT-M (4x30x4) frozen; per clip position token mean ++ token std, averaged over all positions.
- rand-VJEPA = same architecture random-init, seeds 0-2 (collapse control).
- FEI+C = PLAIN FEI (fei_enc_cv_s0_f{k}.pt, L=4000) ++ Branch-C h20 (branchC_h20_enc_cv_s0_f{k}.pt), k = 0..4 NMT
  seed-0 fold encoders -> 5 members. rand-FEI+C = random-init seeds 0-2.
- each also + BSI (hstack).
- Collapse metrics per encoder: token std, across-subject feature std, mean pairwise cosine.

## Classifier / evaluation
StandardScaler + LogisticRegression(C=1, class_weight=balanced), leave-one-subject-out. Multi-member arms:
ensemble = mean member P(stroke) (member AUROC mean +- sd also reported). Threshold 0.5 (never tuned).
Metrics: AUROC (primary), AUC-PR, Acc, BAcc, F1, Sens, Spec, confusion; bootstrap 95% CI (2000, over subjects);
label-permutation p (1000, full LOSO re-run per permutation).
Pre-declared paired comparisons (bootstrap delta-AUROC, same resamples): VJEPA+BSI vs BSI; VJEPA vs rand-VJEPA;
FEI+C vs VJEPA. Our-pretrained ViT-M arm is added after P1.5 as a separate, labelled addition.
