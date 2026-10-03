# Phase B clean FEI+C re-prep -- POST-HOC analysis (not preregistered), fixed before any clean LOSO run
Why: audit 2026-09-26 found the preregistered FEI+C input band (0.5-40 Hz FIR, labelled "NMT device band") does not
reproduce NMT's spectrum (NMT is unfiltered: roll-off ~35-40 Hz onto a floor, little <0.5 Hz power), putting stroke
inputs out of distribution for Branch-C's input BN and collapsing pretrained C on stroke.
Prep: authors' bad-channel interpolation -> CAR -> 19 ch (T7/T8/P7/P8 -> T3/T4/T5/T6) -> resample 200 Hz, no
band-pass -> per-channel z -> ONE fixed zero-phase per-channel spectral-shaping filter per cohort, estimated from the
cohort's mean log10 Welch PSD (0.25 Hz bins, all subjects pooled, labels never used) to NMT-train's mean log10 PSD
(100 fixed-random NMT-train recordings, first 300 s), 4 fixed-point iterations -> per-channel z -> 2.5 s frames.
Transductive but label-free: the same linear filter is applied to every subject.
Gate (reported; LOSO runs regardless, a failed gate is a caveat): matched group-mean log10 PSD within 0.25 decades
of NMT at every channel x bin in 0.5-99 Hz; Branch-C input-BN |z| (channel-averaged, per 1 Hz bin, vs each NMT
checkpoint's running stats) <= 1.5 everywhere; pretrained C across-subject std >= 0.5 x its std on 15 NMT-eval
recordings; <= 5 % constant dims (std < 1e-5).
Encoders/classifier/evaluation identical to the preregistration (5 NMT-only PLAIN FEI + Branch-C h20 members,
3 random-init, StandardScaler + LogReg C=1 balanced, LOSO, 1000-perm p for FEI+C and rand-FEI+C;
2000 bootstrap CIs) plus the CV-robustness schemes (100x stratified 5-fold, leave-pair-out).
Comparisons: FEI+C vs rand-FEI+C; FEI+C vs BSI; FEI+C+BSI vs BSI; FEI+C vs FEI+C (preregistered prep); FEI+C vs
VJEPA; C vs rand-C; FEI vs rand-FEI. Primary cohort only (15 subjects; trimmed for time, decided before any run).
Nothing else in Phase B changes; preregistered outputs are never overwritten.
