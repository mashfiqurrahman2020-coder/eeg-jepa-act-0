# EEG-VJEPA Act 0: Reproduction and FEI+GTJ

Code, checkpoints and results for the Act-0 report of EEE 402 (BUET, Section G2, Group 01):
[docs/Act0_Final_Report.pdf](docs/Act0_Final_Report.pdf).

EEG-VJEPA is a published self-supervised model that adapts a video-learning method (V-JEPA) to EEG and
reports strong results at telling normal recordings from abnormal ones. We tried to reproduce that result
with the authors' own released weights and evaluation code. It did not reproduce. The released checkpoints
are collapsed, meaning they give almost the same output for every recording, and on the TUAB test set a
random, untrained copy of the same network scores higher than the released one.

We then built a smaller alternative from two encoders:
- **FEI** learns from a recording's frequency content.
- **GTJ** learns how its spectrum changes over time.

We tested the two encoders, alone and fused (FEI+GTJ), on three datasets, each further from the training
data than the last. Every claim that pre-training helped is checked against a random-weight twin of the same
network, run through the same code.

```
  ┌────────────────────────────┐      ┌────────────────────────────┐      ┌────────────────────────────┐
  │ NMT  (Pakistan)            │      │ TUAB  (USA)                │      │ Stroke cohort              │
  │ 2,417 recordings           │      │ 2,993 recordings           │      │ 15 subjects                │
  │                            │      │                            │      │                            │
  │ pre-train FEI and GTJ      │ ───► │ frozen encoders            │ ───► │ frozen encoders            │
  │ (no labels), then 5-fold   │      │ + linear probe, against    │      │ + leave-one-subject-out    │
  │ cross-validated probe      │      │ released EEG-VJEPA ViT-M   │      │ probe, against the Brain   │
  │                            │      │                            │      │ Symmetry Index (BSI)       │
  │ Experiment 5               │      │ Experiments 1–4, 6         │      │ Experiments 7–8            │
  └────────────────────────────┘      └────────────────────────────┘      └────────────────────────────┘
```

## Repository layout

| Path | Contents |
|------|----------|
| `docs/Act0_Final_Report.pdf` | The current report PDF |
| `docs/act0_repro/` | Fold, subset and cohort lists; SHA-256 of every checkpoint, result file and raw recording; pinned packages and the machine's environment |
| `code/*.py` | One script per experiment (report Table 5) and per input (Table 4), plus the modules they import |
| `code/*.pt` | Our 70 FEI/GTJ encoder checkpoints (see [Checkpoints](#checkpoints)) |
| `code/runs/act0/` | Saved results (JSON), per-recording predictions (`preds_*.npz`, `oof_*.npz`, `kf5_lpo_*.npz`), run logs, and the stroke study's `PREREGISTRATION.md` and `POSTHOC.md` |
| `code/app`, `code/src`, `code/evals`, `code/configs`, `code/setup.py` | Our copy of the EEG-VJEPA code at upstream commit `c739fad5ad57ec4439a094343901a245edf611ec` |

## How it works

### The two encoders

**FEI** (Frequency-masked Embedding Inference) is a 1-D convolutional encoder that reads 5-second windows
of 19-channel EEG at 200 Hz. In pre-training, between 0% and 70% of each window's frequency bins are
removed, and the encoder learns to predict, in embedding space, what a slowly updated copy of itself makes
of that masked window. A small extra term predicts which bands were removed. Both targets are L2-normalised,
which stops the network from cheating by shrinking its output towards zero.

**GTJ** (GRU-based Time-JEPA) turns 20-second windows into spectrograms and runs a GRU over them. In
pre-training it predicts the future course of the spectrum from its past, again against a slowly updated
copy of itself.

Both use the same training recipe (`code/fei_pretrain.py`, `code/fei_branchC.py`):

| Setting | Value |
|---------|-------|
| Optimiser | AdamW, learning rate 2e-4, weight decay 1e-4 |
| Batch | 128 windows |
| Epochs | 60, keeping the final epoch's weights |
| Windows per recording per epoch | 4, drawn at random |
| Target encoder | exponential moving average, rate 0.996 |
| Embedding size | 256 |

To score a recording, each encoder embeds every non-overlapping window and averages the embeddings, and a
standardised logistic regression makes the call. The NMT encoders come in fives, one per cross-validation
fold; on TUAB and the stroke cohort the five are ensembled.

### How results are checked

| Rule | Why |
|------|-----|
| Every pre-trained encoder has random-weight twins with the same architecture and code | A score alone cannot separate what pre-training learned from what the architecture supplies |
| Every encoder is checked for collapse (spread of its outputs across recordings, pairwise cosine) | A 2-D t-SNE/UMAP picture cannot show collapse |
| Comparisons on one test set use a paired bootstrap over recordings (2,000 resamples) | Both models see the same resampled recordings |
| Cross-validation folds use the Nadeau–Bengio corrected t-test, with Holm's adjustment across a family | Folds share training data, so a plain t-test is too optimistic |
| Gains over several pre-training runs resample the runs, the twin's random seeds and the recordings together | Seed-to-seed variation enters the interval |
| The 15-subject stroke cohort uses label permutations and repeated splits | Too few subjects for asymptotic tests |

## Results

| Experiment | Result |
|------------|--------|
| 1. Released ViT-M/ViT-B on TUAB, the paper's own heads | 0.1–0.2 AUROC short of the paper's 0.877 / 0.879; a random-weight ViT-M scores higher |
| 2. 500 epochs of fine-tuning, the authors' recipe | No better than training the same network from scratch (p = 0.81) |
| 3. Collapse check | The released encoder's output barely varies from one recording to the next |
| 4. Linear probe on the released ViT-M | 0.838 on the full training set, still not separable from a random-weight twin |
| 5. FEI+GTJ on NMT, 5-fold | 0.865 ± 0.024 (mean of three pre-training runs 0.861), against band power 0.786; an untrained twin already reaches 0.834, so pre-training adds +0.027, not significant after correction |
| 6. TUAB, encoders pre-trained on NMT only | FEI beats its random-weight twin by +0.048 AUROC averaged over five pre-training runs (95% CI [0.015, 0.108]); GTJ does not transfer (0.840 against its twin's 0.863) |
| 6. TUAB, re-pre-trained with TUAB's unlabelled recordings | FEI+GTJ 0.862 against the released ViT-M's 0.838 (not significant), and not significantly better than its own twin in any of three pre-training runs; FEI on its own stays above its twin in all three (+0.038 averaged, 95% CI [−0.0003, 0.104]) |
| 7. BSI on the stroke cohort | 0.907 AUROC |
| 8. FEI+GTJ on the stroke cohort | 0.926 leave-one-subject-out AUROC (two seed re-runs: 0.963, 0.981), after a spectral-matching step designed post hoc on this cohort; 15 subjects cannot separate it from BSI |

The report gives the intervals, p-values and caveats behind every row.

## Running the project

### What you need

- Python 3.9 with the packages in `docs/act0_repro/requirements-act0.txt`.
  `docs/act0_repro/environment.txt` records the machine the results came from (one NVIDIA RTX 5060 Ti).
- A CUDA GPU for pre-training and for the released-checkpoint evaluations. The probes and statistics run on
  the CPU.
- The recordings, which this repository does not include:
  - NMT and the stroke cohort are public.
  - TUAB must be requested from its maintainers under their data use agreement (report §5.5.3).
- The released EEG-VJEPA weights, from Hugging Face `amir-hlp/EEG-VJEPA` at commit
  `ec98405f5f3423fd59d3296157ee1570efd8d947`, saved in `pretrained/`.

The saved embeddings (`emb_*.npz` and `emb_*.npy`, several hundred MB) are not included either. Each one is
a forward pass of a checkpoint here or of a seeded random-weight twin, so the scripts regenerate them.

### 1. Put things where the scripts expect them

The scripts expect this checkout at `/home/mashfiq/eeg_vjepa`; that path is the `ROOT` constant at the top
of each script. Either clone to that path or change `ROOT`. The data goes under `data/` and the released
weights under `pretrained/`.

### 2. Check the files

Check the checkpoints and results against the manifests:

```bash
tr -d '\r' < docs/act0_repro/checkpoints_sha256.csv | awk -F, 'NR>1 && $1!~/^pretrained/{print $3"  "$1}' | sha256sum -c --quiet
tr -d '\r' < docs/act0_repro/results_sha256.csv | awk -F, 'NR>1{print $3"  "$1}' | sha256sum -c --quiet
```

Check your copy of the recordings, from the folder that holds `data/`:

```bash
tr -d '\r' < docs/act0_repro/raw_recordings_sha256.csv | awk -F, 'NR>1{d=($1=="TUAB")?"data/TUAB/edf":($1=="NMT")?"data/NMT-Scalp-EEG":"data/zenodo_stroke"; print $4"  "d"/"$2}' | sha256sum -c --quiet
```

No output means every file matched.

### 3. Run something

A quick check that needs no data:

```bash
cd code
python fei_pretrain.py --selftest
python fei_branchC.py --selftest
```

For everything else, report Table 5 names the script behind each experiment, Table 4 the script that
prepares each input, and Appendix §11.6 the pre-training commands and seeds. Long runs save their progress as
they go, so an interrupted run resumes where it stopped when you run the same command again.

The stroke analysis can run without `PREREGISTRATION.md`. When that file is present, its SHA-256 is recorded
in new results; when absent, the field is `null`. The existing saved results keep the fingerprint from the
original study. Use `python stroke_phaseB.py --prereg` only when creating a new preregistration file.

### What to expect

- **From the saved checkpoints and predictions:** every number in the report, exactly.
- **From re-training:** the same results statistically, not bit for bit. The original FEI and GTJ runs set no
  random seed, and cuDNN kernels were left non-deterministic, so even the seeded re-runs vary slightly.

### If something does not work

| Symptom | Likely cause |
|---------|--------------|
| A script cannot find data, checkpoints or results | `ROOT` still points at `/home/mashfiq/eeg_vjepa`; see step 1 |
| `sha256sum` reports every line as failed | Windows line endings were left in. The CSVs use CRLF; keep the `tr -d '\r'` in the commands above |
| A run stops printing for a while | `code/hw_guard.py` pauses work while the CPU is too hot and resumes once it cools. It is not a hang |
| A run stops with a temperature error | The guard aborts if the CPU cannot cool below its hard limit (92 °C by default). Improve cooling and re-run; it resumes |
| A released-checkpoint script cannot load weights | The weights must be in `pretrained/`, from the Hugging Face commit above; their SHA-256 values are in `docs/act0_repro/checkpoints_sha256.csv` |
| A re-trained encoder gives a slightly different number | Expected; see [What to expect](#what-to-expect) |

## Checkpoints

There are 70 checkpoints: five, one per NMT cross-validation fold, for each of these 14 encoders.

| Encoder | Pre-trained on | Runs | Files |
|---------|----------------|------|-------|
| FEI | NMT | original, re-runs 1–4 | `fei_enc_cv_s0_f{k}.pt`, `fei_rep{r}_enc_cv_s0_f{k}.pt` |
| GTJ | NMT | original, re-runs 1–2 | `branchC_h20_enc_cv_s0_f{k}.pt`, `branchC_h20_rep{r}_enc_cv_s0_f{k}.pt` |
| FEI | NMT + TUAB unlabelled | original, re-runs 1–2 | `fei_joint_enc_cv_s0_f{k}.pt`, `fei_joint_rep{r}_enc_cv_s0_f{k}.pt` |
| GTJ | NMT + TUAB unlabelled | original, re-runs 1–2 | `branchC_h20_joint_enc_cv_s0_f{k}.pt`, `branchC_h20_joint_rep{r}_enc_cv_s0_f{k}.pt` |

GTJ's files carry its old working name, Branch C. FEI re-runs 3 and 4 were trained for Experiment 6 only and
have no GTJ partner.

## Notes

**Version used for the report.** The report's §7.2 names the code, checkpoints and results by their full
commit SHA. The report itself is under the tag `act0-report-round15f`.

**TUAB data note.** Our TUAB copy came from an unofficial mirror, not through the corpus's
data use agreement, so the TUAB numbers here are for this course submission only (report §5.5.3).

**Licence.** Our own code and documents are under the MIT licence ([LICENSE](LICENSE)). The EEG-VJEPA code in
`code/app`, `code/src`, `code/evals`, `code/configs` and `code/setup.py` is not ours and stays under its own CC BY-NC 4.0
licence ([code/LICENSE](code/LICENSE)), which MIT cannot override, so anything that runs that code is for
non-commercial use only. The upstream README is `code/UPSTREAM_README.md`; five of its files carry labelled additions
of ours, listed in the report's Experiment 1. The licence covers our work, not the datasets: NMT, TUAB and the stroke
cohort keep their own terms, and the checkpoints and results that used our TUAB copy are for this course submission
only (see above).
