# Step 3 leave-one-domain-out: combined findings across all 20 runs

**Status.** Interim findings memo, written 2026-09-04 ahead of the supervision meeting.
It reports the 20 completed Stage 3 runs together, as requested, so the folds can be
compared side by side rather than one file per run. The full generated report lives at
`run_reports/s3_lodo_combined_report.md` (produced by `aggregate_stage3.py`); this memo
is the reading of it.

**Evidence boundary.** Every figure below comes from the Stage 3 aggregator output over
the 20 runs, the per-seed test summaries, and the per-image counts for
`rim_one_dl` seed 42. In-domain comparison values come from the three Step 2 reports in
this directory. Nothing is carried over from any other source, and no figure is
estimated where the artefact was not read.

---

## 1. What was run

| Field | Value |
| --- | --- |
| Protocol | Leave-one-domain-out: train on every source domain, select on pooled source validation, evaluate on the held-out domain with **no adaptation** |
| Arm | `stage3_lodo_plain_unet` — plain U-Net, base width 16, no conditioning |
| Grid | 4 held-out domains x 5 seeds (42-46) = **20 runs**, all complete |
| Budget | 300 epochs each, monitoring-only early stopping, checkpoint by lowest validation BCE + soft Dice |
| Test partition | The held-out domain's Step 2 test split, so the cross-domain number sits on the same images as the Step 2 in-domain number |
| Arms available | **One.** Global FiLM and Spatial FiLM do not exist yet |

Because only one arm exists, the paired between-arm test **cannot run yet**. The
aggregator says so explicitly rather than silently omitting it: *"the paired comparison
needs two conditioning arms scored on the same locked folds... The substrate is built and
ready; pass `--paired-arms A B` once the second arm exists."* This memo is therefore a
baseline characterisation, not a method comparison.

---

## 2. Cross-domain results, all four folds

Each figure is the mean over five per-seed run means, with a 95% Student-t interval on
4 degrees of freedom taken **over the seeds**.

| Held-out domain | n test | Disc Dice (95% CI) | Cup Dice (95% CI) | HD95 unit |
| --- | ---: | --- | --- | --- |
| refuge_zeiss | 80 | **0.8834** [0.8614, 0.9055] | **0.7139** [0.6661, 0.7616] | grid px |
| refuge_canon_val | 80 | **0.8684** [0.7854, 0.9515] | **0.7184** [0.6206, 0.8162] | grid px |
| drishti_gs | 51 | **0.7589** [0.6207, 0.8972] | **0.6350** [0.6052, 0.6649] | grid px |
| rim_one_dl | 97 | **0.0537** [0.0359, 0.0715] | **0.0247** [-0.0090, 0.0584] | native px |

Secondary metrics:

| Held-out domain | Disc IoU | Cup IoU | Disc HD95 | Cup HD95 |
| --- | ---: | ---: | ---: | ---: |
| refuge_zeiss | 0.8109 | 0.5761 | 25.19 | 17.46 |
| refuge_canon_val | 0.7938 | 0.5894 | 35.50 | 19.85 |
| drishti_gs | 0.6882 | 0.4992 | 80.19 | 48.93 |
| rim_one_dl | 0.0279 | 0.0135 | 184.07 | 241.30 |

HD95 for `rim_one_dl` is in native pixels and is not comparable to the other three rows,
which are in letterboxed-grid pixels. For that fold the aggregator also reports a
restricted `HD95* = 239.23` for cup, because cup HD95 was finite in only 94-97 of 97
images per seed and in just **93 images common to all five seeds** — the unrestricted
mean is taken over five different image sets. Undefined HD95 is a symptom, discussed in
section 4.

---

## 3. What changed relative to the Step 2 in-domain baselines

The cross-domain test images are the same images as the Step 2 test split, so this
comparison is paired on images. It is **not** paired on runs: Step 2 was a single seed,
Step 3 is a five-seed mean.

| Held-out domain | Disc, in-domain | Disc, cross-domain | Change | Cup, in-domain | Cup, cross-domain | Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| refuge_zeiss | 0.9553 | 0.8834 | **-0.0719** | 0.8700 | 0.7139 | **-0.1561** |
| drishti_gs | 0.9519 | 0.7589 | **-0.1930** | 0.8209 | 0.6350 | **-0.1859** |
| rim_one_dl | 0.9412 | 0.0537 | **-0.8875** | 0.7694 | 0.0247 | **-0.7447** |
| refuge_canon_val | not run in Step 2 | 0.8684 | — | not run in Step 2 | 0.7184 | — |

For the two full-fundus folds the degradation is orderly and is the quantity Stage 3
exists to measure: **7 to 19 Dice points on disc, 16 to 19 on cup**, with cup degrading
at least as much as disc despite starting lower. RIM-ONE-DL is a different phenomenon
entirely.

---

## 4. Finding 1 — the RIM-ONE-DL fold measures field of view, not domain shift

**A Dice of 0.05 is not degradation.** The same architecture, trained in-domain on
RIM-ONE-DL in Step 2, reached **0.9412** disc Dice on these same test images. The data,
the masks and the model are all capable. What fails is the transfer.

The per-image counts for `rim_one_dl` seed 42, pooled over all 97 test images, say
precisely what goes wrong:

| Structure | TP | FP | FN | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Disc | 170,467 | 244,972 | 10,300,758 | **41.0%** | **1.63%** |
| Cup | 13,636 | 146,871 | 2,803,880 | 8.5% | 0.48% |

Precision of 41% with recall of 1.6% is not a model predicting in the wrong place. It is
a model predicting in **roughly the right place at roughly one twenty-fifth of the right
size**. Converting to areas per image:

| Quantity, disc | Pixels per image | Share of canvas |
| --- | ---: | ---: |
| Ground truth | 107,951 | ~41% |
| Prediction | 4,283 | ~1.6% |

The predicted share, **~1.6%**, is almost exactly the disc foreground fraction the model
was trained on: the Step 2 DRISHTI-GS run logs a first-batch disc foreground fraction of
**0.02218**. The network has learned the full-fundus prior — "the disc is a small object
near the centre" — and applies it faithfully to images where the disc fills nearly half
the frame.

The cause is visible in the data audit's own image sizes:

| Domain | Native image size |
| --- | --- |
| REFUGE Zeiss | 2124 x 2056 |
| REFUGE Canon | 1634 x 1634 |
| DRISHTI-GS | ~2045 x 1752 |
| **RIM-ONE-DL** | **524 x 524** |

RIM-ONE-DL is distributed as optic-disc-centred crops; the other three are whole fundus
photographs. Letterboxing to a 512 grid preserves that difference rather than removing
it. The five seeds agree closely (sd 0.0143 on disc Dice) because they are all making the
same systematic error, not because the estimate is reliable.

**Consequence.** This fold is not currently measuring cross-domain generalisation, and no
conditioning mechanism — Global FiLM or Spatial FiLM — can recover a 25x scale mismatch.
Any average across the four folds would be dominated by it, and any later between-arm
comparison that includes it would be measuring the field-of-view mismatch rather than the
effect of conditioning.

*Caveat on the arithmetic:* the canvas share assumes TP/FP/FN are counted on the 512 x 512
letterboxed grid, consistent with the stated metric frame. If they were counted natively
at 524 x 524 the ground-truth share is ~39% rather than ~41%; the conclusion is unchanged
either way.

---

## 5. Finding 2 — seed variance is large on two of the three working folds

The per-seed disc and cup Dice for all 20 runs:

| Held-out domain | 42 | 43 | 44 | 45 | 46 | sd (disc) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| refuge_zeiss, disc | 0.8644 | 0.8880 | 0.9064 | 0.8917 | 0.8667 | 0.0178 |
| refuge_zeiss, cup | 0.7011 | 0.7004 | 0.7792 | 0.6779 | 0.7107 | — |
| refuge_canon_val, disc | 0.9004 | 0.8974 | **0.7491** | 0.9049 | 0.8904 | 0.0669 |
| refuge_canon_val, cup | 0.7028 | 0.7460 | **0.5893** | 0.7896 | 0.7641 | — |
| drishti_gs, disc | 0.7812 | 0.7916 | **0.9161** | **0.6327** | 0.6730 | **0.1110** |
| drishti_gs, cup | 0.6266 | 0.6514 | 0.6661 | 0.6269 | 0.6044 | — |
| rim_one_dl, disc | 0.0321 | 0.0638 | 0.0571 | 0.0475 | 0.0680 | 0.0143 |
| rim_one_dl, cup | 0.0140 | 0.0728 | 0.0163 | 0.0141 | 0.0062 | — |

Two patterns are worth naming.

**DRISHTI-GS disc spans 0.633 to 0.916 across seeds** — a range of 0.283 on the structure
that is normally the easy one. Its cup, by contrast, is stable (0.604 to 0.666). Disc
being less stable than cup inverts the pattern seen in every Step 2 report, where disc is
consistently the easier and tighter structure. The 95% interval, [0.621, 0.897], is
consequently too wide to support any claim finer than "somewhere in the 0.6 to 0.9 range".

**REFUGE-Canon seed 44 is an outlier on both structures** (disc 0.749 against ~0.895
elsewhere; cup 0.589 against ~0.75). A single seed dragging both structures down suggests
a partial optimisation failure in that run rather than image-level difficulty.

Neither pattern is explained yet. Until it is, the honest description of the Stage 3
baseline is that **refuge_zeiss is stable and the other folds are not**, and a
between-arm difference smaller than these intervals would not be detectable at five seeds.

---

## 6. What the two statistics mean

These answer different questions and are deliberately kept apart in the tooling.

**The 95% confidence interval is taken over seeds.** For one (arm, domain, structure,
metric) cell there are five numbers, one per seed, each already a mean over that fold's
test images. The interval is `mean +/- t(4, 0.975) * s / sqrt(5)`. It quantifies how much
the answer moves when the same locked data is retrained with a different weight
initialisation, augmentation draw and shuffle order — a claim about **reproducibility of
the training procedure**, not about the population of fundus images.

**The image-to-image spread is a different number and is never used to build an
interval.** It lives in `dice_std` inside each run's `test_metrics.json` (the +/- figures
in the Step 2 reports). Presenting it as uncertainty would overstate what has been
measured.

**The paired between-arm test** averages the five seeds per image per arm, giving one
value per image, then tests the differences across images with a Wilcoxon signed-rank
test, Holm-adjusted across tests. Pairing removes the shared image-difficulty variance
both arms see; it is valid only because the locked manifest guarantees both arms are
scored on identical image sets, which the aggregator asserts rather than assumes. It is
built and tested, and runs the moment a second arm exists.

---

## 7. Recommended decisions

1. **Re-scope the RIM-ONE-DL fold before it enters any headline number.** Either adopt a
   disc-centred crop consistently across all four domains so the folds share a field of
   view, or keep RIM-ONE-DL out of the cross-domain average and report it separately as a
   field-of-view stress case. Reporting 0.05 alongside 0.88 as if they measure the same
   thing would misdescribe both.
2. **Do not average Dice across the four folds** while RIM-ONE-DL is included.
3. **Diagnose the DRISHTI-GS disc variance** before treating the baseline as settled. If
   it is optimisation instability, more seeds will not fix it and the comparison against
   the FiLM arms will inherit the noise.
4. **Commit the working tree on CREATE.** All 20 runs record a `-dirty` git revision, so
   no Stage 3 result is fully identified by its commit. This is a provenance cleanup, not
   a results problem, but it should not persist into the thesis.

---

## 8. Limitations

- Only one conditioning arm exists, so no method comparison is possible yet and no
  paired test has been run.
- Five seeds per cell give 4 degrees of freedom; the resulting intervals are wide, and
  wide enough on two folds to hide moderate between-arm effects.
- The RIM-ONE-DL fold measures a field-of-view mismatch, so its numbers do not describe
  conditioning or adaptation.
- `refuge_canon_val` has no Step 2 in-domain counterpart, so its degradation is unmeasured.
- The Step 2 comparison mixes a single-seed point estimate against a five-seed mean.
- HD95 is in letterboxed-grid pixels for three folds and native pixels for RIM-ONE-DL;
  the two are not interchangeable, and RIM-ONE-DL cup HD95 is additionally computed over
  seed-varying image sets unless the restricted `HD95*` is used.
- All 20 runs were produced from a dirty working tree.

---

## 9. Reproducing this

Regenerate the full combined report, the JSON and the CSV:

```bash
module load anaconda3/2022.10-gcc-13.2.0 && eval "$(conda shell.bash hook)" && conda activate spfilm
cd ~/edward/spfilm && python aggregate_stage3.py \
  --runs artifacts/runs --skip-smoke \
  --report-out run_reports/s3_lodo_combined_report.md \
  --json-out artifacts/stage3_lodo_summary.json \
  --csv-out artifacts/stage3_lodo_summary.csv
```

Once a second arm exists, add the paired test:

```bash
python aggregate_stage3.py --runs artifacts/runs --skip-smoke \
  --paired-arms stage3_lodo_plain_unet stage3_lodo_spfilm \
  --paired-metric dice --paired-method wilcoxon
```

The aggregator refuses rather than guesses: it fails if a cell is short a seed, if one
arm spans two git revisions, if two runs claim the same domain/seed cell, or if a smoke
rehearsal would silently enter the report. Those refusals are the reason the grid above
can be read as complete.
