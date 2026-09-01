# Step 2 in-domain baseline report: RIM-ONE-DL

**Evidence boundary.** This report uses the pasted summary_table.txt, RUN_NOTES.md and sampled history output, together with the project brief and dataset publication. Configuration fields, epoch rows and checkpoint results absent from those sources are omitted rather than represented by placeholders.

## 1. Objective

This run establishes the plain U-Net in-domain segmentation baseline on RIM-ONE-DL. It belongs to Step 2 of the project sequence: validate the plain backbone on a single source before quantifying cross-domain degradation or introducing Global FiLM and Spatial FiLM.[^brief-rim]

The run tests whether the unconditioned pipeline can learn optic-disc and optic-cup masks from the local RIM-ONE-DL mixture. It does not test adaptation and cannot show whether spatial conditioning is beneficial.

## 2. Dataset and split

RUN_NOTES.md identifies the source as rim_one_dl, a mixture of RIM-ONE releases r1, r2 and r3.[^rim-notes] The supplied standing caveat specifies a flat random **70/10/20** image-level split over **485** images.

| Split | Images |
|---|---:|
| Train | 340 |
| Validation | 48 |
| Test | 97 |
| Total | 485 |

Source: image counts are from summary_table.txt.[^rim-summary]

The split is locally derived, not provider-defined. The supplied reporting caveat identifies it as a flat random image split; the supplied excerpts do not identify who fixed the split implementation or whether it was stratified. Filenames encode eye but not patient, so fellow-eye correlation across train, validation and test is undetectable rather than known to be absent. This may inflate the in-domain result.

## 3. Model and training setup

The table below contains the complete set of model and training fields supported by the supplied run excerpts.

| Parameter | Run setting | Source |
|---|---|---|
| Architecture | Plain 2D U-Net; no conditioning; two-channel sigmoid head for disc and cup; InstanceNorm with affine parameters after every convolution | RUN_NOTES.md |
| Output supervision | Single head and single-tensor loss; deep supervision deferred | RUN_NOTES.md |
| Trainable parameters | **1,944,066** | summary_table.txt |
| Epoch budget | **300** | pasted history heading and final logged epoch |
| Epochs run | **300** | pasted history heading and final logged epoch |
| Input resolution | **512 × 512** | RUN_NOTES.md |
| Epoch-1 learning rate | **1.000e-03** | sampled history output |
| Final logged learning rate | **1.000e-06** | sampled history output |
| Loss | Equal-weight BCEWithLogits plus soft Dice | RUN_NOTES.md |
| Seed | **42** | RUN_NOTES.md |
| Hard-mask threshold | **0.5** | summary_table.txt |
| Device | CUDA | summary_table.txt |

### Training protocol and checkpoint selection

The run consumed the full configured budget: **300 of 300 epochs** were logged.[^rim-history] Early stopping did not terminate training. It acted as a counterfactual rule alongside best-checkpoint selection.

| Early-stopping field | Setting |
|---|---|
| Selection metric | Lowest validation BCE plus soft Dice loss |
| Counterfactual firing epoch | **101** |
| Global best epoch after the full run | **81** |

Sources: selection metric and best epoch are from summary_table.txt; the counterfactual firing epoch is from the pasted history heading.[^rim-summary][^rim-history]

Under the previous terminating protocol this run would have ended at epoch **101**, selecting the epoch-**81** checkpoint. Under the current protocol it continued to epoch 300, but epoch 81 remained the global best. The reported test metrics come from that epoch-81 checkpoint, selected by minimum validation loss (summary_table.txt).

## 4. Per-epoch results

### Milestone epochs

The table retains the numeric milestones available in the pasted history sample: the first epoch, sampled intervals, the counterfactual firing epoch, the best epoch and the final epoch. Unsupplied rows are omitted.

| Epoch | Learning rate | Train loss | Validation loss | Validation Dice: disc | Validation Dice: cup |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.000e-03 | 1.1437 | 1.0246 | 0.9217 | 0.3071 |
| 50 | 9.331e-04 | 0.2302 | 0.2324 | 0.9500 | 0.7794 |
| 81 | 8.308e-04 | 0.1769 | 0.1979 | 0.9553 | 0.8203 |
| 100 | 7.503e-04 | 0.1629 | 0.2107 | 0.9538 | 0.7628 |
| 101 | 7.457e-04 | 0.1813 | 0.2396 | 0.9525 | 0.7312 |
| 150 | 5.005e-04 | 0.1225 | 0.2162 | 0.9532 | 0.7705 |
| 200 | 2.507e-04 | 0.0744 | 0.2360 | 0.9552 | 0.7779 |
| 250 | 6.792e-05 | 0.0491 | 0.2696 | 0.9526 | 0.7779 |
| 300 | 1.000e-06 | 0.0428 | 0.2747 | 0.9534 | 0.7771 |

Source: pasted sampled history output.[^rim-history]

The principal optimisation gain occurred before epoch **81**. From epoch **1 to 81**, validation loss fell from **1.0246 to 0.1979**; validation disc Dice rose from **0.9217 to 0.9553**, and cup Dice from **0.3071 to 0.8203**. The logged LR decreased from **1.000e-03 to 8.308e-04**.[^rim-history]

Validation degraded after the best checkpoint while training loss continued to fall. At the counterfactual firing epoch **101**, validation loss was **0.2396** and cup Dice **0.7312**. At epoch **300**, training loss reached **0.0428**, but validation loss was **0.2747**, disc Dice **0.9534** and cup Dice **0.7771**; the LR had reached **1.000e-06**. The late schedule fitted the training data more closely without improving checkpoint selection.

## 5. Test results

Metrics were computed after resizing each square source crop to the **512 × 512** evaluation grid. Each HD95 value was divided by the recorded native-to-grid scale, so HD95 is reported in native-source pixels, not grid pixels or millimetres (summary_table.txt).[^rim-summary] Disc and cup are reported separately; **no combined disc-plus-cup Dice is reported**.

| Checkpoint | Structure | Dice | IoU | HD95 | Accuracy |
|---|---|---:|---:|---:|---:|
| Best, epoch 81 | Disc | **0.9412 ± 0.0399** | **0.8915 ± 0.0657** | **27.00 ± 25.49 native px** | **0.9525 ± 0.0296** |
| Best, epoch 81 | Cup | **0.7694 ± 0.1896** | **0.6564 ± 0.2058** | **43.35 ± 34.62 native px** | **0.9637 ± 0.0289** |

Source: best-checkpoint means and standard deviations over **97** test images are from summary_table.txt; all HD95 counts were **97** with **0** exclusions.[^rim-summary] Final-epoch test metrics were not included in the supplied excerpts, so the best-versus-final test gap cannot be calculated.

The summary's degenerate-case policy retains smoothed near-zero Dice and IoU for an empty prediction against a non-empty target; both-empty Dice and IoU equal **1.0**; undefined HD95 values are excluded rather than set to zero. No test image was excluded from either structure's HD95 summary (summary_table.txt).

## 6. Findings

### Extra epochs did not improve checkpoint selection

The best checkpoint occurred before the counterfactual firing point and remained unchanged through the full budget.

| Interval | Train-loss change | Validation-loss change | Disc-Dice change | Cup-Dice change |
|---|---:|---:|---:|---:|
| Epoch 81 → 101 | +0.0044 | +0.0417 | -0.0028 | **-0.0891** |
| Epoch 81 → 300 | **-0.1341** | **+0.0768** | -0.0019 | **-0.0432** |

Deltas are calculated directly from the supplied sampled history values.[^rim-derived] Nothing after epoch 101 replaced epoch 81 as the selected checkpoint. The final training loss was much lower, while validation loss was higher and cup Dice lower. This is overfitting in the logged validation trajectory, not additional useful learning.

### The checkpoint differences are not established beyond noise

This is a single-seed run, so run-to-run noise is **not estimated**. The per-image test Dice standard deviations are **0.0399** for disc and **0.1896** for cup; they describe heterogeneity across test images, not uncertainty across training runs. The epoch-81-to-300 validation Dice changes, **-0.0019** and **-0.0432**, are smaller in magnitude than those test-image standard deviations.

The validation set contains only **48** images. Per-image validation scores and the complete history were not supplied, so neither a validation standard error nor epoch-to-epoch noise can be calculated. A claim that small checkpoint differences are reproducible would be unsupported.

### Failure is concentrated in cup segmentation

Cup Dice is **0.1718** below disc Dice, cup IoU is **0.2351** lower, cup HD95 is **16.35 native pixels** higher, and all three cup metrics have substantially larger per-image dispersion.[^rim-derived] The cup is therefore the clear failure structure. The supplied summary does not reveal which images or cup boundaries fail because the per-image CSV and qualitative predictions were not supplied.

### Comparison with the other supplied in-domain baseline

Relative to the supplied REFUGE plain U-Net run, RIM-ONE-DL test Dice is lower by **0.0141** for disc and **0.1006** for cup.[^cross-derived-rim] This is descriptive only. The datasets use different source framing and different HD95 units, and RIM-ONE-DL has an undetectable fellow-eye leakage risk. No same-dataset repeat seed, alternative backbone or conditioning baseline was supplied, so the result does not establish a method ranking.

## 7. Sanity check against the literature

The supplied RIM-ONE publication describes the original release, not the local RIM-ONE-DL mixture.[^rim-paper]

| Publication characteristic | RIM-ONE publication | This run |
|---|---:|---:|
| Images | 169 | 485 |
| Structures annotated | Optic disc only | Optic disc and cup |
| Expert-to-gold-standard variability | 2.5%–4.4% mean across experts | not the run's metric |
| Directional variability | 3.3%–4.2% mean across radii | not the run's metric |
| Model Dice reported | not reported | Disc **0.9412**; cup **0.7694** |

Source-publication values are from RIM-ONE.pdf, sections 3.1 and 5, Tables 1 and 2.[^rim-paper] A direct literature Dice comparison is impossible: the paper reports expert boundary variability rather than algorithmic Dice and does not provide optic-cup masks. The run uses a **485-image** r1/r2/r3-derived resource with undocumented local mask provenance in the supplied excerpts. Ground-truth consensus, evaluation procedure and mask resolution therefore differ or cannot be shown to match.

## 8. Limitations and reproducibility

- Filenames encode eye but not patient. Fellow-eye correlation between train and test is undetectable rather than known to be absent. The flat random **70/10/20** split may inflate in-domain performance.
- The validation set contains only **48** images. Per-image validation dispersion and the complete epoch history are missing, so checkpoint uncertainty and epoch-to-epoch noise are not quantified.
- Only one seed was run. Per-image test standard deviations do not estimate run-to-run variability.
- The source images are already optic-nerve-head crops. Performance does not include full-fundus localisation and is not directly comparable with a full-image pipeline.
- The supplied source paper describes an earlier **169-image**, disc-only release, while the run uses a **485-image** r1/r2/r3 mixture. The provenance and consensus construction of the local cup masks are not documented in the supplied materials.
- HD95 is expressed in native-source pixels after scale correction. It must not be compared numerically with REFUGE's letterboxed-grid-pixel HD95.
- Deep supervision was deferred, so this is a single-head comparator.
- Final-epoch test metrics were not supplied, preventing quantification of the test penalty associated with late validation degradation.
- The unavailable resolved configuration, test-metrics file, SLURM log and split manifest prevent an independent audit of several run settings and split details.
- The Git identifier ends in “-dirty”, so uncommitted changes were present and are not captured by the commit alone.

| Reproducibility field | Value | Source |
|---|---|---|
| Experiment | rim_one_s2_plain_unet_20260828_183100 | summary_table.txt |
| Run-directory identifier | rim-one_s2_36826181 | pasted shell output |
| Git SHA | aca9cd34e9d93eef27cbd387f1d17bb6eced3504-dirty | RUN_NOTES.md |
| Seed | **42** | RUN_NOTES.md |
| Device | CUDA | summary_table.txt |
| Training time | **1575.0 seconds** | summary_table.txt |

## Appendix A. Available per-epoch history

The supplied history excerpt contains every tenth epoch plus the best and counterfactual firing milestones. The table below reproduces every numeric epoch available in that excerpt.

| Epoch | Learning rate | Train loss | Validation loss | Validation Dice: disc | Validation Dice: cup |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.000e-03 | 1.1437 | 1.0246 | 0.9217 | 0.3071 |
| 10 | 9.973e-04 | 0.3855 | 0.3713 | 0.9320 | 0.7127 |
| 20 | 9.891e-04 | 0.3004 | 0.3039 | 0.9384 | 0.7228 |
| 30 | 9.756e-04 | 0.2792 | 0.2657 | 0.9426 | 0.7574 |
| 40 | 9.568e-04 | 0.2412 | 0.2510 | 0.9462 | 0.7508 |
| 50 | 9.331e-04 | 0.2302 | 0.2324 | 0.9500 | 0.7794 |
| 60 | 9.046e-04 | 0.2046 | 0.2341 | 0.9512 | 0.7827 |
| 70 | 8.717e-04 | 0.2016 | 0.2194 | 0.9518 | 0.7690 |
| 80 | 8.347e-04 | 0.1794 | 0.2047 | 0.9535 | 0.7990 |
| 81 | 8.308e-04 | 0.1769 | 0.1979 | 0.9553 | 0.8203 |
| 90 | 7.941e-04 | 0.1765 | 0.2212 | 0.9554 | 0.7814 |
| 100 | 7.503e-04 | 0.1629 | 0.2107 | 0.9538 | 0.7628 |
| 101 | 7.457e-04 | 0.1813 | 0.2396 | 0.9525 | 0.7312 |
| 110 | 7.037e-04 | 0.1546 | 0.2111 | 0.9525 | 0.7842 |
| 120 | 6.549e-04 | 0.1504 | 0.2126 | 0.9538 | 0.7852 |
| 130 | 6.044e-04 | 0.1348 | 0.2149 | 0.9549 | 0.7626 |
| 140 | 5.527e-04 | 0.1260 | 0.2205 | 0.9528 | 0.7658 |
| 150 | 5.005e-04 | 0.1225 | 0.2162 | 0.9532 | 0.7705 |
| 160 | 4.483e-04 | 0.1059 | 0.2174 | 0.9560 | 0.7799 |
| 170 | 3.966e-04 | 0.1019 | 0.2335 | 0.9549 | 0.7530 |
| 180 | 3.461e-04 | 0.0868 | 0.2360 | 0.9533 | 0.7720 |
| 190 | 2.973e-04 | 0.0810 | 0.2374 | 0.9549 | 0.7756 |
| 200 | 2.507e-04 | 0.0744 | 0.2360 | 0.9552 | 0.7779 |
| 210 | 2.069e-04 | 0.0674 | 0.2450 | 0.9535 | 0.7820 |
| 220 | 1.663e-04 | 0.0608 | 0.2578 | 0.9533 | 0.7753 |
| 230 | 1.293e-04 | 0.0572 | 0.2570 | 0.9539 | 0.7757 |
| 240 | 9.640e-05 | 0.0520 | 0.2593 | 0.9541 | 0.7737 |
| 250 | 6.792e-05 | 0.0491 | 0.2696 | 0.9526 | 0.7779 |
| 260 | 4.418e-05 | 0.0468 | 0.2717 | 0.9528 | 0.7767 |
| 270 | 2.545e-05 | 0.0446 | 0.2717 | 0.9533 | 0.7769 |
| 280 | 1.192e-05 | 0.0433 | 0.2734 | 0.9534 | 0.7783 |
| 290 | 3.736e-06 | 0.0430 | 0.2739 | 0.9533 | 0.7773 |
| 300 | 1.000e-06 | 0.0428 | 0.2747 | 0.9534 | 0.7771 |

[^brief-rim]: Edward_Project_Brief.pdf, section 6, Step 2.
[^rim-notes]: Pasted RUN_NOTES.md for rim-one_s2_36826181.
[^rim-summary]: Pasted summary_table.txt for rim-one_s2_36826181.
[^rim-history]: Pasted “RIM-ONE — every 10th epoch, plus milestones” history excerpt.
[^rim-paper]: RIM-ONE.pdf, sections 3.1 and 5, Tables 1 and 2.
[^rim-derived]: Arithmetic from values in the pasted RIM-ONE summary and history excerpt.
[^cross-derived-rim]: Arithmetic from the two pasted run summaries.
