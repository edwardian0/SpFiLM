# Step 2 in-domain baseline report: REFUGE Zeiss

  

**Evidence boundary.** This report uses the pasted summary_table.txt, RUN_NOTES.md and sampled history output, together with the project brief and dataset publication. Configuration fields, epoch rows and checkpoint results absent from those sources are omitted rather than represented by placeholders.

  

## 1. Objective

  

This run establishes the plain U-Net in-domain segmentation baseline on the REFUGE Zeiss domain. It belongs to Step 2 of the project sequence: validate a reproducible plain backbone on a single source before measuring cross-domain degradation or adding Global FiLM and Spatial FiLM.[^brief]

  

The run therefore answers whether the unconditioned pipeline can learn optic-disc and optic-cup masks from the REFUGE Training400 source. It does not test domain adaptation and provides no evidence yet that spatial conditioning helps.

  

## 2. Dataset and split

  

The run uses only refuge_zeiss, identified in RUN_NOTES.md as REFUGE Training400 acquired with a Zeiss Visucam camera.[^refuge-notes] The source publication describes this provider partition as **400** images acquired at **2124 × 2056** pixels.[^refuge-paper-data]

  

| Split      | Images |
| ---------- | -----: |
| Train      |    256 |
| Validation |     64 |
| Test       |     80 |
| Total      |    400 |

  

Source: split image counts are from summary_table.txt.[^refuge-summary]

  

The provider fixed the 400-image Zeiss Training400 partition, but the **256/64/80** train/validation/test subdivision is locally derived rather than provider-defined. The supplied excerpts do not identify who fixed the local subdivision or document its allocation, stratification or grouping procedure.

  

The complete fundus image is used; there is no optic-nerve-head ROI crop. Each **2124 × 2056** image is aspect-preservingly resized to **512 × 496** and centre-pasted on a **512 × 512** canvas, leaving vertical zero padding (RUN_NOTES.md).[^refuge-notes] Augmentation transforms and parameters are omitted because resolved_config.json was not supplied. The log summary records **0 of 76,800** augmented samples requiring cup-within-disc repair and **0** repaired pixels; the configured repair is cup &= disc (summary_table.txt).[^refuge-summary]

  

## 3. Model and training setup

  

The table below contains the complete set of model and training fields supported by the supplied run excerpts.

  

| Parameter                  | Run setting                                                                                                                             | Source                                        |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------- |
| Architecture               | Plain 2D U-Net; no conditioning; two-channel sigmoid head for disc and cup; InstanceNorm with affine parameters after every convolution | RUN_NOTES.md                                  |
| Output supervision         | Single head and single-tensor loss; deep supervision deferred                                                                           | RUN_NOTES.md                                  |
| Trainable parameters       | **1,944,066**                                                                                                                           | summary_table.txt                             |
| Epoch budget               | **300**                                                                                                                                 | pasted history heading and final logged epoch |
| Epochs run                 | **300**                                                                                                                                 | pasted history heading and final logged epoch |
| Input resolution           | **512 × 512**                                                                                                                           | RUN_NOTES.md                                  |
| Epoch-1 learning rate      | **1.000e-03**                                                                                                                           | sampled history output                        |
| Final logged learning rate | **1.000e-06**                                                                                                                           | sampled history output                        |
| Loss                       | Equal-weight BCEWithLogits plus soft Dice                                                                                               | RUN_NOTES.md                                  |
| Seed                       | **42**                                                                                                                                  | RUN_NOTES.md                                  |
| Hard-mask threshold        | **0.5**                                                                                                                                 | summary_table.txt                             |
| Device                     | CUDA                                                                                                                                    | summary_table.txt                             |

  

### Training protocol and checkpoint selection

  

The run consumed the full configured budget: **300 of 300 epochs** were logged.[^refuge-history] Early stopping did not terminate training. It acted as a counterfactual rule alongside best-checkpoint selection.

  

| Early-stopping field | Setting |

|---|---|

| Selection metric | Lowest validation BCE plus soft Dice loss |

| Counterfactual firing epoch | **88** |

| Global best epoch after the full run | **94** |

  

Sources: selection metric and best epoch are from summary_table.txt; the counterfactual firing epoch is from the pasted history heading.[^refuge-summary][^refuge-history]

  

Under the previous terminating protocol this run would have ended at epoch **88**; the exact then-best checkpoint cannot be identified from the supplied history excerpt. Continuing the non-terminating protocol allowed epoch **94** to become the full-run best checkpoint. The reported test metrics come from that epoch-94 checkpoint, selected by minimum validation loss (summary_table.txt).

  

## 4. Per-epoch results

  

### Milestone epochs

  

The table retains the numeric milestones available in the pasted history sample: the first epoch, sampled intervals, the counterfactual firing epoch, the best epoch and the final epoch. Unsupplied rows are omitted.

  

| Epoch | Learning rate | Train loss | Validation loss | Validation Dice: disc | Validation Dice: cup |
| ----: | ------------: | ---------: | --------------: | --------------------: | -------------------: |
|     1 |     1.000e-03 |     1.6062 |          1.5371 |                0.8564 |               0.0145 |
|    50 |     9.331e-04 |     0.0980 |          0.1088 |                0.9442 |               0.8640 |
|    88 |     8.025e-04 |     0.0802 |          0.1123 |                0.9472 |               0.8508 |
|    94 |     7.769e-04 |     0.0790 |          0.0957 |                0.9549 |               0.8737 |
|   100 |     7.503e-04 |     0.0769 |          0.0965 |                0.9537 |               0.8722 |
|   150 |     5.005e-04 |     0.0641 |          0.1059 |                0.9501 |               0.8657 |
|   200 |     2.507e-04 |     0.0502 |          0.1110 |                0.9464 |               0.8596 |
|   250 |     6.792e-05 |     0.0385 |          0.1180 |                0.9452 |               0.8513 |
|   300 |     1.000e-06 |     0.0351 |          0.1182 |                0.9455 |               0.8523 |

  

Source: pasted sampled history output.[^refuge-history]

  

The main optimisation gain occurred early. Between epochs **1 and 30**, validation loss fell from **1.5371 to 0.1189**, while validation disc Dice rose from **0.8564 to 0.9461** and cup Dice from **0.0145 to 0.8523**; the logged LR moved from **1.000e-03 to 9.756e-04**.[^refuge-history] Validation then fluctuated while the LR declined. The global minimum validation loss occurred at epoch **94**: **0.0957**, with disc Dice **0.9549**, cup Dice **0.8737** and LR **7.769e-04**.

  

After epoch 94, training loss continued down to **0.0351** at epoch 300, but validation loss rose to **0.1182**. Validation disc and cup Dice ended at **0.9455** and **0.8523**, respectively, while the LR reached **1.000e-06**. The late phase therefore reduced training error without improving the selected validation objective.

  

## 5. Test results

  

Metrics were computed on the **512 × 512 aspect-preserving letterboxed full-image grid**. Predictions were not resampled to native resolution, so HD95 is in letterboxed-grid pixels, not native pixels or millimetres (summary_table.txt).[^refuge-summary] Disc and cup are reported separately; **no combined disc-plus-cup Dice is reported**.

  

| Checkpoint     | Structure |                Dice |                 IoU |               HD95 |            Accuracy |
| -------------- | --------- | ------------------: | ------------------: | -----------------: | ------------------: |
| Best, epoch 94 | Disc      | **0.9553 ± 0.0325** | **0.9160 ± 0.0532** | **4.40 ± 5.38 px** | **0.9986 ± 0.0008** |
| Best, epoch 94 | Cup       | **0.8700 ± 0.0601** | **0.7749 ± 0.0924** | **4.85 ± 2.24 px** | **0.9989 ± 0.0007** |

  

Source: best-checkpoint means and standard deviations over **80** test images are from summary_table.txt; all HD95 counts were **80** with **0** exclusions.[^refuge-summary] Final-epoch test metrics were not included in the supplied excerpts, so the best-versus-final test gap cannot be calculated.

  

The summary's degenerate-case policy retains smoothed near-zero Dice and IoU for an empty prediction against a non-empty target; both-empty Dice and IoU equal **1.0**; undefined HD95 values are excluded rather than set to zero. No test image was excluded from either structure's HD95 summary (summary_table.txt).

  

## 6. Findings

  

### Extra epochs changed the selected checkpoint, but only briefly

  

Continuing past the counterfactual firing point mattered for checkpoint selection. Epoch 94, six logged epochs after epoch 88, became the global best. The pointwise change from the firing epoch to the later best was:

  

| Interval       | Train-loss change | Validation-loss change | Disc-Dice change | Cup-Dice change |
| -------------- | ----------------: | ---------------------: | ---------------: | --------------: |
| Epoch 88 → 94  |           -0.0012 |            **-0.0166** |      **+0.0077** |     **+0.0229** |
| Epoch 94 → 300 |       **-0.0439** |                +0.0225 |          -0.0094 |         -0.0214 |

  

Deltas are calculated directly from the supplied sampled history values.[^refuge-derived] The first row is not the gain over the checkpoint that the old protocol would have selected, because that checkpoint epoch is unavailable. It shows that the rule would have fired during a temporary validation degradation and missed the later recovery to the global minimum. Epochs after 94 did not improve the selector and ended with worse validation overlap despite lower training loss.

  

### The observed checkpoint differences are not established beyond noise

  

This is a single-seed run, so run-to-run noise is **not estimated**. The test-set standard deviations are per-image dispersion, not run-to-run variance. The epoch-88-to-94 validation Dice changes, **0.0077** for disc and **0.0229** for cup, are smaller than the test-image Dice standard deviations, **0.0325** and **0.0601**. That comparison is descriptive only because it mixes validation checkpoint changes with dispersion on a different set.

  

The validation set contains **64** images, but per-image validation scores and the complete epoch history were not supplied. A standard error and epoch-to-epoch noise therefore cannot be calculated. No claim that epoch 94 is reliably superior across resampling or repeated runs is supported.

  

### The cup remains the weaker structure

  

On test data, cup Dice is **0.0853** below disc Dice, cup IoU is **0.1411** lower, and cup Dice dispersion is larger.[^refuge-derived] Cup HD95 is also **0.45** grid pixels higher, although its standard deviation is lower. The artefacts show a structure-specific weakness in cup delineation; they do not identify retinal locations or image subgroups where errors occur because no qualitative outputs or per-image CSV were supplied.

  

### Comparison with the other supplied in-domain baseline

  

The only other same-protocol baseline supplied in this request is the plain U-Net on RIM-ONE-DL. REFUGE test Dice is higher by **0.0141** for disc and **0.1006** for cup.[^cross-derived] This is a cross-dataset difference, not a method comparison: the image crops, populations, masks, split risks and HD95 frames differ. No same-dataset alternative architecture, repeat seed or conditioning baseline was supplied, so this run cannot establish superiority over another model.

  

## 7. Sanity check against the literature

  

The REFUGE publication reports challenge-test Dice for several methods.[^refuge-paper-results]

  

| Comparator from REFUGE publication | Disc Dice | Difference: this run minus publication | Cup Dice | Difference: this run minus publication |
|---|---:|---:|---:|---:|
| CUHKMED, overall rank 1 | 0.9602 | -0.0049 | 0.8826 | -0.0126 |
| Masker, best published cup Dice | 0.9464 | +0.0089 | 0.8837 | -0.0137 |
| BUCT, classical U-Net entry | 0.9525 | +0.0028 | 0.8728 | -0.0028 |
| This run, plain U-Net | **0.9553** | — | **0.8700** | — |

  

Differences are direct subtractions from the listed run and publication values.[^refuge-derived] The run sits in the published challenge score range and is close to the BUCT entry, but the comparison is only approximate. The challenge figures use the fixed Canon CR-2 test partition, whose source images are **1634 × 1634**, whereas this run uses a derived hold-out from **2124 × 2056** Zeiss Training400.[^refuge-paper-data] The publication's reference masks were produced by majority vote across seven glaucoma specialists followed by senior quality control; the provenance of the local masks is not documented in the supplied run excerpts, so equivalence of the ground-truth consensus cannot be verified. The challenge used its own submission and evaluation procedure, whereas this run thresholds at **0.5** and evaluates on a **512 × 512** letterboxed grid. The resolution at which each challenge team generated its submitted mask was method-specific and is not established by Table 6. Ground-truth consensus, evaluation procedure and mask resolution therefore differ or cannot be shown to match.

  

## 8. Limitations and reproducibility

  

- Only one seed was run. The report contains per-image test variability but no run-to-run variability.

- The **64-image** validation set is modest, and neither per-image validation scores nor the complete history.csv were supplied. Epoch-to-epoch noise is not quantified.

- The local split policy, class counts and patient/eye grouping cannot be audited without split_manifest.csv.

- Full-image letterboxing introduces zero padding and reduces the native **2124 × 2056** image to **512 × 496** content. HD95 is therefore tied to the letterboxed grid.

- The REFUGE publication states that its data contain selected high-quality images from Chinese patients. This limits generalisation beyond the source population and acquisition quality.[^refuge-paper-data]

- Deep supervision was deferred, so this is a single-head comparator and not the intended later deep-supervised configuration.

- Final-epoch test metrics were not supplied, preventing quantification of the test penalty from late validation degradation.

- The unavailable resolved configuration, test-metrics file, SLURM log and split manifest prevent an independent audit of several run settings and split details.

- The Git identifier ends in “-dirty”, so uncommitted changes were present and are not captured by the commit alone.

  

| Reproducibility field | Value | Source |
|---|---|---|
| Experiment | refuge_s2_plain_unet_20260828_181214 | summary_table.txt |
| Run-directory identifier | refuge_s2_36826166 | pasted shell output |
| Git SHA | aca9cd34e9d93eef27cbd387f1d17bb6eced3504-dirty | RUN_NOTES.md |
| Seed | **42** | RUN_NOTES.md |
| Device | CUDA | summary_table.txt |
| Training time | **1855.6 seconds** | summary_table.txt |

  

## Appendix A. Available per-epoch history

  

The supplied history excerpt contains every tenth epoch plus the best and counterfactual firing milestones. The table below reproduces every numeric epoch available in that excerpt.

  

| Epoch | Learning rate | Train loss | Validation loss | Validation Dice: disc | Validation Dice: cup |
|------:|--------------:|-----------:|----------------:|----------------------:|---------------------:|
| 1 | 1.000e-03 | 1.6062 | 1.5371 | 0.8564 | 0.0145 |
| 10 | 9.973e-04 | 0.7703 | 0.7353 | 0.9070 | 0.6683 |
| 20 | 9.891e-04 | 0.1712 | 0.1813 | 0.9279 | 0.8145 |
| 30 | 9.756e-04 | 0.1197 | 0.1189 | 0.9461 | 0.8523 |
| 40 | 9.568e-04 | 0.1075 | 0.1242 | 0.9427 | 0.8473 |
| 50 | 9.331e-04 | 0.0980 | 0.1088 | 0.9442 | 0.8640 |
| 60 | 9.046e-04 | 0.0942 | 0.1043 | 0.9476 | 0.8695 |
| 70 | 8.717e-04 | 0.0915 | 0.1081 | 0.9468 | 0.8621 |
| 80 | 8.347e-04 | 0.0829 | 0.1102 | 0.9474 | 0.8574 |
| 88 | 8.025e-04 | 0.0802 | 0.1123 | 0.9472 | 0.8508 |
| 90 | 7.941e-04 | 0.0787 | 0.1080 | 0.9499 | 0.8577 |
| 94 | 7.769e-04 | 0.0790 | 0.0957 | 0.9549 | 0.8737 |
| 100 | 7.503e-04 | 0.0769 | 0.0965 | 0.9537 | 0.8722 |
| 110 | 7.037e-04 | 0.0753 | 0.1029 | 0.9497 | 0.8646 |
| 120 | 6.549e-04 | 0.0728 | 0.0977 | 0.9528 | 0.8722 |
| 130 | 6.044e-04 | 0.0668 | 0.1014 | 0.9552 | 0.8608 |
| 140 | 5.527e-04 | 0.0659 | 0.1002 | 0.9529 | 0.8685 |
| 150 | 5.005e-04 | 0.0641 | 0.1059 | 0.9501 | 0.8657 |
| 160 | 4.483e-04 | 0.0611 | 0.1040 | 0.9506 | 0.8652 |
| 170 | 3.966e-04 | 0.0604 | 0.1097 | 0.9470 | 0.8604 |
| 180 | 3.461e-04 | 0.0541 | 0.1191 | 0.9446 | 0.8467 |
| 190 | 2.973e-04 | 0.0539 | 0.1081 | 0.9481 | 0.8575 |
| 200 | 2.507e-04 | 0.0502 | 0.1110 | 0.9464 | 0.8596 |
| 210 | 2.069e-04 | 0.0464 | 0.1165 | 0.9476 | 0.8486 |
| 220 | 1.663e-04 | 0.0441 | 0.1135 | 0.9482 | 0.8561 |
| 230 | 1.293e-04 | 0.0420 | 0.1151 | 0.9475 | 0.8546 |
| 240 | 9.640e-05 | 0.0409 | 0.1178 | 0.9448 | 0.8481 |
| 250 | 6.792e-05 | 0.0385 | 0.1180 | 0.9452 | 0.8513 |
| 260 | 4.418e-05 | 0.0376 | 0.1181 | 0.9461 | 0.8503 |
| 270 | 2.545e-05 | 0.0370 | 0.1176 | 0.9459 | 0.8528 |
| 280 | 1.192e-05 | 0.0357 | 0.1183 | 0.9456 | 0.8521 |
| 290 | 3.736e-06 | 0.0357 | 0.1179 | 0.9456 | 0.8525 |
| 300 | 1.000e-06 | 0.0351 | 0.1182 | 0.9455 | 0.8523 |

  

[^brief]: Edward_Project_Brief.pdf, section 6, Step 2.

[^refuge-notes]: Pasted RUN_NOTES.md for refuge_s2_36826166.

[^refuge-summary]: Pasted summary_table.txt for refuge_s2_36826166.

[^refuge-history]: Pasted “REFUGE — every 10th epoch, plus milestones” history excerpt.

[^refuge-paper-data]: REFUGE-Challenge.pdf, section 3.1 and Table 2.

[^refuge-paper-results]: REFUGE-Challenge.pdf, Table 6.

[^refuge-derived]: Arithmetic from values in the pasted REFUGE summary and history excerpt.

[^cross-derived]: Arithmetic from the two pasted run summaries.