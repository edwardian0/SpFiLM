# Step 2 in-domain baseline report: DRISHTI-GS

**Evidence boundary.** This report uses the supplied SLURM terminal record, including its resolved configuration, 300 epoch lines, split summary and best-checkpoint test summary, together with Edward_Project_Brief.pdf and DRISHTI-GS.pdf. Values are not carried over from the REFUGE or RIM-ONE reports.

## 1. Objective

This run establishes the plain U-Net in-domain segmentation baseline on DRISHTI-GS. It is part of Step 2 of the project sequence: validate a reproducible unconditioned backbone on one source before measuring cross-domain degradation or adding Global FiLM and Spatial FiLM.[^brief]

The run tests whether the common segmentation pipeline can learn optic-disc and optic-cup masks within the DRISHTI-GS domain. It does not test adaptation and provides no evidence by itself about whether spatial conditioning helps.

  

## 2. Dataset and split

DRISHTI-GS contains **101** fundus images. The provider defines **50** training images and **51** test images; the run retains the provider test set and divides the 50-image training partition into 40 training and 10 validation images.[^paper-data][^run]


| Split      | Images | Glaucoma | Normal | Policy                                       |
| ---------- | -----: | -------: | -----: | -------------------------------------------- |
| Train      |     40 |       26 |     14 | Derived from the provider training partition |
| Validation | **10** |        6 |      4 | Derived from the provider training partition |
| Test       |     51 |       38 |     13 | Provider-defined test partition              |
| Total      |    101 |       70 |     31 | —                                            |


The run seed is **42**.

The publication released ground truth only for the **50 training images**. The local run has masks for all **51 test images**, but their provenance is undocumented in the supplied material.[^paper-data] This affects the interpretation of every test result below.

## 3. Model and training setup

| Parameter                     | Run setting                                                                 | Source                                  |
| ----------------------------- | --------------------------------------------------------------------------- | --------------------------------------- |
| Architecture                  | Plain U-Net; base width **16**; no Global FiLM or Spatial FiLM conditioning | Resolved configuration and run note     |
| Outputs                       | Two masks: optic disc and optic cup                                         | Logged target shape and metric outputs  |
| Trainable parameters          | **1,944,066**                                                               | Test summary                            |
| Deep supervision              | Deferred; single-head, single-tensor loss                                   | Run note                                |
| Epoch budget and completion   | **300 of 300** epochs                                                       | Training log                            |
| Batch size                    | **8**                                                                       | Resolved configuration                  |
| Data-loader workers           | **8**                                                                       | Resolved configuration                  |
| Input size                    | **512 × 512**                                                               | Resolved configuration                  |
| Initial learning rate         | **0.001**                                                                   | Resolved configuration                  |
| Final logged learning rate    | **1.000e-06**                                                               | Epoch 300                               |
| Weight decay                  | **1e-05**                                                                   | Resolved configuration                  |
| Loss                          | BCE plus soft Dice                                                          | Checkpoint-selection description        |
| Hard-mask threshold           | **0.5**                                                                     | Resolved configuration and test summary |
| Seed                          | **42**                                                                      | Resolved configuration                  |
| Horizontal-flip probability   | **0.5**                                                                     | Resolved configuration                  |
| Rotation                      | **10.0 degrees**                                                            | Resolved configuration                  |
| Brightness/contrast parameter | **0.1**                                                                     | Resolved configuration                  |
| Device                        | CUDA on **NVIDIA A100-PCIE-40GB**                                           | SLURM header                            |
| LR Scheduler                  | CosineAnnealingLR                                                           |                                         |

  

The record omits the optimiser class and the scheduler's formal name, T_max and eta_min, so none is inferred. The logged learning rate decreases continuously from **1.000e-03** at epoch 1 to **1.000e-06** at epoch 300.

The first augmented training batch had disc and cup foreground fractions of **0.02218** and **0.01234** of the canvas, respectively. Cup-within-disc repair was needed for **0 of 12,000** augmented samples and changed **0 pixels**.[^run]

  

### Training protocol and checkpoint selection

Training used monitoring-only early stopping: it consumed the full 300-epoch budget and retained the rule as a counterfactual and checkpoint monitor.

| Early-stopping field                                        | Setting                                   |
| ----------------------------------------------------------- | ----------------------------------------- |
| Selection metric                                            | Lowest validation BCE plus soft Dice loss |
| Mode                                                        | Monitor                                   |
| min_delta                                                   | **1e-05**                                 |
| Patience                                                    | **20** epochs                             |
| Minimum epochs                                              | **30**                                    |
| Counterfactual firing epoch                                 | **186**                                   |
| Checkpoint selected under the previous terminating protocol | **166**                                   |
| Full-run best checkpoint                                    | **259**                                   |


Source: resolved configuration, epoch markers and terminal training summary.[^run][^history] Under the previous terminating protocol this run would have ended at epoch **186**, selecting the epoch-**166** checkpoint. Under the current protocol it completed epoch 300 and selected epoch **259** by minimum combined validation loss. The printed test metrics come from the epoch-259 best checkpoint.

  

## 4. Per-epoch results

The table reports epoch 1 followed by every tenth epoch, as requested.


| Epoch | Learning rate | Train loss | Validation loss | Validation Dice: disc | Validation Dice: cup |
| ----: | ------------: | ---------: | --------------: | --------------------: | -------------------: |
|     1 |     1.000e-03 |     1.7012 |          1.6280 |                0.3854 |               0.0237 |
|    10 |     9.973e-04 |     1.4484 |          1.4420 |                0.9018 |               0.0624 |
|    20 |     9.891e-04 |     1.2969 |          1.2888 |                0.9299 |               0.4252 |
|    30 |     9.756e-04 |     1.1416 |          1.1332 |                0.9221 |               0.5494 |
|    40 |     9.568e-04 |     0.9858 |          0.9790 |                0.9388 |               0.7122 |
|    50 |     9.331e-04 |     0.8391 |          0.8358 |                0.9271 |               0.6950 |
|    60 |     9.046e-04 |     0.6980 |          0.7014 |                0.9233 |               0.7456 |
|    70 |     8.717e-04 |     0.5770 |          0.5798 |                0.9496 |               0.7748 |
|    80 |     8.347e-04 |     0.4749 |          0.4868 |                0.9458 |               0.7755 |
|    90 |     7.941e-04 |     0.3747 |          0.3919 |                0.9452 |               0.7577 |
|   100 |     7.503e-04 |     0.2681 |          0.2950 |                0.9424 |               0.8143 |
|   110 |     7.037e-04 |     0.1894 |          0.2335 |                0.9469 |               0.7931 |
|   120 |     6.549e-04 |     0.1620 |          0.2013 |                0.9454 |               0.8062 |
|   130 |     6.044e-04 |     0.1322 |          0.1753 |                0.9524 |               0.8144 |
|   140 |     5.527e-04 |     0.1093 |          0.1641 |                0.9513 |               0.8190 |
|   150 |     5.005e-04 |     0.0989 |          0.1851 |                0.9383 |               0.7721 |
|   160 |     4.483e-04 |     0.0884 |          0.1704 |                0.9392 |               0.7949 |
|   170 |     3.966e-04 |     0.0826 |          0.1654 |                0.9430 |               0.8003 |
|   180 |     3.461e-04 |     0.0737 |          0.1578 |                0.9457 |               0.8044 |
|   190 |     2.973e-04 |     0.0674 |          0.1605 |                0.9412 |               0.8013 |
|   200 |     2.507e-04 |     0.0620 |          0.1569 |                0.9443 |               0.8047 |
|   210 |     2.069e-04 |     0.0582 |          0.1568 |                0.9468 |               0.7994 |
|   220 |     1.663e-04 |     0.0546 |          0.1593 |                0.9420 |               0.7978 |
|   230 |     1.293e-04 |     0.0523 |          0.1510 |                0.9497 |               0.8006 |
|   240 |     9.640e-05 |     0.0485 |          0.1524 |                0.9503 |               0.7996 |
|   250 |     6.792e-05 |     0.0483 |          0.1524 |                0.9508 |               0.7981 |
|   260 |     4.418e-05 |     0.0458 |          0.1510 |                0.9506 |               0.8007 |
|   270 |     2.545e-05 |     0.0450 |          0.1517 |                0.9506 |               0.7995 |
|   280 |     1.192e-05 |     0.0459 |          0.1513 |                0.9503 |               0.7999 |
|   290 |     3.736e-06 |     0.0446 |          0.1518 |                0.9504 |               0.7991 |
|   300 |     1.000e-06 |     0.0449 |          0.1518 |                0.9504 |               0.7992 |

  

Source: the 300-line training block in the supplied SLURM record.[^history]

  

The main optimisation gain occurred in the first third of training. From epoch **1 to 100**, validation loss fell from **1.6280 to 0.2950**, disc Dice rose from **0.3854 to 0.9424**, and cup Dice rose from **0.0237 to 0.8143**. Cup learning was delayed relative to disc, with the largest early change occurring after epoch 19.

  

After epoch 100, the validation loss continued to decline more slowly while both structure-specific Dice scores fluctuated. The minimum combined loss occurred at epoch **259**. Training loss then remained close to its floor, while validation loss ended at **0.1518** and the learning rate reached **1.000e-06**.

  

## 5. Test results

  

The test metrics are computed on the **512 × 512 aspect-preserving letterboxed full-image grid**. Predictions are not resampled to native resolution, so HD95 is in letterboxed-grid pixels, not native pixels or millimetres.[^test] Disc and cup are reported separately; **no combined disc-plus-cup Dice is reported**.

  

| Checkpoint      | Structure |                Dice |                 IoU |                 HD95 |            Accuracy |
| --------------- | --------- | ------------------: | ------------------: | -------------------: | ------------------: |
| Best, epoch 259 | Disc      | **0.9519 ± 0.0611** | **0.9132 ± 0.0860** |  **8.64 ± 27.87 px** | **0.9973 ± 0.0049** |
| Best, epoch 259 | Cup       | **0.8209 ± 0.1164** | **0.7112 ± 0.1522** | **13.98 ± 28.25 px** | **0.9956 ± 0.0037** |

  

Source: mean ± standard deviation over **51** per-image test values; HD95 excluded **0 of 51** images for both structures.[^test] The terminal output evaluates only the selected best checkpoint. Although a final-epoch checkpoint was written, final-epoch test metrics are not printed, so a best-versus-final test gap cannot be calculated.

  

For degenerate cases, empty predictions against non-empty targets retain their near-zero smoothed Dice and IoU; both-empty Dice and IoU equal **1.0**; undefined HD95 values are excluded rather than replaced with zero. No HD95 exclusions occurred in this run.[^test]

  

## 6. Findings

  

### Extra epochs improved the scalar loss, not segmentation overlap

  

Continuing beyond the counterfactual stop changed the selected checkpoint from epoch 166 to epoch 259.

  

| Comparison                         | Validation loss | Disc Dice | Cup Dice |
| ---------------------------------- | --------------: | --------: | -------: |
| Epoch 166, old-protocol checkpoint |          0.1542 |    0.9525 |   0.8060 |
| Epoch 259, full-run minimum loss   |      **0.1495** |    0.9510 |   0.8040 |
| Change, 166 → 259                  |     **-0.0047** |   -0.0015 |  -0.0020 |

  

Source: direct differences from the training history.[^derived] The additional budget improved the checkpoint-selection loss by **0.0047**, but neither validation Dice score improved. The extra epochs therefore changed the scalar optimum without producing better overlap on either structure.

  

### The checkpoint objective is misaligned with the harder structure

  

Validation cup Dice reached its maximum at epoch **109**, while the combined loss selected epoch 259.

  

| Candidate                             |   Epoch | Validation loss |  Disc Dice |    Cup Dice | Macro disc/cup Dice |
| ------------------------------------- | ------: | --------------: | ---------: | ----------: | ------------------: |
| Maximum cup and macro Dice            | **109** |          0.2291 | **0.9512** |  **0.8294** |          **0.8903** |
| Minimum combined loss                 | **259** |      **0.1495** |     0.9510 |      0.8040 |              0.8775 |
| Selected-minus-cup-optimum difference |       — |         -0.0796 |    -0.0002 | **-0.0254** |             -0.0128 |
| Final epoch                           |     300 |          0.1518 |     0.9504 |      0.7992 |              0.8748 |

  

Source: direct values and arithmetic from the training history.[^derived] The loss-selected checkpoint gives up **0.0254 validation cup Dice** relative to the cup-aware checkpoint and gains no disc Dice. By epoch 300, cup Dice has settled at **0.7992**, **0.0302** below its observed peak.

  

The supplied engine.py observation identifies the combined validation loss as disc-dominated. This run shows the practical consequence: the selector continues to reward a lower scalar after cup overlap has peaked. Cup is both the harder structure and the clinically consequential structure for cup-to-disc assessment; the DRISHTI-GS paper specifically describes cup segmentation as less mature, highly variable between observers and central to estimating glaucomatous cupping.[^paper-data]

  

If the same disc-dominated selector is used for every dataset and model arm, it imposes the same structural bias against cup-aware checkpoint selection throughout the comparison. That is a protocol-level risk, although the realised Dice cost must be measured separately for each run. A pre-specified macro disc/cup Dice or other cup-aware selector should be applied consistently across all arms; minimum combined loss can remain a secondary checkpoint.

  

### The 10-image validation set makes the exact optimum unstable

  

Only **10 images** determine every checkpoint decision. Using the test-set cup dispersion only as an approximate scale gives **0.1164 / √10 = 0.0368 Dice**, or roughly **0.04**.[^noise] This is not a formal validation standard error because the per-image validation scores are unavailable, but it shows the uncertainty expected from a sample this small. The observed **0.0254** cup-Dice difference between epochs 109 and 259 is smaller than that uncertainty scale, so checkpoint selection is close to arbitrary.

  

The complete history nevertheless shows that cup is noisier than disc after the model has broadly converged:

  

| Metric, epochs 100–300 | Standard deviation of one-epoch change | Mean absolute one-epoch change |
|---|---:|---:|
| Validation loss | 0.0046 | 0.0031 |
| Validation Dice: disc | 0.0036 | 0.0023 |
| Validation Dice: cup | **0.0110** | **0.0075** |

  

Source: calculations over the **200** consecutive epoch transitions from epoch 100 through 300.[^noise] The cup-selector gap is larger than ordinary single-epoch jitter, but the small validation sample and selection over many epochs prevent a reliable claim that epoch 109 would generalise better. The robust conclusion is objective misalignment, not certainty about one optimal epoch.

  

There is also no empirical run-to-run estimate because this is a single-seed run. The test Dice standard deviations, **0.0611** for disc and **0.1164** for cup, measure variation between images rather than between training runs.

  

### Failure is concentrated in cup segmentation

  

On the selected checkpoint, cup Dice is **0.1310** below disc Dice, cup IoU is **0.2020** lower, and cup HD95 is **5.34 grid pixels** higher.[^derived] Cup Dice and IoU also have substantially larger per-image standard deviations. The structure-level evidence therefore identifies cup delineation as the main weakness. The supplied summary does not provide the per-image rows or qualitative predictions needed to identify specific anatomical locations or image subgroups.

  

This is the plain U-Net reference for DRISHTI-GS. No competing model on the same split is present in this run record, so it cannot establish a model ranking. Cross-dataset comparisons with REFUGE and RIM-ONE are not method comparisons because acquisition, crop geometry, split risk and test-mask provenance differ.

  

## 7. Sanity check against the literature

  

The DRISHTI-GS source publication reports test-set F-scores of **0.96** for disc and **0.79** for cup, with boundary-localisation errors of **8.93** and **25.48 pixels**, respectively.[^paper-results] For binary masks, F1 and Dice are algebraically equivalent overlap measures, but only under the same masks and binarisation.

  

| Structure | This run: Dice | Publication: test F-score | Overlap difference | This run: HD95 | Publication: boundary localisation |
|---|---:|---:|---:|---:|---:|
| Disc | **0.9519** | 0.96 | -0.0081 | 8.64 grid px | 8.93 px |
| Cup | **0.8209** | 0.79 | +0.0309 | 13.98 grid px | 25.48 px |


The overlap scores are in the same broad range, with this run slightly below the publication for disc and above it for cup. The comparison remains approximate for three reasons:

- the publication forms ground truth from four experts and thresholds the soft consensus map at **0.75**, corresponding to agreement by at least three experts, whereas the provenance of the local 51 test masks is undocumented.

- the publication's F-score and radial boundary-localisation procedure differ from this run's hard Dice and HD95 implementation.

- this run evaluates at 512 × 512 after letterboxing, whereas the publication describes a much larger released fundus image. The boundary numbers must therefore not be compared as if they were the same metric or pixel frame.

  

## 8. Limitations and reproducibility

  

- The validation set contains only **10 images**. Checkpoint estimates are noisy, and the exact epoch ranking is unstable.

- The source publication released masks for the 50 training images only. The provenance of the local masks for all 51 test images is undocumented.

- The checkpoint metric is a disc-dominated combined loss. It selected an epoch with lower cup and macro Dice than the observed maxima.

- Only one seed was run. Test-set standard deviations do not estimate run-to-run variability.

- The source publication states that poor-quality images were discarded and all images came from one hospital. Generalisation to lower-quality or multi-centre data is therefore untested.

- The run operates on a 512 × 512 letterboxed grid. HD95 is not in native pixels or millimetres.

- The logged native-size example conflicts with the DRISHTI-GS publication and should be corrected in the run metadata before final archival.

- Deep supervision was deferred, so this is a single-head comparator.

- Final-epoch test metrics are absent, preventing a direct test-set measurement of late-checkpoint degradation.

- The optimiser class and formal LR-scheduler metadata are absent from the supplied terminal record.

- The Git state was dirty, so the commit identifier alone does not capture all code used for the run.

  

| Reproducibility field | Value                                          | Source                 |
| --------------------- | ---------------------------------------------- | ---------------------- |
| Experiment            | drishti_s2_plain_unet_20260829_112931          | SLURM record           |
| Job ID                | **36839728**                                   | SLURM header           |
| Host                  | erc-hpc-comp057                                | SLURM record           |
| Git SHA               | aca9cd34e9d93eef27cbd387f1d17bb6eced3504-dirty | SLURM record           |
| Seed                  | **42**                                         | Resolved configuration |
| GPU                   | NVIDIA A100-PCIE-40GB                          | SLURM header           |
| Python                | **3.11.15**                                    | SLURM record           |
| PyTorch               | **2.4.1+cu121**                                | SLURM record           |
| Training time         | **1991.5 seconds**                             | Test summary           |

  

[^brief]: Edward_Project_Brief.pdf, section 6, Step 2.

[^run]: Pasted markdown(2).md, drishti_s2_36839728.out configuration, data and run-summary blocks.

[^history]: Pasted markdown(2).md, 300-line training block.

[^test]: Pasted markdown(2).md, TEST RESULTS block.

[^paper-data]: DRISHTI-GS.pdf, sections 1 and 3.

[^paper-results]: DRISHTI-GS.pdf, Table 1 and section 5.

[^derived]: Arithmetic from the supplied DRISHTI-GS training and test values.

[^noise]: Derived from all consecutive validation values between epochs 100 and 300 in the supplied training block.