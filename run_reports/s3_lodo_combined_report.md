# Stage 3 leave-one-domain-out baseline: combined five-seed report

**Evidence boundary.** This interim report uses the supplied `aggregate_stage3.py` terminal output for 20 scientific runs, the supplied per-seed Dice and prediction-count diagnostic, the earlier REFUGE-Zeiss pilot log in `Pasted text.txt`, `Edward_Project_Brief.pdf`, and the three dataset publications. The aggregate does not contain the 20 resolved configurations, histories, split manifests or run-level metadata. The earlier pilot is kept separate because its test scores do not match the final aggregate's REFUGE-Zeiss seed-42 scores.

## 1. Objective

Stage 3 measures the loss of segmentation performance when a plain model is applied to an unseen acquisition domain without adaptation. The project brief defines the protocol as leave one domain out (LODO): train on all named domains except one, test on the held-out domain, and rotate the held-out domain.[^brief]

This experiment ran the plain U-Net arm over four held-out domains and five seeds per fold, giving **20 scientific runs**.[^aggregate] It establishes that the LODO machinery executes and aggregates successfully. It does **not** yet establish a valid four-domain baseline: three folds produce non-degenerate cross-domain segmentations, whereas the RIM-ONE-DL fold is a repeatable qualitative failure associated with a field-of-view mismatch.

## 2. Datasets and locked LODO folds

Each test set was locked and reused across seeds. The training sources in the table follow directly from the LODO rule and the four domains named by the aggregate.[^aggregate]

| Held-out domain | Training domains | Locked test images | HD95 frame |
|---|---|---:|---|
| `drishti_gs` | `refuge_zeiss`, `refuge_canon_val`, `rim_one_dl` | **51** | Letterboxed-grid pixels |
| `refuge_canon_val` | `drishti_gs`, `refuge_zeiss`, `rim_one_dl` | **80** | Letterboxed-grid pixels |
| `refuge_zeiss` | `drishti_gs`, `refuge_canon_val`, `rim_one_dl` | **80** | Letterboxed-grid pixels |
| `rim_one_dl` | `drishti_gs`, `refuge_canon_val`, `refuge_zeiss` | **97** | Native-source pixels |

The aggregate identifies the test sets as locked but does not include the manifests, training/validation counts or class counts. It therefore does not permit an audit of the realised split assignments or whether each internal split is provider-defined or derived.

The supplied data audit reports a marked framing mismatch. RIM-ONE-DL consists of **524 × 524** optic-disc-centred crops, whereas REFUGE uses full posterior-pole images of **2124 × 2056** pixels for the Zeiss subset and **1634 × 1634** pixels for the Canon subsets.[^audit][^refuge-paper] The original RIM-ONE publication independently confirms that its optic-nerve-head images were manually cropped from full fundus photographs.[^rim-paper] This difference is central to the RIM-ONE-DL result in Section 6.

## 3. Model and training setup

The settings that can be verified from the supplied combined output are:

| Parameter | Confirmed setting |
|---|---|
| Experimental arm | `stage3_lodo_plain_unet` |
| Architecture family | Plain U-Net |
| Conditioning or adaptation | None |
| Evaluation protocol | Four-fold leave-one-domain-out |
| Seeds | **42, 43, 44, 45 and 46** |
| Scientific runs | **20** |
| Test-set policy | Same locked images for every seed within a fold |
| Reported structures | Optic disc and optic cup, separately |
| Reported metrics | Dice, IoU and HD95 |
| Between-arm test | Not run; only one arm is present |

Source: combined aggregation output.[^aggregate] The exact U-Net width and parameter count, batch size, optimiser, loss, weight decay, augmentation parameters, input transform, epoch budget, learning-rate schedule, checkpoint rule and hardware cannot be verified for the 20 scientific runs from the aggregate alone. Those fields must be populated from each run's `resolved_config.json`, `history.csv`, `test_metrics.json` and SLURM log before this becomes the archival supervisor report.


## 4. Held-out test results

The following means, standard deviations and 95% confidence intervals are calculated across the five seed-level means, not across individual images.[^aggregate] Dice and IoU are unitless. HD95 is in letterboxed-grid pixels for DRISHTI-GS and both REFUGE folds, but in native-source pixels for RIM-ONE-DL. Disc and cup are separate; no combined disc-plus-cup Dice is reported.

### 4.1 Dice

| Held-out domain | Test images | Disc Dice, mean ± seed SD | 95% CI | Cup Dice, mean ± seed SD | 95% CI |
|---|---:|---:|---:|---:|---:|
| `drishti_gs` | 51 | **0.7589 ± 0.111** | [0.6207, 0.8972] | **0.6350 ± 0.024** | [0.6052, 0.6649] |
| `refuge_canon_val` | 80 | **0.8684 ± 0.0669** | [0.7854, 0.9515] | **0.7184 ± 0.0788** | [0.6206, 0.8162] |
| `refuge_zeiss` | 80 | **0.8834 ± 0.0178** | [0.8614, 0.9055] | **0.7139 ± 0.0385** | [0.6661, 0.7616] |
| `rim_one_dl` | 97 | **0.0537 ± 0.0143** | [0.0359, 0.0715] | **0.0247 ± 0.0272** | [-0.0090, 0.0584] |

The negative lower limit for RIM-ONE-DL cup Dice is an untruncated five-seed confidence interval, not a physically possible Dice value.

### 4.2 IoU and HD95

| Held-out domain | Structure | IoU, mean ± seed SD | 95% CI | HD95, mean ± seed SD | 95% CI | Unit |
|---|---|---:|---:|---:|---:|---|
| `drishti_gs` | Disc | 0.6882 ± 0.123 | [0.5352, 0.8412] | 80.19 ± 40.3 | [30.11, 130.27] | Grid px |
| `drishti_gs` | Cup | 0.4992 ± 0.0262 | [0.4667, 0.5317] | 48.93 ± 18.8 | [25.55, 72.32] | Grid px |
| `refuge_canon_val` | Disc | 0.7938 ± 0.0752 | [0.7005, 0.8871] | 35.50 ± 26.9 | [2.05, 68.94] | Grid px |
| `refuge_canon_val` | Cup | 0.5894 ± 0.0825 | [0.4869, 0.6919] | 19.85 ± 12.1 | [4.86, 34.84] | Grid px |
| `refuge_zeiss` | Disc | 0.8109 ± 0.024 | [0.7811, 0.8408] | 25.19 ± 13.4 | [8.59, 41.79] | Grid px |
| `refuge_zeiss` | Cup | 0.5761 ± 0.0468 | [0.5180, 0.6342] | 17.46 ± 5.31 | [10.86, 24.05] | Grid px |
| `rim_one_dl` | Disc | 0.0279 ± 0.00756 | [0.0185, 0.0373] | 184.07 ± 19.9 | [159.30, 208.84] | Native px |
| `rim_one_dl` | Cup | 0.0135 ± 0.0152 | [-0.0054, 0.0325] | 241.30 ± 35.7 | [197.01, 285.60] | Native px |

Source: combined aggregation output.[^aggregate] Disc HD95 was finite for every test image in every seed. Cup HD95 was also complete for the three non-RIM folds. For RIM-ONE-DL, it was finite for **94–97 of 97** images per seed, and only **93 of 97** images were finite in every seed. Restricting to those common images gave cup HD95* of **239.23 ± 34.2 native pixels**, with a 95% CI of **[196.83, 281.64]**.[^aggregate]

### 4.3 Per-seed Dice

Each cell is disc Dice / cup Dice.[^seed-table]

| Held-out domain | Seed 42 | Seed 43 | Seed 44 | Seed 45 | Seed 46 |
|---|---:|---:|---:|---:|---:|
| `drishti_gs` | 0.7812 / 0.6266 | 0.7916 / 0.6514 | 0.9161 / 0.6661 | 0.6327 / 0.6269 | 0.6730 / 0.6044 |
| `refuge_canon_val` | 0.9004 / 0.7028 | 0.8974 / 0.7460 | 0.7491 / 0.5893 | 0.9049 / 0.7896 | 0.8904 / 0.7641 |
| `refuge_zeiss` | 0.8644 / 0.7011 | 0.8880 / 0.7004 | 0.9064 / 0.7792 | 0.8917 / 0.6779 | 0.8667 / 0.7107 |
| `rim_one_dl` | 0.0321 / 0.0140 | 0.0638 / 0.0728 | 0.0571 / 0.0163 | 0.0475 / 0.0141 | 0.0680 / 0.0062 |

## 5. Findings

### 5.1 The runner works, but the four-domain baseline does not

All 20 jobs produced aggregatable scientific results on locked test sets. The pipeline substrate therefore works mechanically. Scientifically, only three folds currently behave as plausible cross-domain baselines. RIM-ONE-DL disc Dice of **0.0537 ± 0.0143** and cup Dice of **0.0247 ± 0.0272** are not ordinary degradation; both structures have failed across every seed.

The failure is repeatable rather than seed noise. RIM-ONE-DL disc Dice ranges only from **0.0321 to 0.0680** across seeds. The other three folds lie far above it even at their weakest individual seeds.[^seed-table]

### 5.2 RIM-ONE-DL predictions are severely undersized

The seed-42 pixel-count diagnostic rules out the simplest claim that every prediction is empty. It found zero fully empty disc or cup predictions among the **97** images. However, disc predictions contained **415,439** foreground pixels in total against **10,471,225** target foreground pixels; cup predictions contained **160,507** foreground pixels against **2,817,516** target foreground pixels. False negatives were **10,300,758** for disc and **2,803,880** for cup.[^rim-diagnostic] The model is therefore producing masks that are much too small and miss most of both targets. The incomplete cup HD95 coverage in the other seeds is also consistent with some empty cup predictions.

The leading explanation is the field-of-view mismatch: the model held out from RIM-ONE-DL was trained only on full-fundus domains, then tested on tight ONH crops in which disc and cup occupy a radically different fraction of the canvas. The image audit and source publications support this mechanism. They do not prove causality. Confirmation requires a pre-specified, harmonised framing policy followed by a rerun.

This fold should remain in the record as a protocol failure, but it should not be averaged into a headline measure of model generalisation or used to judge Global FiLM against Spatial FiLM. Any apparent gain on this fold could otherwise reflect recovery from a crop-scale mismatch rather than the intended domain-conditioning effect.

### 5.3 Two of the remaining folds are not stable across seeds

DRISHTI-GS disc Dice spans **0.6327 to 0.9161**, a range of **0.2834**, with seed SD **0.111**. Its cup Dice is substantially more stable at seed SD **0.024**. The reversal—disc varying much more than cup—is atypical for these results and is driven by high disc performance at seed 44 and weak disc performance at seeds 45 and 46.[^seed-table][^derived]

REFUGE Canon is dominated by one weak run: seed 44 gives disc/cup Dice of **0.7491/0.5893**, while the other four disc values lie between **0.8904 and 0.9049**. REFUGE Zeiss is the most stable fold for disc, with seed SD **0.0178**, although its cup result still ranges from **0.6779 to 0.7792**.[^seed-table]

These are not small run-to-run fluctuations. The checkpoint histories, selected epochs, prediction-area distributions and validation trajectories for the outlying seeds must be inspected before the three non-RIM folds are called a stable baseline.

### 5.4 Cup remains the harder structure where segmentation is non-degenerate

For DRISHTI-GS, REFUGE Canon and REFUGE Zeiss, mean cup Dice is respectively **0.1239**, **0.1500** and **0.1695** below mean disc Dice.[^derived] This is the consistent structure-level failure. RIM-ONE-DL is not interpretable through the same gap because both outputs have collapsed.

### 5.5 Stage 3 is not yet complete against the brief's move-on criterion

The project brief requires a table of in-domain versus cross-domain Dice with the gap quantified before moving on.[^brief] The supplied aggregate contains only cross-domain results, so the exact Stage 2-to-Stage 3 gaps cannot be calculated without carrying values from earlier reports. The corrected LODO rerun should be joined directly to the exact Step 2 artefacts before declaring Stage 3 complete.

The aggregate also cannot answer whether training beyond each counterfactual stopping point improved the final scientific checkpoints, because the 20 histories and selected epochs are absent. The separate pilot suggests late scalar overfitting, but it is not a substitute for those histories.

### 5.6 Immediate protocol decision

Before beginning the conditioning-arm comparison, fix one evaluation policy and apply it unchanged to all arms:

1. Harmonise spatial framing, for example with a leakage-safe image-based ONH localisation/cropping procedure applied to every domain; or
2. Treat RIM-ONE-DL as a separate crop-regime stress test rather than one fold of the primary full-fundus LODO rotation.

Then rerun the plain arm from a clean commit, verify the outlying DRISHTI-GS and REFUGE Canon seeds, and only then run Global FiLM and Spatial FiLM on the same locked folds. The aggregate correctly did not run a paired test because only the plain arm exists.[^aggregate]

## 6. Sanity check against the literature

The literature check supports the preprocessing diagnosis rather than a direct performance comparison. The REFUGE publication states that its images show the posterior pole with both macula and optic disc visible, and reports the Zeiss and Canon resolutions given in Section 2.[^refuge-paper] The RIM-ONE publication states explicitly that its ONH images were manually cropped from full fundus images.[^rim-paper] A failure specific to holding out RIM-ONE-DL is therefore consistent with a learned object-scale and framing prior.

The numerical segmentation results should not be compared directly with the dataset papers. Their ground-truth consensus procedures, evaluation measures and mask resolutions differ from this pipeline. DRISHTI-GS adds a specific provenance limitation: its publication released ground truth for the **50 training images**, while the local Stage 3 evaluation uses **51 test images** whose mask provenance is undocumented in the supplied evidence.[^drishti-paper][^aggregate]

The project brief identifies DoFE as the closest prior LODO fundus-segmentation comparison.[^brief] No DoFE result table was supplied with these artefacts, so no numerical literature claim is introduced here. That comparison should be made only after the spatial-framing defect is resolved.

## 7. Limitations and reproducibility

- The RIM-ONE-DL fold measures an unresolved crop-scale mismatch, not a clean acquisition-domain shift.
- DRISHTI-GS and REFUGE Canon show large seed sensitivity; the aggregate alone cannot identify whether checkpoint selection, optimisation or prediction scale caused the outlying runs.
- The aggregate reports seed-level standard deviations. It does not provide per-image standard deviations, accuracy, or final-epoch test results.
- With only **five seeds**, the 95% confidence intervals are wide and sensitive to a single outlying seed. The negative RIM cup lower bounds are artefacts of the unbounded interval calculation.
- HD95 units differ: RIM-ONE-DL uses native pixels and the other folds use letterboxed-grid pixels. HD95 must not be averaged across folds.
- RIM-ONE-DL filenames encode eye but not patient, so fellow-eye correlation is undetectable rather than known to be absent. This can affect train-validation separation whenever RIM-ONE-DL is a source domain.[^rim-caveat]
- The DRISHTI-GS test-mask provenance is undocumented, as described in Section 7.
- Validation-set sizes and epoch-to-epoch noise for the 20 scientific runs are absent from the aggregate. They must be quantified before any claim that one checkpoint is superior to another.
- All **20 runs** were produced from a dirty working tree. The aggregate warning therefore states that the recorded commits do not fully identify the code that ran.[^aggregate]
- The earlier REFUGE-Zeiss pilot and final aggregate disagree for seed 42. Their run identities must be reconciled before histories are attached to final results.

| Reproducibility field | Available value |
|---|---|
| Arm | `stage3_lodo_plain_unet` |
| Held-out domains | `drishti_gs`, `refuge_canon_val`, `refuge_zeiss`, `rim_one_dl` |
| Seeds | **42–46** |
| Scientific runs | **20** |
| Aggregate JSON | `artifacts/stage3_lodo_summary.json` |
| Aggregate CSV | `artifacts/stage3_lodo_summary.csv` |
| Generated source report | `run_reports/s3_lodo_combined_report.md` |
| Repository state | Dirty for every aggregated run |

Source: aggregation command and warning.[^aggregate] The final archival report must add each run's Git SHA, job ID, GPU, exact configuration, selected checkpoint and full epoch history from the individual artefacts.

[^brief]: `Edward_Project_Brief.pdf`, Section 5 and Step 3 (“Measure the shift”).
[^aggregate]: Supplied `aggregate_stage3.py` terminal output, covering 20 runs across four held-out domains and five seeds.
[^audit]: Field-of-view audit in the supplied diagnostic commentary.
[^refuge-paper]: `REFUGE-Challenge.pdf`, Table 2 and dataset-description text.
[^rim-paper]: `RIM-ONE.pdf`, Section 3.1.
[^pilot]: `Pasted text.txt`, REFUGE-Zeiss seed-42 pilot SLURM output.
[^seed-table]: Supplied per-seed Dice terminal output.
[^pilot-derived]: Direct arithmetic from epochs 126 and 300 in `Pasted text.txt`.
[^rim-diagnostic]: Supplied RIM-ONE-DL seed-42 `tp`, `fp`, `fn` and empty-prediction diagnostic; foreground totals are direct sums of `tp + fp` and `tp + fn`.
[^derived]: Direct arithmetic from the aggregate and per-seed Dice values.
[^drishti-paper]: `DRISHTI-GS.pdf`, Sections 3 and 3.2.
[^rim-caveat]: Standing RIM-ONE-DL dataset caveat supplied by the author.
