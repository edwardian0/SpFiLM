# Stage 3 single-source, three-domain report: plain U-Net

**Evidence boundary.** Every figure below is computed by `aggregate_stage3_1_3.py` directly from the per-image metric CSVs and validated against the locked budgeted manifest. This version retains the **15 runs** trained on the three full-fundus domains and excludes every source or target cell from the crop-centred fourth domain. The pooled `test_pooled` block is ignored because it is not a per-domain result. The brief findings interpret only the retained values.

## 1. Protocol

Each retained run trains on a single acquisition domain under a fixed labelled budget of **40 training** and **10 validation** images drawn from that domain's own locked partitions. This report then compares it on **50 held-out test images** from each of the other two retained domains. The source domain's own test partition is excluded, so no image ever changes role.

The budget is the Drishti-GS floor: it has both the fewest training and the fewest test images, so capping every domain to it makes the cells comparable. Without the cap, pooled training volume would swing with which domain was used and confound the comparison.

This is a single-source generalisation analysis that complements, but is not identical to, the project's LODO protocol, which trains on all domains except the target.[^brief] Its purpose is to expose directional transfer and identify which source-domain properties dominate performance.

| Setting | Value |
|---|---|
| Arm | `stage3_single_source_plain_unet` |
| Source domains | `drishti_gs`, `refuge_canon_val`, `refuge_zeiss` |
| Seeds | 42, 43, 44, 45, 46 |
| Retained runs | 15 |
| Train / val / test budget | 40 / 10 / 50 |
| Checkpoint selection | lowest validation loss on the source domain |
| Manifest | `single_source_manifest.json` (`af4fa639ce10…`) |
| Parent LODO manifest | `c77c35376a56…` |
| Git revision | `1aa9c6fbc65ca40fbb3c2925808cf3f8a3e4c377-dirty` |

**Brief finding.** The fixed budget removes training-set size as an explanation for differences between rows. However, the **40-image** training sets make this a deliberately data-limited diagnostic rather than a replacement for the full LODO baseline.

## 2. Cross-domain Dice matrix

Rows are the single training domain; columns are the unseen target domain. Each cell is the mean over the per-seed run means, plus or minus the seed-level standard deviation. The diagonal is empty because a domain is never a target of its own fold. Each row mean has been recomputed over its two retained targets.

### 2.1 Optic disc

| Trained on ↓ / tested on → | `drishti_gs` | `refuge_canon_val` | `refuge_zeiss` | Mean over targets |
|---|---:|---:|---:|---:|
| `drishti_gs` | — | 0.8582 ± 0.0134 | 0.8372 ± 0.0108 | **0.8477** |
| `refuge_canon_val` | 0.8793 ± 0.0225 | — | 0.8913 ± 0.0098 | **0.8853** |
| `refuge_zeiss` | 0.7273 ± 0.0790 | 0.7327 ± 0.0722 | — | **0.7300** |

**Brief finding.** REFUGE Canon is the strongest source for disc transfer, with a mean of **0.8853** over its two targets. DRISHTI-GS follows at **0.8477**, while REFUGE Zeiss is weaker and more variable at **0.7300**.

### 2.2 Optic cup

| Trained on ↓ / tested on → | `drishti_gs` | `refuge_canon_val` | `refuge_zeiss` | Mean over targets |
|---|---:|---:|---:|---:|
| `drishti_gs` | — | 0.6661 ± 0.0569 | 0.5598 ± 0.0292 | **0.6129** |
| `refuge_canon_val` | 0.7033 ± 0.0357 | — | 0.7601 ± 0.0199 | **0.7317** |
| `refuge_zeiss` | 0.4745 ± 0.0387 | 0.6696 ± 0.0870 | — | **0.5721** |

**Brief finding.** Cup transfer is weaker than disc transfer in every retained direction. REFUGE Canon remains the strongest source, with mean cup Dice **0.7317**; DRISHTI-GS reaches **0.6129**, and REFUGE Zeiss reaches **0.5721**.

## 3. Full per-cell results

Dice and IoU are unitless. HD95 is in letterboxed-grid pixels for every retained target.

| Trained on | Tested on | Structure | Images | Dice, mean ± seed SD | 95% CI | IoU | HD95 | HD95 unit |
|---|---|---|---:|---:|---:|---:|---:|---|
| `drishti_gs` | `refuge_canon_val` | disc | 50 | **0.8582 ± 0.0134** | [0.8416, 0.8748] | 0.7719 ± 0.0172 | 27.95 ± 6.40 | Grid px |
| `drishti_gs` | `refuge_canon_val` | cup | 50 | **0.6661 ± 0.0569** | [0.5955, 0.7367] | 0.5266 ± 0.0662 | 24.50 ± 5.19 | Grid px |
| `drishti_gs` | `refuge_zeiss` | disc | 50 | **0.8372 ± 0.0108** | [0.8238, 0.8505] | 0.7481 ± 0.0133 | 27.51 ± 5.43 | Grid px |
| `drishti_gs` | `refuge_zeiss` | cup | 50 | **0.5598 ± 0.0292** | [0.5234, 0.5961] | 0.4057 ± 0.0296 | 26.14 ± 3.77 | Grid px |
| `refuge_canon_val` | `drishti_gs` | disc | 50 | **0.8793 ± 0.0225** | [0.8513, 0.9073] | 0.7983 ± 0.0354 | 47.45 ± 14.73 | Grid px |
| `refuge_canon_val` | `drishti_gs` | cup | 50 | **0.7033 ± 0.0357** | [0.6590, 0.7477] | 0.5636 ± 0.0421 | 31.59 ± 17.13 | Grid px |
| `refuge_canon_val` | `refuge_zeiss` | disc | 50 | **0.8913 ± 0.0098** | [0.8792, 0.9034] | 0.8206 ± 0.0126 | 27.67 ± 6.90 | Grid px |
| `refuge_canon_val` | `refuge_zeiss` | cup | 50 | **0.7601 ± 0.0199** | [0.7354, 0.7849] | 0.6293 ± 0.0254 | 13.40 ± 3.16 | Grid px |
| `refuge_zeiss` | `drishti_gs` | disc | 50 | **0.7273 ± 0.0790** | [0.6291, 0.8254] | 0.6132 ± 0.0894 | 85.70 ± 32.12 | Grid px |
| `refuge_zeiss` | `drishti_gs` | cup | 50 | **0.4745 ± 0.0387** | [0.4265, 0.5225] | 0.3364 ± 0.0322 | 43.24 ± 5.40 | Grid px |
| `refuge_zeiss` | `refuge_canon_val` | disc | 50 | **0.7327 ± 0.0722** | [0.6431, 0.8223] | 0.6261 ± 0.0758 | 93.88 ± 22.16 | Grid px |
| `refuge_zeiss` | `refuge_canon_val` | cup | 50 | **0.6696 ± 0.0870** | [0.5616, 0.7777] | 0.5464 ± 0.0827 | 41.33 ± 23.91 | Grid px |

**Brief finding.** Dice, IoU and HD95 agree that REFUGE Canon transfers best to both retained targets. REFUGE Zeiss gives the lowest overlap and largest boundary errors, particularly when tested on REFUGE Canon.

## 4. Per-seed disc Dice

| Trained on | Tested on | Seed 42 | Seed 43 | Seed 44 | Seed 45 | Seed 46 |
|---|---|---:|---:|---:|---:|---:|
| `drishti_gs` | `refuge_canon_val` | 0.8404 | 0.8696 | 0.8473 | 0.8666 | 0.8672 |
| `drishti_gs` | `refuge_zeiss` | 0.8263 | 0.8350 | 0.8428 | 0.8527 | 0.8289 |
| `refuge_canon_val` | `drishti_gs` | 0.8595 | 0.8791 | 0.9031 | 0.8542 | 0.9004 |
| `refuge_canon_val` | `refuge_zeiss` | 0.8747 | 0.9000 | 0.8923 | 0.8963 | 0.8932 |
| `refuge_zeiss` | `drishti_gs` | 0.6143 | 0.7000 | 0.7713 | 0.8250 | 0.7257 |
| `refuge_zeiss` | `refuge_canon_val` | 0.6120 | 0.7265 | 0.7933 | 0.7795 | 0.7522 |

**Brief finding.** DRISHTI-GS and REFUGE Canon give comparatively stable disc transfer across seeds. REFUGE Zeiss is substantially more seed-sensitive, particularly when tested on DRISHTI-GS and REFUGE Canon.

## 5. HD95 completeness

These cells had at least one image with an undefined HD95, which happens when a prediction or a target is empty. Their HD95 means are taken over the finite subset only and are not comparable with cells where every image was finite.

| Trained on | Tested on | Structure | Excluded images (all seeds) |
|---|---|---|---:|
| `refuge_zeiss` | `drishti_gs` | cup | 3 |
| `refuge_zeiss` | `refuge_canon_val` | cup | 4 |

**Brief finding.** Undefined HD95 occurs only for cup predictions from REFUGE-Zeiss-trained models. This is consistent with occasional degenerate cup masks and reinforces the greater instability of that source row. The affected HD95 means describe only their finite subsets.

## 6. Findings

### 6.1 Training on DRISHTI-GS

DRISHTI-GS transfers reasonably to both REFUGE targets for disc, with **0.8582 ± 0.0134** on Canon and **0.8372 ± 0.0108** on Zeiss. Cup transfer is weaker, particularly on Zeiss at **0.5598 ± 0.0292**. Its two-target means are **0.8477** for disc and **0.6129** for cup.

### 6.2 Training on REFUGE Canon

REFUGE Canon is the strongest retained source: disc/cup Dice are **0.8793/0.7033** on DRISHTI-GS and **0.8913/0.7601** on REFUGE Zeiss. Its two-target means are **0.8853** for disc and **0.7317** for cup, with comparatively low seed variability.

### 6.3 Training on REFUGE Zeiss

REFUGE Zeiss transfers less effectively and less consistently than Canon. Disc Dice is **0.7273** on DRISHTI-GS and **0.7327** on Canon, with seed SDs of **0.0790** and **0.0722**. The Canon-to-Zeiss and Zeiss-to-Canon results are therefore markedly asymmetric despite both sources belonging to REFUGE.

### 6.4 Overall interpretation

Source-domain choice has a large and directional effect. Across all six retained transfer directions, the unweighted cell mean is **0.8210** for disc and **0.6389** for cup. REFUGE Canon is the strongest source, whereas REFUGE Zeiss is weaker and more seed-sensitive. The row means remain descriptive because each source is evaluated on a different pair of targets.

## 7. Comparison against the literature

The REFUGE publication deliberately separates the Zeiss acquisition subset from the Canon subsets to test device generalisation.[^refuge-paper] The directional difference observed here—strong Canon-to-Zeiss transfer but weaker Zeiss-to-Canon transfer—is therefore plausible, although this experiment reverses and subsamples the publication's original split.

Numerical comparison with the REFUGE and DRISHTI-GS publications would only be approximate. This experiment uses a fixed **40/10** training/validation budget, **50-image** target subsets, different ground-truth consensus procedures, and a common evaluation pipeline at a different mask resolution. The papers therefore provide context rather than directly matched benchmarks.[^drishti-paper]

## 8. Limitations

- Each cell rests on 50 test images and 5 seeds; the confidence intervals are over seeds, not images.
- Training uses only 40 images, far below what either dataset's own baseline used, so absolute Dice is not comparable with the Stage 2 in-domain numbers.
- There is no in-domain diagonal under the same **40/10/50** budget. The experiment therefore measures relative transfer but cannot separate source-model underfitting from the exact domain-shift penalty.
- The analysis is restricted to three full-fundus domains, so its conclusions do not cover crop-centred datasets.
- Mean-over-target values average different target pairs and should not be treated as a controlled source ranking.
- The REFUGE-Zeiss source is seed-sensitive, and its cup HD95 excludes degenerate cases in both target cells.
- Checkpoint selection uses only 10 validation images, so the selected epoch may be unstable.
- Only the plain U-Net arm is included; these results do not compare conditioning methods.
- The DRISHTI-GS publication released ground truth for its 50 training images only; the provenance of the local masks used for its test images is undocumented.[^drishti-paper]
- All runs share a dirty Git revision, so the recorded SHA does not fully identify the executed code.

[^brief]: `Edward_Project_Brief.pdf`, Step 3 and the leave-one-domain-out protocol.
[^refuge-paper]: `REFUGE-Challenge.pdf`, dataset description.
[^drishti-paper]: `DRISHTI-GS.pdf`, Sections 3 and 3.2.
