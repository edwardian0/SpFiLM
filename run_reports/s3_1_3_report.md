# Stage 3 train-on-one, test-on-three report: plain U-Net

**Evidence boundary.** Every figure below is computed by `aggregate_stage3_1_3.py` directly from the per-image metric CSVs of 20 completed runs, validated against the locked budgeted manifest, and recomputed from those CSVs rather than read from any run's stored summary. The pooled `test_pooled` block in each run is deliberately ignored: it averages three acquisition domains and is not a per-domain result. Sections marked TODO require judgement the tool does not make.

## 1. Protocol

Each run trains on a single acquisition domain under a fixed labelled budget of **40 training** and **10 validation** images drawn from that domain's own locked partitions, then scores every other domain separately on **50 held-out test images** each. The source domain's own test partition is excluded, so no image ever changes role.

The budget is the Drishti-GS floor: it has both the fewest training and the fewest test images, so capping every domain to it makes the cells comparable. Without the cap, pooled training volume would swing with which domain was used and confound the comparison.

| Setting | Value |
|---|---|
| Arm | `stage3_single_source_plain_unet` |
| Source domains | `drishti_gs`, `refuge_canon_val`, `refuge_zeiss`, `rim_one_dl` |
| Seeds | 42, 43, 44, 45, 46 |
| Runs | 20 |
| Train / val / test budget | 40 / 10 / 50 |
| Checkpoint selection | lowest validation loss on the source domain |
| Manifest | `single_source_manifest.json` (`af4fa639ce10…`) |
| Parent LODO manifest | `c77c35376a56…` |
| Git revision | `1aa9c6fbc65ca40fbb3c2925808cf3f8a3e4c377-dirty` |

## 2. Cross-domain Dice matrix

Rows are the single training domain; columns are the unseen target domain. Each cell is the mean over the per-seed run means, plus or minus the seed-level standard deviation. The diagonal is empty because a domain is never a target of its own fold.

### 2.1 Optic disc

| Trained on ↓ / tested on → | `drishti_gs` | `refuge_canon_val` | `refuge_zeiss` | `rim_one_dl` | Mean over targets |
|---|---:|---:|---:|---:|---:|
| `drishti_gs` | — | 0.8582 ± 0.0134 | 0.8372 ± 0.0108 | 0.1043 ± 0.0344 | **0.5999** |
| `refuge_canon_val` | 0.8793 ± 0.0225 | — | 0.8913 ± 0.0098 | 0.0710 ± 0.0110 | **0.6138** |
| `refuge_zeiss` | 0.7273 ± 0.0790 | 0.7327 ± 0.0722 | — | 0.0826 ± 0.0583 | **0.5142** |
| `rim_one_dl` | 0.1207 ± 0.0027 | 0.0886 ± 0.0074 | 0.0789 ± 0.0050 | — | **0.0961** |

### 2.2 Optic cup

| Trained on ↓ / tested on → | `drishti_gs` | `refuge_canon_val` | `refuge_zeiss` | `rim_one_dl` | Mean over targets |
|---|---:|---:|---:|---:|---:|
| `drishti_gs` | — | 0.6661 ± 0.0569 | 0.5598 ± 0.0292 | 0.1331 ± 0.1200 | **0.4530** |
| `refuge_canon_val` | 0.7033 ± 0.0357 | — | 0.7601 ± 0.0199 | 0.0773 ± 0.0483 | **0.5136** |
| `refuge_zeiss` | 0.4745 ± 0.0387 | 0.6696 ± 0.0870 | — | 0.0574 ± 0.0243 | **0.4005** |
| `rim_one_dl` | 0.4157 ± 0.0962 | 0.1576 ± 0.0610 | 0.1640 ± 0.0531 | — | **0.2458** |

## 3. Full per-cell results

Dice and IoU are unitless. HD95 is in letterboxed-grid pixels for every target except RIM-ONE-DL, which is in native source pixels; the two are never averaged together.

| Source | Target | Structure | Images | Dice, mean ± seed SD | 95% CI | IoU | HD95 | HD95 unit |
|---|---|---|---:|---:|---:|---:|---:|---|
| `drishti_gs` | `refuge_canon_val` | disc | 50 | **0.8582 ± 0.0134** | [0.8416, 0.8748] | 0.7719 ± 0.0172 | 27.95 ± 6.40 | Grid px |
| `drishti_gs` | `refuge_canon_val` | cup | 50 | **0.6661 ± 0.0569** | [0.5955, 0.7367] | 0.5266 ± 0.0662 | 24.50 ± 5.19 | Grid px |
| `drishti_gs` | `refuge_zeiss` | disc | 50 | **0.8372 ± 0.0108** | [0.8238, 0.8505] | 0.7481 ± 0.0133 | 27.51 ± 5.43 | Grid px |
| `drishti_gs` | `refuge_zeiss` | cup | 50 | **0.5598 ± 0.0292** | [0.5234, 0.5961] | 0.4057 ± 0.0296 | 26.14 ± 3.77 | Grid px |
| `drishti_gs` | `rim_one_dl` | disc | 50 | **0.1043 ± 0.0344** | [0.0615, 0.1471] | 0.0559 ± 0.0188 | 174.56 ± 18.00 | Native px |
| `drishti_gs` | `rim_one_dl` | cup | 50 | **0.1331 ± 0.1200** | [-0.0159, 0.2820] | 0.0838 ± 0.0786 | 166.51 ± 67.26 | Native px |
| `refuge_canon_val` | `drishti_gs` | disc | 50 | **0.8793 ± 0.0225** | [0.8513, 0.9073] | 0.7983 ± 0.0354 | 47.45 ± 14.73 | Grid px |
| `refuge_canon_val` | `drishti_gs` | cup | 50 | **0.7033 ± 0.0357** | [0.6590, 0.7477] | 0.5636 ± 0.0421 | 31.59 ± 17.13 | Grid px |
| `refuge_canon_val` | `refuge_zeiss` | disc | 50 | **0.8913 ± 0.0098** | [0.8792, 0.9034] | 0.8206 ± 0.0126 | 27.67 ± 6.90 | Grid px |
| `refuge_canon_val` | `refuge_zeiss` | cup | 50 | **0.7601 ± 0.0199** | [0.7354, 0.7849] | 0.6293 ± 0.0254 | 13.40 ± 3.16 | Grid px |
| `refuge_canon_val` | `rim_one_dl` | disc | 50 | **0.0710 ± 0.0110** | [0.0573, 0.0846] | 0.0370 ± 0.0059 | 172.71 ± 13.49 | Native px |
| `refuge_canon_val` | `rim_one_dl` | cup | 50 | **0.0773 ± 0.0483** | [0.0174, 0.1372] | 0.0453 ± 0.0296 | 179.30 ± 36.36 | Native px |
| `refuge_zeiss` | `drishti_gs` | disc | 50 | **0.7273 ± 0.0790** | [0.6291, 0.8254] | 0.6132 ± 0.0894 | 85.70 ± 32.12 | Grid px |
| `refuge_zeiss` | `drishti_gs` | cup | 50 | **0.4745 ± 0.0387** | [0.4265, 0.5225] | 0.3364 ± 0.0322 | 43.24 ± 5.40 | Grid px |
| `refuge_zeiss` | `refuge_canon_val` | disc | 50 | **0.7327 ± 0.0722** | [0.6431, 0.8223] | 0.6261 ± 0.0758 | 93.88 ± 22.16 | Grid px |
| `refuge_zeiss` | `refuge_canon_val` | cup | 50 | **0.6696 ± 0.0870** | [0.5616, 0.7777] | 0.5464 ± 0.0827 | 41.33 ± 23.91 | Grid px |
| `refuge_zeiss` | `rim_one_dl` | disc | 50 | **0.0826 ± 0.0583** | [0.0102, 0.1550] | 0.0443 ± 0.0330 | 173.30 ± 7.72 | Native px |
| `refuge_zeiss` | `rim_one_dl` | cup | 50 | **0.0574 ± 0.0243** | [0.0273, 0.0876] | 0.0327 ± 0.0143 | 167.66 ± 40.61 | Native px |
| `rim_one_dl` | `drishti_gs` | disc | 50 | **0.1207 ± 0.0027** | [0.1173, 0.1240] | 0.0644 ± 0.0015 | 214.39 ± 3.19 | Grid px |
| `rim_one_dl` | `drishti_gs` | cup | 50 | **0.4157 ± 0.0962** | [0.2962, 0.5352] | 0.2794 ± 0.0823 | 99.22 ± 42.59 | Grid px |
| `rim_one_dl` | `refuge_canon_val` | disc | 50 | **0.0886 ± 0.0074** | [0.0794, 0.0978] | 0.0465 ± 0.0041 | 271.20 ± 9.93 | Grid px |
| `rim_one_dl` | `refuge_canon_val` | cup | 50 | **0.1576 ± 0.0610** | [0.0819, 0.2333] | 0.0899 ± 0.0388 | 154.32 ± 44.28 | Grid px |
| `rim_one_dl` | `refuge_zeiss` | disc | 50 | **0.0789 ± 0.0050** | [0.0727, 0.0851] | 0.0411 ± 0.0027 | 311.39 ± 12.22 | Grid px |
| `rim_one_dl` | `refuge_zeiss` | cup | 50 | **0.1640 ± 0.0531** | [0.0980, 0.2300] | 0.0936 ± 0.0334 | 134.77 ± 54.37 | Grid px |

## 4. Per-seed disc Dice

| Source | Target | Seed 42 | Seed 43 | Seed 44 | Seed 45 | Seed 46 |
|---|---|---:|---:|---:|---:|---:|
| `drishti_gs` | `refuge_canon_val` | 0.8404 | 0.8696 | 0.8473 | 0.8666 | 0.8672 |
| `drishti_gs` | `refuge_zeiss` | 0.8263 | 0.8350 | 0.8428 | 0.8527 | 0.8289 |
| `drishti_gs` | `rim_one_dl` | 0.0576 | 0.0800 | 0.1137 | 0.1342 | 0.1359 |
| `refuge_canon_val` | `drishti_gs` | 0.8595 | 0.8791 | 0.9031 | 0.8542 | 0.9004 |
| `refuge_canon_val` | `refuge_zeiss` | 0.8747 | 0.9000 | 0.8923 | 0.8963 | 0.8932 |
| `refuge_canon_val` | `rim_one_dl` | 0.0580 | 0.0780 | 0.0853 | 0.0631 | 0.0704 |
| `refuge_zeiss` | `drishti_gs` | 0.6143 | 0.7000 | 0.7713 | 0.8250 | 0.7257 |
| `refuge_zeiss` | `refuge_canon_val` | 0.6120 | 0.7265 | 0.7933 | 0.7795 | 0.7522 |
| `refuge_zeiss` | `rim_one_dl` | 0.1791 | 0.0385 | 0.0346 | 0.0768 | 0.0839 |
| `rim_one_dl` | `drishti_gs` | 0.1187 | 0.1221 | 0.1218 | 0.1171 | 0.1236 |
| `rim_one_dl` | `refuge_canon_val` | 0.0790 | 0.0973 | 0.0882 | 0.0842 | 0.0943 |
| `rim_one_dl` | `refuge_zeiss` | 0.0717 | 0.0843 | 0.0770 | 0.0788 | 0.0828 |

## 5. HD95 completeness

These cells had at least one image with an undefined HD95, which happens when a prediction or a target is empty. Their HD95 means are taken over the finite subset only and are not comparable with cells where every image was finite.

| Source | Target | Structure | Excluded images (all seeds) |
|---|---|---|---:|
| `refuge_zeiss` | `drishti_gs` | cup | 3 |
| `refuge_zeiss` | `refuge_canon_val` | cup | 4 |
| `refuge_zeiss` | `rim_one_dl` | cup | 2 |

## 6. Findings

<!-- TODO: written by hand; the tool does not infer this. -->

## 7. Comparison against the literature

<!-- TODO: written by hand; the tool does not infer this. -->

## 8. Limitations

- Each cell rests on 50 test images and 5 seeds; the confidence intervals are over seeds, not images.
- Training uses only 40 images, far below what either dataset's own baseline used, so absolute Dice is not comparable with the Stage 2 in-domain numbers.
- RIM-ONE-DL HD95 is in native source pixels while every other target is in letterboxed-grid pixels.
<!-- TODO: written by hand; the tool does not infer this. -->

