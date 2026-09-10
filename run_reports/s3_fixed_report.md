# Stage 3 fixed-budget leave-one-domain-out: plain U-Net

**Evidence boundary.** Every figure below is computed by `aggregate_stage3_fixed.py` directly from the per-image metric CSVs of 20 fixed-budget runs and 20 train-on-one runs, validated against the shared budgeted manifest, and recomputed from those CSVs rather than read from any stored summary. Sections marked TODO require judgement the tool does not make.

## 1. Protocol

Leave-one-domain-out under a fixed labelled budget. For each held-out domain the model trains on the other three domains' budgeted partitions — **40 images from each, 120 in total** — validates on their pooled **30**, and is scored on the held-out domain's **50** locked test images with no adaptation.

The budget is the Drishti-GS floor. Capping every domain to it removes the confound in the original full-data arm, whose pooled training set swung between 552 and 852 images depending on which domain was dropped, and whose test sets ranged from 51 to 97 images.

| Setting | Value |
|---|---|
| Arm | `stage3_lodo_fixed_budget_plain_unet` |
| Seeds | 42, 43, 44, 45, 46 |
| Runs | 20 |
| Train / val / test | 120 / 30 / 50 |
| Checkpoint selection | lowest pooled source validation loss |
| Manifest | `single_source_manifest.json` (`af4fa639ce10…`) |

## 2. Held-out results per domain

Means over the per-seed run means, with the seed-level standard deviation and a 95% confidence interval. Dice and IoU are unitless. HD95 is in letterboxed-grid pixels except for RIM-ONE-DL, which is in native source pixels; the two are never averaged together. Disc and cup are separate throughout.

| Held-out domain | Structure | Images | Dice, mean ± seed SD | 95% CI | IoU | HD95 | HD95 unit |
|---|---|---:|---:|---:|---:|---:|---|
| `drishti_gs` | disc | 50 | **0.7853 ± 0.0386** | [0.7373, 0.8332] | 0.6872 ± 0.0433 | 67.44 ± 33.30 | Grid px |
| `drishti_gs` | cup | 50 | **0.6461 ± 0.0432** | [0.5924, 0.6998] | 0.4951 ± 0.0508 | 24.93 ± 6.54 | Grid px |
| `refuge_canon_val` | disc | 50 | **0.8370 ± 0.0487** | [0.7765, 0.8975] | 0.7500 ± 0.0564 | 44.75 ± 19.22 | Grid px |
| `refuge_canon_val` | cup | 50 | **0.6486 ± 0.0650** | [0.5680, 0.7293] | 0.5071 ± 0.0711 | 22.93 ± 9.52 | Grid px |
| `refuge_zeiss` | disc | 50 | **0.8212 ± 0.0207** | [0.7955, 0.8470] | 0.7275 ± 0.0224 | 34.49 ± 9.23 | Grid px |
| `refuge_zeiss` | cup | 50 | **0.5729 ± 0.0110** | [0.5593, 0.5866] | 0.4184 ± 0.0103 | 25.70 ± 4.33 | Grid px |
| `rim_one_dl` | disc | 50 | **0.0737 ± 0.0415** | [0.0222, 0.1252] | 0.0391 ± 0.0232 | 176.77 ± 12.36 | Native px |
| `rim_one_dl` | cup | 50 | **0.0174 ± 0.0140** | [-0.0000, 0.0348] | 0.0093 ± 0.0076 | 215.97 ± 28.82 | Native px |

## 3. Paired test: 120 training images against 40

The brief requires a paired significance test on per-image Dice over the same test images. Both arms draw from the same budgeted manifest, so for each held-out domain they score the identical 50 images; that is asserted before each test rather than assumed. Each image's five seed scores are averaged first, so the pairs are one value per image rather than five correlated ones.

Each row compares the pooled 120-image model against one single-source 40-image model on the same held-out domain. A positive Δ means pooling three domains helped. p-values are Wilcoxon signed-rank, adjusted across all 24 tests by Holm-Bonferroni; significance is at α = 0.05 on the adjusted value.

| Held-out domain | Structure | Single source (40) | 120 Dice | 40 Dice | Δ (120−40) | p | p (Holm) | Significant |
|---|---|---|---:|---:|---:|---:|---:|:---:|
| `drishti_gs` | cup | `refuge_canon_val` | 0.6461 | 0.7033 | -0.0572 | 0.0001363 | 0.00109 | **yes** |
| `drishti_gs` | cup | `refuge_zeiss` | 0.6461 | 0.4745 | +0.1716 | 2.634e-12 | 4.478e-11 | **yes** |
| `drishti_gs` | cup | `rim_one_dl` | 0.6461 | 0.4157 | +0.2304 | 2.994e-11 | 4.791e-10 | **yes** |
| `drishti_gs` | disc | `refuge_canon_val` | 0.7853 | 0.8793 | -0.0940 | 3.007e-06 | 2.706e-05 | **yes** |
| `drishti_gs` | disc | `refuge_zeiss` | 0.7853 | 0.7273 | +0.0580 | 0.0003522 | 0.002466 | **yes** |
| `drishti_gs` | disc | `rim_one_dl` | 0.7853 | 0.1207 | +0.6646 | 1.776e-15 | 4.263e-14 | **yes** |
| `refuge_canon_val` | cup | `drishti_gs` | 0.6486 | 0.6661 | -0.0175 | 0.1232 | 0.4044 | no |
| `refuge_canon_val` | cup | `refuge_zeiss` | 0.6486 | 0.6696 | -0.0210 | 0.1074 | 0.4044 | no |
| `refuge_canon_val` | cup | `rim_one_dl` | 0.6486 | 0.1576 | +0.4910 | 1.776e-15 | 4.263e-14 | **yes** |
| `refuge_canon_val` | disc | `drishti_gs` | 0.8370 | 0.8582 | -0.0212 | 0.1011 | 0.4044 | no |
| `refuge_canon_val` | disc | `refuge_zeiss` | 0.8370 | 0.7327 | +0.1043 | 6.71e-08 | 8.723e-07 | **yes** |
| `refuge_canon_val` | disc | `rim_one_dl` | 0.8370 | 0.0886 | +0.7484 | 1.776e-15 | 4.263e-14 | **yes** |
| `refuge_zeiss` | cup | `drishti_gs` | 0.5729 | 0.5598 | +0.0132 | 0.02122 | 0.1061 | no |
| `refuge_zeiss` | cup | `refuge_canon_val` | 0.5729 | 0.7601 | -0.1872 | 3.404e-11 | 5.106e-10 | **yes** |
| `refuge_zeiss` | cup | `rim_one_dl` | 0.5729 | 0.1640 | +0.4089 | 1.776e-15 | 4.263e-14 | **yes** |
| `refuge_zeiss` | disc | `drishti_gs` | 0.8212 | 0.8372 | -0.0159 | 1.646e-06 | 1.646e-05 | **yes** |
| `refuge_zeiss` | disc | `refuge_canon_val` | 0.8212 | 0.8913 | -0.0701 | 4.381e-11 | 6.133e-10 | **yes** |
| `refuge_zeiss` | disc | `rim_one_dl` | 0.8212 | 0.0789 | +0.7423 | 3.553e-15 | 7.105e-14 | **yes** |
| `rim_one_dl` | cup | `drishti_gs` | 0.0174 | 0.1331 | -0.1157 | 8.882e-15 | 1.688e-13 | **yes** |
| `rim_one_dl` | cup | `refuge_canon_val` | 0.0174 | 0.0773 | -0.0599 | 3.041e-07 | 3.649e-06 | **yes** |
| `rim_one_dl` | cup | `refuge_zeiss` | 0.0174 | 0.0574 | -0.0401 | 1.063e-06 | 1.17e-05 | **yes** |
| `rim_one_dl` | disc | `drishti_gs` | 0.0737 | 0.1043 | -0.0306 | 3.002e-13 | 5.404e-12 | **yes** |
| `rim_one_dl` | disc | `refuge_canon_val` | 0.0737 | 0.0710 | +0.0027 | 0.4097 | 0.4097 | no |
| `rim_one_dl` | disc | `refuge_zeiss` | 0.0737 | 0.0826 | -0.0089 | 0.003913 | 0.02348 | **yes** |

**Counted outcome.** Of 24 paired comparisons, 9 favour the pooled 120-image model at Holm-adjusted α = 0.05, 10 favour the single 40-image model, and 5 are not separable.

## 4. Findings

<!-- TODO: written by hand; the tool does not infer this. -->

## 5. Limitations

- This is not the brief's headline comparison. The brief specifies the paired test between Global FiLM and SpFiLM; neither conditioning arm has been run. This uses the same statistical protocol on the one controlled pair that exists, training volume.
- Each cell rests on 50 test images and 5 seeds; the confidence intervals are over seeds, not images.
- The paired test is on Dice only. HD95 is excluded from pairing because its degenerate-case exclusions move from seed to seed, so it is not defined over a common image set.
- Training volume and domain diversity are confounded with each other: the 120-image arm sees three domains, the 40-image arm sees one. This design cannot separate 'more data' from 'more domains'.
<!-- TODO: written by hand; the tool does not infer this. -->

