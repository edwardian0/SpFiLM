# Step 5: Global FiLM against the plain U-Net, leave-one-domain-out

**Evidence boundary.** Every figure is computed by `aggregate_stage5_lodo.py` from the per-image metric CSVs of 15 plain and 15 Global FiLM runs, validated against the shared budgeted manifest `single_source_manifest.json`. Interpretation is written by hand.

## 1. Held-out Dice side by side

Leave-one-domain-out over 3 domains (`drishti_gs`, `refuge_canon_val`, `refuge_zeiss`): train on 2, test on the held-out one, 80 / 20 / 50 images per fold. Same backbone, folds, budget, seeds, augmentation, optimiser and test images; the arms differ only in the conditioning. Δ is FiLM minus plain on per-image Dice with seeds averaged first; p-values are wilcoxon, Holm-adjusted over 6 tests, significance at α = 0.05.

| Held-out domain | Structure | Images | Plain Dice, mean ± seed SD | FiLM Dice, mean ± seed SD | Δ (FiLM − plain) | p | p (Holm) | Significant |
|---|---|---:|---:|---:|---:|---:|---:|:---:|
| `drishti_gs` | disc | 50 | 0.8659 ± 0.0093 | 0.8107 ± 0.0253 | -0.0552 | 1.158e-08 | 5.792e-08 | **yes** |
| `drishti_gs` | cup | 50 | 0.6223 ± 0.0407 | 0.5712 ± 0.0476 | -0.0510 | 3.041e-07 | 1.216e-06 | **yes** |
| `refuge_canon_val` | disc | 50 | 0.8792 ± 0.0286 | 0.7309 ± 0.0673 | -0.1482 | 2.434e-13 | 1.46e-12 | **yes** |
| `refuge_canon_val` | cup | 50 | 0.6780 ± 0.0345 | 0.6763 ± 0.0883 | -0.0018 | 0.709 | 0.709 | no |
| `refuge_zeiss` | disc | 50 | 0.8633 ± 0.0064 | 0.8537 ± 0.0177 | -0.0096 | 0.0005735 | 0.00172 | **yes** |
| `refuge_zeiss` | cup | 50 | 0.6318 ± 0.0292 | 0.6393 ± 0.0220 | +0.0076 | 0.05936 | 0.1187 | no |

Plain arm: `stage5_lodo_fixed_budget_plain_unet_3dom`. FiLM arm: `stage5_lodo_fixed_budget_global_film_3dom`.

## 2. Did the conditioning do anything, and did the selector find it?

The held-out domain's code is decided once, from the mean colour statistics of its unlabelled reference sample (its budgeted training partition, labels unused, disjoint from the test images), as the source domain with the nearest training centroid. Selector accuracy is measured on source-domain validation images whose true domain is known, for the per-domain rule (one decision per domain) and the per-image rule. The per-image column shows how the held-out test images would individually be assigned; unanimity means the domain-level decision is uncontroversial. The fixed-code sweep scores the held-out set once under each source code: a spread near zero means the codes are interchangeable and the decision is irrelevant; a large spread with a negative 'used − best' means the rule picked a worse code than was available.

| Held-out domain | Seeds | Code used | Selector accuracy on source val (per domain / per image) | Per-image nearest source among held-out images | Fixed-code sweep spread, disc / cup | Used − best fixed code, disc / cup | Best fixed code votes, disc / cup |
|---|---|---|---:|---|---:|---:|---|
| `drishti_gs` | 5 | `refuge_zeiss` | 1.00 / 1.000 | `refuge_canon_val`: 0.0, `refuge_zeiss`: 50.0 | 0.0776 / 0.1213 | -0.0776 / -0.1213 | `refuge_canon_val`×5 / `refuge_canon_val`×5 |
| `refuge_canon_val` | 5 | `refuge_zeiss` | 1.00 / 0.900 | `drishti_gs`: 0.0, `refuge_zeiss`: 50.0 | 0.1549 / 0.0784 | -0.1549 / -0.0562 | `drishti_gs`×5 / `drishti_gs`×3, `refuge_zeiss`×2 |
| `refuge_zeiss` | 5 | `drishti_gs` | 1.00 / 1.000 | `drishti_gs`: 48.0, `refuge_canon_val`: 2.0 | 0.0365 / 0.1258 | -0.0365 / -0.1258 | `refuge_canon_val`×5 / `refuge_canon_val`×5 |

## 3. Findings

<!-- TODO: written by hand; the tool does not infer this. -->

