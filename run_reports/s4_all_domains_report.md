# Step 4: train on all domains, test on each — Global FiLM against the plain U-Net

**Evidence boundary.** Every figure is computed by `aggregate_stage4_all_domains.py` from the per-image metric CSVs of 5 plain and 5 Global FiLM runs, validated against the shared budgeted manifest `single_source_manifest.json`. Interpretation is written by hand.

## 1. Protocol

One model per seed trained on the pooled budgeted training partitions of `drishti_gs`, `refuge_canon_val`, `refuge_zeiss` (40 each, 120 in total), selected on their pooled validation (30), and scored on each domain's own 50 locked test images. Nothing is held out: the FiLM arm is trained with the true domain code and tested with it, the regime of the SpFiLM draft's 'both' setting. It asks whether conditioning helps when the camera is known, and — through the fixed-code sweep — whether the network uses the code at all.

## 2. Dice per domain, side by side

Same backbone, folds, budget, seeds, augmentation, optimiser and test images; the arms differ only in the conditioning. Δ is FiLM minus plain on per-image Dice with seeds averaged first; p-values are wilcoxon, Holm-adjusted over 6 tests, significance at α = 0.05.

| Test domain | Structure | Images | Plain Dice, mean ± seed SD | FiLM Dice, mean ± seed SD | Δ (FiLM − plain) | p | p (Holm) | Significant |
|---|---|---:|---:|---:|---:|---:|---:|:---:|
| `drishti_gs` | disc | 50 | 0.9594 ± 0.0041 | 0.9537 ± 0.0035 | -0.0057 | 0.003315 | 0.01657 | **yes** |
| `drishti_gs` | cup | 50 | 0.8606 ± 0.0123 | 0.8467 ± 0.0126 | -0.0139 | 0.0112 | 0.0448 | **yes** |
| `refuge_canon_val` | disc | 50 | 0.9432 ± 0.0094 | 0.9427 ± 0.0089 | -0.0005 | 0.5786 | 1 | no |
| `refuge_canon_val` | cup | 50 | 0.8617 ± 0.0145 | 0.8669 ± 0.0143 | +0.0052 | 0.5336 | 1 | no |
| `refuge_zeiss` | disc | 50 | 0.9350 ± 0.0065 | 0.9300 ± 0.0066 | -0.0050 | 0.0003108 | 0.001865 | **yes** |
| `refuge_zeiss` | cup | 50 | 0.8362 ± 0.0021 | 0.8310 ± 0.0072 | -0.0051 | 0.09513 | 0.2854 | no |

Plain arm: `stage4_all_domains_fixed_budget_plain_unet_3dom`. FiLM arm: `stage4_all_domains_fixed_budget_global_film_3dom`.

## 3. Does the network use the code? The wrong-code penalty

Each FiLM model was also scored on every test domain under every *other* domain's code. 'Penalty' is Dice under the domain's own code minus Dice under the worst other code. A penalty near zero means the codes are interchangeable and the FiLM layers are inert; a clear penalty means the code carries information the network acts on.

| Test domain | Structure | Own code Dice | Best other code | Worst other code | Penalty (own − worst) | Own code best in N/seeds | Dice under each code (mean over seeds) |
|---|---|---:|---:|---:|---:|:---:|---|
| `drishti_gs` | disc | 0.9537 | 0.9191 | 0.8532 | +0.1005 | 5/5 | `drishti_gs`: 0.9537, `refuge_canon_val`: 0.9191, `refuge_zeiss`: 0.8532 |
| `drishti_gs` | cup | 0.8467 | 0.7299 | 0.5904 | +0.2563 | 5/5 | `drishti_gs`: 0.8467, `refuge_canon_val`: 0.7299, `refuge_zeiss`: 0.5904 |
| `refuge_canon_val` | disc | 0.9427 | 0.9138 | 0.8936 | +0.0491 | 5/5 | `drishti_gs`: 0.9031, `refuge_canon_val`: 0.9427, `refuge_zeiss`: 0.9043 |
| `refuge_canon_val` | cup | 0.8669 | 0.8235 | 0.7540 | +0.1129 | 5/5 | `drishti_gs`: 0.7540, `refuge_canon_val`: 0.8669, `refuge_zeiss`: 0.8235 |
| `refuge_zeiss` | disc | 0.9300 | 0.9086 | 0.8584 | +0.0716 | 5/5 | `drishti_gs`: 0.8584, `refuge_canon_val`: 0.9086, `refuge_zeiss`: 0.9300 |
| `refuge_zeiss` | cup | 0.8310 | 0.7880 | 0.6228 | +0.2083 | 5/5 | `drishti_gs`: 0.6228, `refuge_canon_val`: 0.7880, `refuge_zeiss`: 0.8310 |

## 4. Findings

<!-- TODO: written by hand; the tool does not infer this. -->

