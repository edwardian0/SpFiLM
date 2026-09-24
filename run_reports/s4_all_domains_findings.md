# Step 4 train-on-all, three-domain report: Global FiLM against the plain U-Net

**Evidence boundary.** Every figure below is copied from `run_reports/s4_all_domains_report.md`, which `aggregate_stage4_all_domains.py` computed on CREATE (2026-09-16) from the per-image metric CSVs of **5 plain and 5 Global FiLM runs**, validated against the locked budgeted manifest `single_source_manifest.json`. The pooled `test_pooled` block is ignored because it is not a per-domain result. The findings in section 4 are written by hand and interpret only the values shown. Stage 3 numbers quoted for context come from `s3_1_3_findings_no_rim_one.md` and `s3_1_3_intensity_shift_analysis.md`.

## 1. Protocol

One model per seed is trained on the pooled budgeted training partitions of `drishti_gs`, `refuge_canon_val` and `refuge_zeiss` (40 each, 120 in total), selected by lowest pooled validation loss (30 images), and scored on each domain's own 50 locked test images. Nothing is held out. The Global FiLM arm is trained with the true domain code and tested with it (codes 0, 1, 2), the regime the SpFiLM draft calls "both". The plain arm is the same U-Net without the FiLM layers; the two configs differ only in `arm`.

The regime asks two things. First, does conditioning help when the camera is *known*? Second, through the fixed-code sweep, does the network use the code at all? Every FiLM model was also scored on each test domain under each *other* domain's code.

| Setting | Value |
|---|---|
| Arms | `stage4_all_domains_fixed_budget_plain_unet_3dom`, `stage4_all_domains_fixed_budget_global_film_3dom` |
| Active domains | `drishti_gs`, `refuge_canon_val`, `refuge_zeiss` (`rim_one_dl` inactive) |
| Seeds | 42, 43, 44, 45, 46 |
| Train / val / test | 120 pooled / 30 pooled / 50 per domain |
| FiLM | one-hot 64 → MLP(256, 256) → (γ, β) per channel, clamp ±5, after each of the 5 encoder blocks |
| Test-time code | oracle (the domain's own code) |
| Paired test | Wilcoxon on per-image Dice with seeds averaged first, Holm over 6 tests, α = 0.05 |
| Git revision | `2167d88b947b2f9a96a2e59d7a810392202c3f18` (clean) for all 10 runs |
| CREATE jobs | FiLM 37227801–37227805; plain 37256883, 37280328, 37280330, 37280331, 37256887 |

## 2. Dice per domain, side by side

Δ is FiLM minus plain, so a positive value means conditioning helped.

| Test domain | Structure | Images | Plain Dice, mean ± seed SD | FiLM Dice, mean ± seed SD | Δ (FiLM − plain) | p | p (Holm) | Significant |
|---|---|---:|---:|---:|---:|---:|---:|:---:|
| `drishti_gs` | disc | 50 | 0.9594 ± 0.0041 | 0.9537 ± 0.0035 | -0.0057 | 0.003315 | 0.01657 | **yes** |
| `drishti_gs` | cup | 50 | 0.8606 ± 0.0123 | 0.8467 ± 0.0126 | -0.0139 | 0.0112 | 0.0448 | **yes** |
| `refuge_canon_val` | disc | 50 | 0.9432 ± 0.0094 | 0.9427 ± 0.0089 | -0.0005 | 0.5786 | 1 | no |
| `refuge_canon_val` | cup | 50 | 0.8617 ± 0.0145 | 0.8669 ± 0.0143 | +0.0052 | 0.5336 | 1 | no |
| `refuge_zeiss` | disc | 50 | 0.9350 ± 0.0065 | 0.9300 ± 0.0066 | -0.0050 | 0.0003108 | 0.001865 | **yes** |
| `refuge_zeiss` | cup | 50 | 0.8362 ± 0.0021 | 0.8310 ± 0.0072 | -0.0051 | 0.09513 | 0.2854 | no |

**Brief finding.** Conditioning never helps. Δ is negative in five of six cells and significantly so in three (Drishti disc and cup, Zeiss disc); the one positive cell (Canon cup, +0.005) is not significant. The effects are small — the largest is −0.014 Dice — but they are consistent across images, which is why the paired test finds them despite seed SDs of the same size.

## 3. Does the network use the code? The wrong-code penalty

"Penalty" is Dice under the domain's own code minus Dice under the worst other code. Near zero would mean the codes are interchangeable and the FiLM layers inert.

| Test domain | Structure | Own code Dice | Best other code | Worst other code | Penalty (own − worst) | Own code best in N/seeds | Dice under each code (mean over seeds) |
|---|---|---:|---:|---:|---:|:---:|---|
| `drishti_gs` | disc | 0.9537 | 0.9191 | 0.8532 | +0.1005 | 5/5 | `drishti_gs`: 0.9537, `refuge_canon_val`: 0.9191, `refuge_zeiss`: 0.8532 |
| `drishti_gs` | cup | 0.8467 | 0.7299 | 0.5904 | +0.2563 | 5/5 | `drishti_gs`: 0.8467, `refuge_canon_val`: 0.7299, `refuge_zeiss`: 0.5904 |
| `refuge_canon_val` | disc | 0.9427 | 0.9138 | 0.8936 | +0.0491 | 5/5 | `drishti_gs`: 0.9031, `refuge_canon_val`: 0.9427, `refuge_zeiss`: 0.9043 |
| `refuge_canon_val` | cup | 0.8669 | 0.8235 | 0.7540 | +0.1129 | 5/5 | `drishti_gs`: 0.7540, `refuge_canon_val`: 0.8669, `refuge_zeiss`: 0.8235 |
| `refuge_zeiss` | disc | 0.9300 | 0.9086 | 0.8584 | +0.0716 | 5/5 | `drishti_gs`: 0.8584, `refuge_canon_val`: 0.9086, `refuge_zeiss`: 0.9300 |
| `refuge_zeiss` | cup | 0.8310 | 0.7880 | 0.6228 | +0.2083 | 5/5 | `drishti_gs`: 0.6228, `refuge_canon_val`: 0.7880, `refuge_zeiss`: 0.8310 |

**Brief finding.** The code is used, and heavily. The own code is the best of the three in every cell for every seed (30/30), and the wrong code costs **0.05–0.10 disc** and **0.11–0.26 cup** Dice. The FiLM layers are not inert; the network has learned three distinct per-domain behaviours.

## 4. Findings

Plain words first: a *code* is a label on each image saying which camera it came from (Drishti = 0, Canon = 1, Zeiss = 2). FiLM reads the label and adjusts the network's features per channel. Here nothing is held out, so every test image gets its true label. (The histogram work from Stage 3 is not the code; it only comes in later, in LODO, to pick a label for a camera the network has never seen(TBD))

### 4.1 Telling the network the camera does not help

**What we saw.** FiLM with the correct label is never better than the plain U-Net. It is slightly worse in 5 of 6 cells and significantly worse in 3 (Drishti disc −0.006, Drishti cup −0.014, Zeiss disc −0.005). The gaps are small but consistent across images.

**Why.** The three cameras are similar (Stage 3 removed the one outlier, RIM-ONE). The plain network learns all three from 120 shared images. The FiLM network splits its behaviour by label, so each camera is effectively learned from its own 40. With cameras this alike, sharing everything beats specialising, by a little.

### 4.2 The network does use the label

**What we saw.** With the wrong label, Dice drops by 0.05–0.10 for disc and 0.11–0.26 for cup. The correct label scored best in every seed for every domain (30/30).

**Why it matters.** This proves the FiLM layer works: the label changes the output, and the network learned three genuinely different settings. It also proves 4.1 is not "FiLM does nothing" — it does something, it just isn't needed when the camera is known.

### 4.3 Canon is the middle camera

**What we saw.** If a Drishti image must carry a wrong label, Canon's costs little (0.919 disc) and Zeiss's costs a lot (0.853). The same for a Zeiss image: Canon's label 0.909, Drishti's 0.858. Canon images barely mind which label they get.

**Why.** As the network sees them, Drishti and Zeiss are far apart and Canon sits between them. Stage 3 found the same shape from a different direction: Canon was the best training source for both other cameras, and Drishti ↔ Zeiss transferred worst.

### 4.4 This is a warning for the LODO rule

  

**The rule.** In LODO the held-out domain has no label of its own, so it borrows the label of the training camera that *looks* most like it (mean and spread of R, G, B in the field of view).

  

**The problem.** Looking alike and behaving alike are different things here. Stage 3 measured Drishti and Zeiss as almost identical in brightness (W₁ 0.008) yet the worst pair for transfer; Canon looks different from both yet transfers best. 4.3 shows the network agrees. So the rule is expected to give Drishti the Zeiss label and Zeiss the Drishti label — the worst choice in each case — and toss a coin for Canon.

  

**What is at stake.** By 4.2, a wrong label costs up to a quarter of the cup Dice. FiLM could lose to plain in LODO because of the label, not because conditioning is useless. That would blur every later comparison that uses the same rule.

  

**Why we run it anyway.** It is the supervisor's rule, it uses no test labels, and this is a prediction from a three-label model; LODO models learn only two labels, so the ordering can change. Each LODO run also scores the held-out camera under every available label, so the report will say whether the rule picked the best one and what it cost if not. That number decides whether the rule needs changing.

  

### 4.5 Queued analysis (after LODO)

  

The per-image sweep files are already written. Two short checks would explain the label rather than just measure it: which images lose most under a wrong label (small or faint cups?), and what the wrong label does to the prediction (does the Zeiss label shrink cups on Drishti images?). Half a day, no new runs; fits Step 8 of the brief.

  

### 4.6 Next steps

  

1. Pull the hardened submit scripts on CREATE, then launch the 30 LODO jobs (`submit_stage4_plain.sh`, `submit_stage4_global_film.sh` × 3 held-out domains × 5 seeds) from an `erc-hpc-login` node.

2. Aggregate with `aggregate_stage4_film.py`. Read the label table (chosen vs best available) before the Dice table.

3. Report LODO FiLM under the honest rule as the main result, with the best available label as the ceiling it missed, if it did.