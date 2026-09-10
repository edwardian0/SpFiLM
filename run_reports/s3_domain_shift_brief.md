# Domain shift: geometric, not photometric

**One-page brief — Becket House, 10 Sep 2026.** Condenses `s3_1_3_intensity_shift_analysis.md` (single-source, 1→3) and `s3_3_1_intensity_shift_analysis.md` (leave-one-domain-out, 3→1). 35 runs, 5 seeds, fixed 40-image-per-domain budget. No model was re-run for the shift analysis.

---

## The result

**Brightness and colour predict nothing about transfer — and at the extremes they point backwards.**

| | Photometric distance | Dice |
|---|---|---|
| **Closest pair** (Drishti-GS ↔ REFUGE-Zeiss) | 2 grey levels apart | **worst** pair: 0.782 disc / 0.517 cup |
| **Furthest pair** (Canon ↔ Zeiss) | 46 grey levels apart | contains **best** cell: 0.891 / 0.760 |
| **LODO: closest to training pool** (RIM-ONE-DL) | 5 grey levels (green) | **collapses: 0.074 disc** |
| **LODO: furthest from pool** (REFUGE-Canon) | 36 grey levels | **best: 0.837 disc** |

Rank correlation is +1.00 in grey against disc Dice — a perfect reversal of the expected direction.

**Re-checked across every stored variant** — full sweep, 50- and 100-image subsamples, and both diagnosis splits × two pixel populations × four channels × three distance metrics, 336 configurations. In the single-source arm the reversal holds in **168 of 168**, disc and cup alike, with no exception. In the leave-one-out arm 121 of 168 run backwards, 18 are flat, 29 point the expected way.

**Those 29 are the tell.** They sit almost entirely in the red channel and in the border-inclusive pixel population. Red is near-saturated in fundus photography, so its histogram largely reports how much of the frame is bright retina rather than dark surround — a framing measurement, not a photometric one. Correlating each distance's agreement with disc scale against its agreement with Dice gives **ρ = −0.86**: the only photometric measurements that predict transfer correctly are the ones accidentally measuring geometry.

**It is not a measurement artefact.** The reversal survives every repair we tried: per-channel RGB instead of grey; support-overlap instead of distance (the direction where 96% of the target's blue pixels are outside anything seen in training returns the *second-best* cell); weighting each channel by how much disc/cup signal it actually carries; and conditioning on glaucoma status. No variant reorders anything.

**What is left of the photometric difference is precisely what FiLM does.** Split each pair's distance into the part removed by matching per-channel mean and variance: **92–97% is a per-channel gain-and-offset** for every pair with a real gap (median 94% across the three retained datasets; what survives is 0.3–3.6 grey levels). That is the part ordinary normalisation deletes — and the part with no measurable relationship to Dice.

**The affine decomposition breaks on RIM-ONE-DL specifically.** It removes 92–97% for full-fundus pairs but only 40–86% for any pair involving RIM-ONE, with residuals up to 10.2 grey levels against at most 3.6 elsewhere. Its histogram differs in *shape*, not just level, because the crop changes which tissue is in frame — the photometric anomaly is downstream of the geometric one.

**Geometry does track the results.** Disc area as a fraction of the frame is 1.6–2.9% for the three full-fundus sets and **41% for RIM-ONE-DL**. Rank each held-out dataset by scale mismatch to its *nearest* training set and the disc ordering follows (ρ = −0.80). RIM-ONE's mismatch is **4.05×**, and it is the only total failure in the study.

**The dominant effect is directional, and no symmetric descriptor can express it.** Which dataset you *train on* matters ~2× more than which you test on (spread 0.155 vs 0.069 disc). One pair, two directions: Canon→Zeiss = 0.891, Zeiss→Canon = 0.733. That **0.159 gap between two directions of one pair exceeds the entire spread between pairs** (0.087). Any pairwise distance assigns both directions the same number.

**Pooling domains is not reliably better than choosing one good source.** Of 24 paired tests on identical test images (120 images / 3 domains vs 40 images / 1 domain): **9 favour pooling, 10 favour the single source, 5 are inseparable.** Three times the data and three times the domain diversity is a coin flip against one well-chosen source.

**One correction to numbers we quoted before.** Drishti's cups were reported as 3.4–3.8× REFUGE's. That figure was pooled and confounded by case mix (Drishti is 69% glaucoma, REFUGE 10%). Within a diagnosis class it is **2.2×**. Real, but half of what we said.

---

## Conclusion

**Global intensity statistics do not describe this domain shift. Object scale does.** A conditioning descriptor built from global photometry would assign near-identical codes to the pair that most needs separating, and would modulate the one component demonstrably uncorrelated with performance.

---

## What it means for SpFiLM

1. **The brief's stated motivation no longer holds as written.** SpFiLM is motivated by vignetting — a smooth, position-dependent *brightness* effect. This data does not support brightness as the operative axis.

2. **But this cuts against Global FiLM harder than against SpFiLM.** Every measurement above is a *whole-image* histogram: it is structurally blind to *where* brightness varies. Global photometry is now empirically dead; *spatially varying* photometry is untested — and that is exactly SpFiLM's claim. The honest reading is that we have removed the baseline's justification and left ours unresolved. **We should say this first, before someone else says the first half of it.**

3. **Any descriptor we build needs two properties it does not currently have:** a scale/geometry term, and directionality (source-aware), because the largest single effect is invisible to a symmetric distance.

4. **Risk to have an answer ready for.** Result 5 implies source *selection* may buy more than conditioning does. If a reviewer asks whether SpFiLM beats "just pick the right training set", we currently cannot answer.

---

## Candidate next steps — for discussion

| | Experiment | Cost | What it decides |
|---|---|---|---|
| **A** | **Spatial shift measurement.** Region-wise / grid-wise intensity statistics instead of whole-image. Does a *spatial* descriptor separate Drishti from Zeiss where the global one cannot? | ~1 day, **no training** | Whether SpFiLM's premise is empirically real *before* we build it. Cheapest and most decision-relevant. |
| **B** | **Scale-matched control.** Re-crop/resize so disc-to-frame ratio matches across domains, re-run LODO. | ~1 day compute | Whether geometry is *causal* or merely correlated. If RIM-ONE recovers, we have the mechanism. |
| **C** | **Normalised-input baseline.** Per-channel standardisation, re-run. | cheap | If Dice is unchanged, that is a one-line proof that photometric conditioning has no headroom to exploit. |
| **D** | **The headline arms: Global FiLM vs SpFiLM.** | main compute | The brief's primary comparison. **Neither arm has been run yet** — everything so far is baselines. |

Suggested ordering: **A first** (it decides whether D's premise survives), then D. B and C are one-day insurance against the two most likely reviewer objections.

---

## Honest limits, in one place

- 3–4 datasets, 6 non-independent cells. The 336-configuration sweep varies the *measurement*, not the sample — it shows the result is not an artefact of one channel or metric, but adds no datasets. ρ describes an ordering; it is not a powered test. The claim is "photometry does not predict Dice, and at the extremes points the wrong way" — not "photometry ranks domains in reverse".
- Dice are 5-seed means at 40 training images; the Zeiss row's seed SD reaches 0.079, so individual cells are not precise to 4 digits. The row- and pair-level contrasts used above are larger than the seed spread.
- Whole-image histograms cannot see localised shift, colour covariance, local contrast or vignetting. **A null here is not a null for SpFiLM** — see implication 2.
- Per-diagnosis figures rest on 31 images per dataset per class and sit close to their own noise floor.
