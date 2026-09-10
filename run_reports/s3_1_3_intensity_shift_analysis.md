# Does brightness explain the single-source results? No — the two most alike datasets transfer to each other worst

**The short version.** Train on one dataset, test on another, and the score does not follow how similar the two look. It follows the opposite. Drishti-GS and REFUGE-Zeiss are, in brightness terms, almost the same dataset — about **2 grey levels apart** — and that pair produces the **worst** cell in the whole matrix. REFUGE-Canon is the most unusual-looking dataset of the three and is the **best** dataset to train on. Looking at colour instead of grey does not rescue the idea — not even after measuring which channels actually carry the disc and cup, and weighting by it — and neither does splitting by glaucoma. What *does* partly track the results is object size — but once you correct for the fact that Drishti is mostly glaucoma cases and REFUGE mostly is not, that effect is about half the size the pooled numbers suggest. This is a negative result about brightness, a partial positive one about scale, and a caution about the size numbers we quoted before.

This is the single-source companion to `s3_3_1_intensity_shift_analysis.md`, which asks the same question of the leave-one-out arm. Both reach the same verdict by different routes.

---

## Terms used here

| Term | Plain meaning |
|---|---|
| **Single-source (1→3)** | Train on one dataset, test on each of the others separately. The companion report trains on three and tests on the fourth. |
| **Source / target** | The dataset trained on, and the unseen dataset tested on. Every result here is *directed*: source→target is not the same as target→source. |
| **W₁** (Wasserstein-1 distance) | One number for how far apart two brightness histograms are. 0 means identical. Multiply by 255 to read it as grey levels: W₁ = 0.10 ≈ 26 grey levels. |
| **Pairwise distance** | How far two datasets sit from each other. This is the one that matches a source→target experiment. (The companion report uses leave-one-out distance, which matches its design instead.) |
| **ρ** (Spearman rank correlation) | Do two rankings agree? **+1** = identical order, **−1** = exactly reversed, **0** = unrelated. We compare "ranked by brightness difference" against "ranked by Dice". Because we *expect* more difference to mean a worse score, a **positive** ρ here means the relationship is backwards. |
| **Dice** | Segmentation accuracy, 0 to 1. Higher is better. |
| **FOV** (field of view) | Only the pixels inside the round retinal photo, ignoring the black border. |
| **Disc / cup** | The two structures being segmented. The cup sits inside the disc. |
| **d′** (separability) | How cleanly two regions — say cup and rim — are told apart by brightness alone in one channel. It is the gap between their averages divided by the spread within them, so channels sitting at different levels stay comparable. Roughly: 1 = heavily overlapping, 4 = well separated. |
| **Per-channel rescale** | Shifting and stretching one colour channel to match another's average and spread — the kind of correction ordinary image normalisation does automatically. |
| **± values** | Spread across the 5 random seeds, not across images. |

**Where the numbers come from.** Dice scores are copied from `s3_1_3_findings_no_rim_one.md` (15 runs, 40 training / 10 validation / 50 test images per cell, RIM-ONE-DL excluded throughout). Brightness distances are recomputed from `artifacts/domain_shift/densities.npz`, covering all 1,386 images. Sizes come from `domain_structure_scale.csv` and, per diagnosis, from `artifacts/domain_shift_dx_global/`. The noise floor comes from `diagnosis_noise_floor.csv`, and the per-channel structure contrast in §4 from `channel_contrast.csv`. No model was re-run; §4 is a new measurement of the existing images.

---

## 1. The results we are trying to explain

Rows are the dataset trained on, columns the unseen dataset tested on. RIM-ONE-DL is excluded here — it is a crop-centred outlier, and the point of this arm is to test the brightness idea *without* that escape hatch.

Each cell is **disc / cup** Dice, averaged over 5 seeds.

| Trained on ↓ / tested on → | `drishti_gs` | `refuge_canon_val` | `refuge_zeiss` | Row mean |
|---|---:|---:|---:|---:|
| `drishti_gs` | — | 0.8582 / 0.6661 | 0.8372 / 0.5598 | 0.8477 / 0.6129 |
| `refuge_canon_val` | 0.8793 / 0.7033 | — | **0.8913 / 0.7601** ← best | **0.8853 / 0.7317** |
| `refuge_zeiss` | **0.7273 / 0.4745** ← worst | 0.7327 / 0.6696 | — | 0.7300 / 0.5720 |

**Two things to notice before going further.**

*Which dataset you train on matters about twice as much as which one you test on.* Spread across the rows is 0.155 disc and 0.160 cup; across the columns it is 0.069 and 0.079. REFUGE-Canon is the best source by a clear margin; REFUGE-Zeiss is the worst and also the most seed-sensitive (disc SD 0.079 and 0.072, individual seeds ranging 0.61–0.83).

*The same pair behaves very differently in the two directions.* Canon→Zeiss scores 0.8913 on disc; Zeiss→Canon scores 0.7327. That is a **0.159 gap between two directions of one pair** — larger than the entire spread between pairs. Any explanation built on a single number per pair is already in trouble, whatever that number measures.

---

## 2. Brightness does not explain it

How far apart is each pair, and how well does that pair transfer? (Field of view only; pair mean is the average of the two directions, which is all a symmetric distance can speak to.)

| Pair | Mean disc | Mean cup | grey | red | green | blue |
|---|---:|---:|---:|---:|---:|---:|
| Drishti ↔ Zeiss | **0.7823** ← worst | **0.5172** ← worst | **0.0080** ← closest | **0.0625** | **0.0127** | **0.0454** |
| Canon ↔ Zeiss | 0.8120 | 0.7149 | 0.1801 | 0.2099 | 0.1617 | 0.1911 |
| Drishti ↔ Canon | 0.8688 ← best | 0.6847 | 0.1738 | 0.1480 | 0.1737 | 0.2365 |

Read the grey column against the Dice columns: the closest pair is the worst, by **0.087 disc** and **0.198 cup**. As a rank correlation:

| Channel | ρ vs disc | ρ vs cup | Meaning |
|---|---:|---:|---|
| grey | +0.50 | **+1.00** | backwards |
| red | +0.50 | **+1.00** | backwards |
| green | **+1.00** | +0.50 | backwards |
| blue | **+1.00** | +0.50 | backwards |

Every channel is positive: every channel gets the relationship the wrong way round.

**The clearest single case.** Drishti-GS and REFUGE-Zeiss sit **0.0080** apart in grey — about **2 grey levels**, below what we can resolve. Their summaries are near-identical (mean 63.3 vs 61.7, spread 23.1 vs 22.3), and both are full retinal photos, so the framing excuse used for RIM-ONE-DL does not apply. That pair produces the worst cell on **both** structures. The pair furthest apart in grey, red and green — Canon and Zeiss, **46 grey levels** apart — contains the best cell.

**Being "covered" by the training set does not help either.** The natural repair is that what matters is not distance but whether the target's brightness falls *inside* the source's range. It fails harder:

| Source → target | grey | red | green | blue | Disc | Cup |
|---|---:|---:|---:|---:|---:|---:|
| Canon → Zeiss | 61.5% | 62.2% | 70.2% | **20.0%** | **0.8913** | **0.7601** |
| Canon → Drishti | 64.4% | 80.5% | 59.7% | **3.9%** | 0.8793 | 0.7033 |
| Drishti → Canon | 68.2% | 87.3% | 65.8% | **5.4%** | 0.8582 | 0.6661 |
| Drishti → Zeiss | 97.1% | 92.0% | 97.9% | 89.7% | 0.8372 | 0.5598 |
| Zeiss → Canon | 66.4% | 82.5% | 66.6% | 30.6% | 0.7327 | 0.6696 |
| Zeiss → Drishti | 95.3% | 96.6% | 90.1% | 65.8% | **0.7273** | **0.4745** |

(How much of the target's brightness falls inside the source's central 96% range.) Ordered by disc Dice, the table runs almost perfectly *against* coverage — the two best-covered directions are the two worst results. The extreme case is Canon→Drishti, where **96% of the target's blue pixels sit outside anything the model saw in training** and it still returns the second-best cell on both structures.

**Finding.** Brightness difference does not predict which transfers work. The closest pair is the worst, in every channel, and a near-disjoint colour channel costs nothing measurable.

> **How hard to push this.** Three datasets give three pairwise distances and six non-independent cells. ρ = +1.00 here is a perfect reversal of three numbers, not a powered test. The safe claim is "brightness does not predict Dice and at the extreme points the wrong way" — not "brightness ranks the pairs in reverse".

---

## 3. Grey is misleading for exactly the pair that matters

The model never sees greyscale — `model.py` takes 3-channel RGB. And for the pair carrying the argument, the grey number is the least representative one available.

| Dataset, FOV mean (0–255) | red | green | blue | red − blue |
|---|---:|---:|---:|---:|
| `drishti_gs` | 106.3 | 51.0 | 15.6 | **90.7** |
| `refuge_canon_val` | 144.1 | 95.3 | 76.0 | 68.1 |
| `refuge_zeiss` | 90.6 | 54.1 | 27.2 | 63.3 |

Drishti and Zeiss are **7.8× further apart in red** and **5.7× further apart in blue** than the grey figure implies (0.0625 and 0.0454 against 0.0080). The reason is arithmetic: grey weights green at 0.587, the two datasets genuinely do coincide in green, and Drishti's red excess cancels against its blue deficit in the weighted sum. **They are the same brightness in a different colour.** The two REFUGE sets are the reverse case: nearly the same colour balance (red−blue 68 vs 63) at very different exposure.

This does not rescue the brightness idea — Drishti and Zeiss stay the closest pair in *every* channel — but it does mean the grey figure understates a real difference.

**Finding.** Quote brightness per channel, not as one number, and never quote grey alone for the Drishti/Zeiss pair.

---

## 4. Do the channels differ in what they carry? Yes — and it still does not help

§3 shows the channels disagree about how far apart the datasets are. The obvious follow-up: perhaps a gap only matters in the channel the model actually needs, and the harmless blue gap in §2 is harmless because blue carries nothing.

That is testable, but not from the histograms — they never look at the structures being segmented. `channel_contrast.py` measures it directly, still with no model involved. For every image it takes three regions — the cup, the rim (disc minus cup) and the retina outside the disc — and reports how separable they are in each channel as **d′**: the difference between region averages divided by the spread within them, so channels sitting at very different levels stay comparable.

**How well each channel separates the structures** (average over all images of the three retained datasets; higher is more signal):

| Channel | Disc vs retina | Cup vs rim |
|---|---:|---:|
| red | **3.98** | **1.63** |
| grey | 2.79 | 1.37 |
| green | 2.07 | 1.18 |
| blue | 1.85 | **1.45** |

*Red carries the disc*, by a wide margin — the disc is the brightest thing in the red channel in all three datasets. *Blue is not a dead channel*: it is weakest for the disc but second-best for the cup, ahead of both green and grey. *Grey is worse than red at both jobs*, which is arithmetic — luma weights green at 0.587, and green is the weakest disc channel of the three.

Per dataset, for the disc:

| Dataset | grey | red | green | blue |
|---|---:|---:|---:|---:|
| `drishti_gs` | 3.12 | 4.45 | 2.22 | **1.59** |
| `refuge_canon_val` | 2.92 | 3.78 | 2.41 | **2.36** |
| `refuge_zeiss` | 2.33 | 3.71 | 1.57 | **1.60** |

Drishti's blue channel is the weakest cell in the table because it is compressed against the floor: retina averages **15**, rim **24**, cup **36**, so the whole disc-to-retina step is **15 grey levels** where in red it is **87**. Canon's blue is not compressed — its step is **73** levels — which is the same fact that made the two datasets nearly disjoint in blue in §2.

**Now cross the two.** If gaps mattered in proportion to signal, the pairs separated in red should suffer most:

| Channel | Signal it carries (disc d′) | Drishti↔Zeiss | Canon↔Zeiss | Drishti↔Canon |
|---|---:|---:|---:|---:|
| red | **3.98** (most) | 0.0625 | **0.2099** | 0.1480 |
| grey | 2.79 | 0.0080 | 0.1801 | 0.1738 |
| green | 2.07 | 0.0127 | 0.1617 | 0.1737 |
| blue | 1.85 (least, for disc) | 0.0454 | 0.1911 | 0.2365 |
| | **Mean disc Dice** | **0.7823** ← worst | 0.8120 | 0.8688 ← best |

Red is at once the channel carrying the most disc signal *and* the channel in which Canon and Zeiss are furthest apart — and that pair contains the best cell in the matrix. Drishti and Zeiss are the closest pair in red as well, and are the worst.

**The direct test.** Re-weight each pair's RGB distance by how much signal each channel carries, so a mismatch in red counts for more than one in blue:

| Weighting | Drishti↔Zeiss | Canon↔Zeiss | Drishti↔Canon | ρ vs disc | ρ vs cup |
|---|---:|---:|---:|---:|---:|
| Unweighted | 0.0402 | 0.1876 | 0.1860 | +0.50 | +1.00 |
| Weighted by disc d′ | 0.0455 | 0.1929 | 0.1754 | +0.50 | +1.00 |
| Weighted by cup d′ | 0.0429 | 0.1902 | 0.1852 | +0.50 | +1.00 |

The correlations do not move at all. Weighting towards the channels the structures actually live in leaves the relationship exactly as backwards as it was.

**Finding.** The channels genuinely differ in what they carry — red dominates the disc, blue matters more for the cup than expected, and grey is worse than red at both. But that does not rescue the brightness account: the biggest domain gap sits in the most informative channel and costs nothing, and weighting the distance by channel informativeness changes no ordering and no correlation.

> **How hard to push this.** d′ says how separable two regions are on a single-pixel intensity basis. A convolutional network is not restricted to that — it can use texture, vessels and shape, and can recombine channels. A channel with low d′ is not necessarily a channel the model ignores. The claim here is narrow: the *ordering* does not change under any signal weighting we can measure.

---

## 5. Almost all of the difference is a simple rescale

Split each pair's distance into the part removed by matching each channel's average and spread, and the part that survives:

| Pair | Distance (grey / red / green / blue) | What survives a rescale | Removed |
|---|---|---|---:|
| Drishti ↔ Canon | 0.174 / 0.148 / 0.174 / 0.237 | 0.008 / 0.011 / 0.008 / 0.009 | 93–96% |
| Drishti ↔ Zeiss | 0.008 / 0.063 / 0.013 / 0.045 | 0.004 / 0.007 / 0.005 / 0.002 | 51–96% |
| Canon ↔ Zeiss | 0.180 / 0.210 / 0.162 / 0.191 | 0.012 / 0.015 / 0.013 / 0.009 | 92–96% |

Once each channel is shifted and stretched to match, all three datasets have nearly the same histogram *shape*: what is left is **0.002–0.015**, under 4 grey levels. Drishti↔Zeiss shows the low 51% only because its raw distance is already near zero — in absolute terms its residuals are the smallest in the table in every channel.

**Finding.** What the histograms call a dataset difference is, to 92–96%, a brightness-and-contrast rescale — the part ordinary image normalisation removes, and the part with no measurable relationship to Dice.

---

## 6. A single number per pair cannot work anyway

Brightness distance is symmetric: it gives one value to both directions of a pair. The results are not symmetric.

| Pair | Disc gap between directions | Cup gap |
|---|---:|---:|
| Canon ↔ Zeiss | 0.1586 | 0.0905 |
| Drishti ↔ Zeiss | 0.1099 | 0.0853 |
| Drishti ↔ Canon | 0.0211 | 0.0372 |
| **Average** | **0.0965** | **0.0710** |

The average gap *between the two directions of a pair* (0.097 disc) is larger than the entire spread *between* pairs (0.087). Splitting the six cells into "which source" and "which target" effects: source spread 0.155 disc, target spread 0.069, and an additive model of the two leaves residuals of at most 0.054.

**Finding.** Most of the structure is directional. A symmetric distance can only ever speak to the pair-specific leftover, which is the smallest of the three terms — and this is true of every channel, since W₁ is symmetric in all of them.

---

## 7. Glaucoma vs non-glaucoma does not change the answer

Drishti is 69% glaucoma; both REFUGE sets are 10%. So a pooled histogram could in principle be reporting case mix rather than camera. It is not.

Distances between datasets, recomputed inside one diagnosis class at a time (31 images per dataset per class):

| Pair | Class | grey | red | green | blue |
|---|---|---:|---:|---:|---:|
| Drishti ↔ Zeiss | glaucoma | **0.0194** | 0.0695 | **0.0091** | 0.0404 |
| | non-glaucoma | **0.0227** | 0.0910 | **0.0093** | 0.0503 |
| Drishti ↔ Canon | glaucoma | 0.2094 | 0.2087 | 0.2032 | 0.2355 |
| | non-glaucoma | 0.1610 | 0.1263 | 0.1622 | 0.2405 |
| Canon ↔ Zeiss | glaucoma | 0.2271 | 0.2775 | 0.2062 | 0.1951 |
| | non-glaucoma | 0.1820 | 0.2172 | 0.1613 | 0.1901 |

Drishti and Zeiss stay indistinguishable in grey and green in both classes; the other two pairs stay 0.16–0.23 apart. Canon stays the most unusual dataset in both classes.

**How much does diagnosis move brightness at all?** Less than we previously reported. Measured against a noise floor — two disjoint halves of 31 images from the *same* dataset *and the same* diagnosis, which average **0.011–0.021** in grey and reach **0.037** on an unlucky split:

| Dataset | grey | red | green | blue |
|---|---:|---:|---:|---:|
| `drishti_gs` | 0.0222 | 0.0350 | 0.0192 | 0.0078 |
| `refuge_canon_val` | **0.0306** | **0.0474** | 0.0281 | 0.0127 |
| `refuge_zeiss` | 0.0199 | 0.0132 | 0.0231 | 0.0174 |

Only REFUGE-Canon clears the floor — red 0.0474, above every split-half maximum recorded, and grey 0.0306, above every floor average. **This corrects the earlier reading that diagnosis moves every dataset by 0.022–0.031: most of that range is the draw at 31 images.**

And where a shift does exist it has no common direction. Glaucoma images are *darker* in Drishti (−5.5 grey levels) and Zeiss (−4.7) but *brighter* in Canon (+6.8), and Canon reverses sign between red (+12.1) and blue (−3.0). The same displacement means "glaucoma" in one dataset and "not glaucoma" in another.

**Finding.** Diagnosis is not a confound for the brightness conclusion. It *does* matter for size — see §8.

---

## 8. What partly explains it: size — but half of that was case mix

The datasets differ far more in how big the structures are than in how bright they are (averages over all images):

| Dataset | Disc width / image | Disc area % | Cup area % | Cup / disc area |
|---|---:|---:|---:|---:|
| `drishti_gs` | 0.180 | 3.0% | **1.49%** | **0.486** |
| `refuge_canon_val` | 0.144 | 1.7% | 0.43% | 0.249 |
| `refuge_zeiss` | 0.144 | 1.7% | 0.45% | 0.262 |

This lines up with the target column for cup, where Drishti is the hardest dataset to be tested on (0.589, against 0.668 and 0.660): a REFUGE-trained model has seen small cups and is asked to find large ones.

**But the pooled cup numbers mix two things.** Cup size is the clinical marker of glaucoma, and Drishti is mostly glaucoma cases. Splitting by diagnosis (medians throughout, so the *pooled* rows differ slightly from the averages above; 31 images per dataset per class):

| Statistic | Class | Drishti | Canon | Zeiss | Drishti / REFUGE |
|---|---|---:|---:|---:|---:|
| Cup area % | glaucoma | 1.77 | 0.79 | 0.81 | **2.2×** |
| | non-glaucoma | 0.81 | 0.36 | 0.35 | **2.2×** |
| | *pooled* | 1.52 | 0.37 | 0.39 | *3.9–4.1×* |
| Cup / disc area | glaucoma | 0.581 | 0.421 | 0.461 | **1.3–1.4×** |
| | non-glaucoma | 0.299 | 0.219 | 0.221 | **1.4×** |
| | *pooled* | 0.525 | 0.231 | 0.236 | *2.2–2.3×* |

Rebuilding each pooled average from the two class averages at each dataset's real case mix reproduces it to within 1–3%, so the split is arithmetically sound. (Averages, not medians — medians do not combine across a mixture.)

- **Disc size is a pure dataset effect** (~1.25×, unchanged by diagnosis — glaucoma moves disc width by at most 10% within a dataset). This supports the boundary-error reading: three of the four disc cells not sourced from Zeiss sit at 27.5–28.0 grid-pixel HD95, and only Canon→Drishti is elevated (47.45), consistent with Drishti's larger discs inflating the same relative error.
- **Cup-to-disc ratio is mostly a *diagnosis* effect** — it roughly doubles with glaucoma in every dataset. The dataset difference left over is only **1.3–1.4×**, not the ~2× the pooled row shows.
- **Cup area is both.** Drishti's cups are **2.2×** REFUGE's within each class; the pooled 3.9–4.1× is that 2.2× multiplied by a further ~1.8× from case mix. **Our earlier "3.4–3.8×" quoted the pooled figure and credited the whole product to the dataset.**

**Finding.** Object size does track the cup column, but the effect is about half what the pooled numbers say. Quote the within-diagnosis figures.

> **How hard to push this.** Size does not explain the directional gaps at all. Canon and Zeiss are geometric twins (disc width ratio 1.02) yet are the most asymmetric pair in the matrix (0.159 disc). Size is symmetric too, so like brightness it cannot produce that. The Zeiss row is also the most seed-unstable in the report, which points at training instability at 40 images rather than at any property of the dataset.

---

## 9. What this means for SpFiLM

Both this report and the leave-one-out companion land in the same place from opposite directions.

- **A brightness descriptor would fail on the pair that most needs distinguishing.** Drishti and Zeiss would get nearly the same code — they are 2 grey levels apart — and they are the pair with the worst transfer. Moving to RGB widens the gap but does not reorder anything, and adds the opposite failure: the pair *most* separated in blue has the best mean disc score. Weighting the channels by how much disc or cup signal they carry (§4) does not change that either.
- **FiLM's modulation is itself a per-channel rescale**, and §5 shows a rescale is 92–96% of the measured difference — the removable part, with no relationship to Dice.
- **A symmetric descriptor cannot carry the main effect**, which is directional: which dataset you train on, at about twice the weight of which one you test on.
- **Diagnosis-conditioned brightness is no better** — at the noise floor in two of the three datasets, sign-reversing in the third.
- **The one axis not ruled out is colour balance rather than level.** The two REFUGE sets share a balance at different exposure and contain the best cell; Drishti and Zeiss share an exposure at different balance and contain the worst. It points the right way for the worst pair but misranks the other two, rests on three numbers, and is still symmetric. Worth noting, not worth building on yet.

None of this argues against SpFiLM. It argues that the conditioning descriptor cannot be motivated by global histograms, and should include a size term. It also flags something conditioning does not address at all: the largest single effect in this arm is that some datasets are simply better to train on.

---

## 10. Caveats

- Three datasets give three pairwise distances and six non-independent cells. No correlation coefficient is defensible here; ρ is reported as a description of ordering, with the caveat in §2.
- Dice are five-seed averages at 40 training images. Zeiss seed SDs reach 0.079, so individual cells are not precise to four digits. The pair- and row-level contrasts used above are larger than the seed spread; the per-cell residuals in §6 are not.
- These are whole-image histograms, one channel at a time. They cannot see *where* brightness differs, nor colour covariance, local contrast, vignetting, or how the disc region itself looks. A null here does not rule out a localised or disc-region shift — which is precisely what SpFiLM targets.
- The per-diagnosis numbers rest on 31 images per dataset per class and sit close to their own noise floor. The class-conditional distances for the two distant pairs are safely above it; the Drishti↔Zeiss ones are not, and are reported as indistinguishable from zero rather than as measurements. The floor itself depends on channel and on whether you quote the average or the worst split — name the statistic when comparing across reports.
- d′ in §4 is measured at 512 px with each region eroded by one pixel, so cup and rim do not share boundary pixels. Every image in the three retained datasets was measurable; spread between images is large (SD 0.49–0.69), so the per-dataset averages are solid but individual images vary widely.
- The rescale in §5 matches average and spread only — a decomposition of the measured distance, not a claim about any network. Coverage in §2 uses a central 96% band; other definitions change the percentages but not the ordering.
- The histograms cover all 1,386 images, not the exact images in the locked `single_source_manifest.json` partitions; manifest-aligned distances are still outstanding. All runs share a dirty Git revision, inherited from the parent report.

---

## 11. Figures

| Figure | What it shows |
|---|---|
| `domain_shift/intensity_overlay_fov_{gray,red,green,blue}.png` | **Main figure.** Brightness per channel (§2–4). The blue panel is the one to look at: Drishti and Canon barely overlap, and it costs nothing. |
| `domain_shift/intensity_histograms.png` | All four channels against both pixel populations on one sheet. |
| `domain_shift_dx_global/intensity_overlay_fov_*_{glaucoma,non_glaucoma}.png` | The same split by diagnosis (§7). |
| `domain_shift/structure_scale.png` | Disc and cup size relative to the frame (§8). |
| `domain_shift_dx_global/structure_scale_{glaucoma,non_glaucoma}.png` | The same, split by diagnosis — this is the figure that shows the case-mix correction. |

**Two corrections to the sample-50 overlay** (`domain_shift_sample50/intensity_overlay_fov.png`), if it is quoted anywhere in a three-dataset write-up:

1. Its legend's "W₁ to rest" includes RIM-ONE-DL in the comparison set. RIM-ONE-DL is REFUGE-Canon's nearest neighbour (0.068), so it hides how unusual Canon is: the printed 0.125 becomes **0.179** once the three retained datasets are used alone — about 40% higher.
2. The legend also weights every *pixel* equally, so the larger datasets dominate. This experiment draws 40 training images from each dataset, so each should count equally. That reweighting matters most for Zeiss (0.147 → 0.093).
