# Does brightness explain the LODO results? No — it points the wrong way

**The short version.** The four datasets do look different from each other in brightness and colour. But that difference does not explain which ones the model segments well. If anything it explains it backwards: the dataset that looks *least* like the others (REFUGE-Canon) scores **best**, and the dataset that looks *most* like the others (RIM-ONE-DL) **collapses**. What does track the results is physical size — how big the optic disc is inside the picture frame. This is a negative result about brightness, and a positive one about scale.

---

## Terms used here

| Term | Plain meaning |
|---|---|
| **W₁** (Wasserstein-1 distance) | One number for how far apart two brightness histograms are. 0 means identical. It is measured in brightness units, so multiply by 255 to read it as grey levels. W₁ = 0.10 ≈ 26 grey levels apart. |
| **Leave-one-out distance** | How far one dataset's brightness sits from the *average of the other three*. This matches the experiment, where a model trains on three datasets and is tested on the fourth. |
| **ρ** (Spearman rank correlation) | Do two rankings agree? **+1** = identical order, **−1** = exactly reversed, **0** = unrelated. Here we compare "ranked by brightness difference" against "ranked by Dice score". |
| **Dice** | Segmentation accuracy, 0 to 1. Higher is better. |
| **FOV** (field of view) | Only the pixels inside the round retinal photo, ignoring the black border around it. |
| **Disc / cup** | The two structures being segmented. The cup sits inside the disc. |
| **± values** | Spread across the 5 random seeds, not across images. |

**Where the numbers come from.** Dice scores are copied from `s3_fixed_report.md` (20 runs, 120 training / 30 validation / 50 test images per fold). Brightness distances are recomputed from `artifacts/domain_shift/densities.npz`, covering all 1,386 images. Sizes come from `domain_structure_scale.csv`. The glaucoma split uses `artifacts/domain_shift_dx_global/` (31 images per dataset per class). Nothing was re-run.

---

## 1. The results we are trying to explain

Each dataset takes a turn being held out: the model trains on the other three and is tested on it.

| Rank | Held out | Disc Dice | | Rank | Held out | Cup Dice |
|---:|---|---:|---|---:|---|---:|
| 1 | `refuge_canon_val` | 0.8370 ± 0.0487 | | 1 | `refuge_canon_val` | 0.6486 ± 0.0650 |
| 2 | `refuge_zeiss` | 0.8212 ± 0.0207 | | 2 | `drishti_gs` | 0.6461 ± 0.0432 |
| 3 | `drishti_gs` | 0.7853 ± 0.0386 | | 3 | `refuge_zeiss` | 0.5729 ± 0.0110 |
| 4 | `rim_one_dl` | **0.0737** ± 0.0415 | | 4 | `rim_one_dl` | **0.0174** ± 0.0140 |

**Two things to notice before going further.**

The top three are not really separated. Canon beats Zeiss on disc by 0.0158, and beats Drishti on cup by 0.0025 — both smaller than the seed-to-seed wobble, and their confidence intervals overlap. Treat the first three as one cluster.

Disc and cup do not give the same order. Drishti is 3rd on disc but 2nd on cup, and Zeiss drops from 2nd to 3rd. So there is no single "domain difficulty" ranking.

The one gap that is unambiguous is the fall to RIM-ONE-DL: **0.71 lower on disc**, with no overlap at all. That is the result actually needing an explanation.

---

## 2. Brightness does not explain it

For each dataset, how far is its brightness from the average of the other three? (Field of view only, each dataset counted equally, since every fold trains on 40 images from each.)

| Held out | grey | red | green | blue | Disc Dice |
|---|---:|---:|---:|---:|---:|
| `refuge_canon_val` | **0.1406** | 0.1076 | **0.1466** | **0.1917** | **0.8370** ← best |
| `refuge_zeiss` | 0.0995 | **0.1722** | 0.0730 | 0.0759 | 0.8212 |
| `drishti_gs` | 0.0912 | 0.0904 | 0.0851 | 0.1236 | 0.7853 |
| `rim_one_dl` | **0.0525** | 0.1548 | **0.0190** | **0.0272** | **0.0737** ← worst |

Read the grey column top to bottom: the numbers fall as the Dice scores fall. The dataset that looks most unusual scores best; the one that blends in scores worst. That is the opposite of the expected relationship.

As a rank correlation against disc Dice:

| Channel | ρ | Meaning |
|---|---:|---|
| grey | **+1.00** | perfectly reversed |
| green | +0.80 | mostly reversed |
| blue | +0.80 | mostly reversed |
| red | 0.00 | no relationship |

**The clearest single case.** In the green channel, RIM-ONE-DL sits **0.0190** from the other three — about 5 grey levels, which is inside our own measurement noise. It is, in brightness terms, almost the same as the other datasets. It scores 0.0737. REFUGE-Canon is nearly 8× further away and scores 0.8370.

**Finding.** Brightness difference does not predict which dataset the model handles well. At the extreme it predicts the opposite.

> **How hard to push this.** ρ = +1.00 is a perfect reversal of the four *average* scores, but three of those four overlap each other, so the correlation is really carried by RIM-ONE-DL sitting at the bottom while being the closest in brightness. The safe claim is "brightness does not predict Dice, and at the extreme points the wrong way" — not "brightness ranks the datasets in reverse".

---

## 3. The colour channels disagree with each other

Red is the odd one out. It is the only channel with no relationship to Dice (ρ = 0.00) and the only one where RIM-ONE-DL is not the closest dataset — there, REFUGE-Zeiss is the outlier (0.1722) and RIM-ONE-DL sits third (0.1548). Red in fundus photos is close to saturated and carries little anatomical detail, so it mostly reflects exposure.

Apart from all four channels agreeing that Canon is the most unusual dataset, they disagree about the rest of the order.

**Finding.** Quote brightness results per channel, not as one number. And note the model never sees greyscale — `model.py` takes 3-channel RGB input.

---

## 4. Glaucoma vs non-glaucoma does not change the answer

Distances recomputed separately within each diagnosis class (31 images per dataset per class):

| Channel | Class | `drishti_gs` | `refuge_canon_val` | `refuge_zeiss` | `rim_one_dl` |
|---|---|---:|---:|---:|---:|
| grey | glaucoma | 0.1038 | **0.1754** | 0.1274 | **0.0569** |
| grey | non-glaucoma | 0.0749 | **0.1398** | 0.1028 | **0.0409** |
| green | glaucoma | 0.0959 | **0.1751** | 0.1010 | **0.0238** |
| green | non-glaucoma | 0.0703 | **0.1459** | 0.0712 | **0.0097** |

Canon stays the most unusual and RIM-ONE-DL the least, in both classes. The reversal in §2 is a property of the datasets, not of how many glaucoma cases each contains.

Diagnosis barely moves brightness at all. Within a dataset, glaucoma vs non-glaucoma differs by **0.0195–0.0306** (about 5–8 grey levels), against a measurement noise floor of **0.0158–0.0260** at this sample size. Only Canon's clears the floor with any margin.

For perspective: the two most similar datasets, Drishti and Zeiss, are **0.0080** apart — about 2 grey levels. They are closer to each other than glaucoma is to non-glaucoma inside a single dataset.

**Finding.** Diagnosis is not a confound for the brightness conclusion. It *does* matter for cup size, where the case mix is very uneven — Drishti is 69% glaucoma, both REFUGE sets are 10% — but that is a size question, not a brightness one.

---

## 5. What does explain it: how big the disc is in the frame

RIM-ONE-DL ships tight crops around the optic nerve. The others are full retinal photos. So the same structure occupies a completely different fraction of the image:

| Dataset | Disc area as % of frame | Disc width as fraction of image |
|---|---:|---:|
| `refuge_canon_val` | 1.6% | 0.144 |
| `refuge_zeiss` | 1.7% | 0.143 |
| `drishti_gs` | 2.9% | 0.178 |
| `rim_one_dl` | **41%** | **0.721** |

For LODO the model only needs *one* training dataset at roughly the right scale. So the useful measure is the distance to the **nearest** available training dataset:

| Held out | Nearest training set | How different in scale | Disc Dice |
|---|---|---:|---:|
| `refuge_canon_val` | `refuge_zeiss` | 1.01× (near-identical) | 0.8370 |
| `refuge_zeiss` | `refuge_canon_val` | 1.01× | 0.8212 |
| `drishti_gs` | `refuge_canon_val` | 1.24× | 0.7853 |
| `rim_one_dl` | `drishti_gs` | **4.05×** | 0.0737 |

That ordering matches the Dice ordering (ρ = −0.80, negative meaning "more mismatch → worse score", which is the direction we want). The two REFUGE sets are near-twins, so whichever is held out, its twin is still in the training pool. Drishti has no twin. RIM-ONE-DL has nothing close at all.

**Two honest limits.** Using distance to the *nearest* training set works; using distance to their *average* does not (ρ = −0.40 only). And this explains disc well but cup poorly (ρ = −0.40), where the uneven glaucoma mix is a live confound.

**Finding.** Performance follows whether a similarly-scaled dataset was available to train on — not whether the colours matched.

---

## 6. What this means for SpFiLM

The project brief motivates spatial conditioning with vignetting: a smooth, position-dependent *brightness* effect. These measurements do not support that motivation on this data. The brightness axis is real but points the wrong way, and the axis that actually orders the results is object scale, which FiLM-style feature scaling does not address.

Concretely: a conditioning descriptor built from brightness histograms would give nearly the same code to Drishti and Zeiss (0.0080 apart, the closest pair here) — and that is exactly the pair that most needs distinguishing. Any domain descriptor should include a scale term.

---

## 7. Caveats

- Four datasets means four data points. ρ = +1.00 is a perfect ordering, not a statistically powered test, and three of the four overlap.
- These are whole-image histograms. They cannot see *where* in the image brightness differs — which is precisely the spatial structure SpFiLM targets. A null here does not rule out a localised shift.
- RIM-ONE-DL's cup score (0.0174, interval touching zero) is a failed prediction, not a measurement.
- The per-diagnosis numbers rest on 31 images per dataset per class and sit close to their own noise floor.
- Including the black border around the retina changes which dataset is most unusual (RIM-ONE-DL becomes the outlier, since it has no border) but still does not recover the ordering — Canon is second-most unusual and still best.

---

## 8. Figures

| Figure | What it shows |
|---|---|
| `domain_shift/intensity_overlay_fov_{gray,red,green,blue}.png` | **Main figure.** Brightness distributions per channel (§2–3). Use the distances in this document, not the ones printed in the legend — see below. |
| `domain_shift/intensity_overlay_all.png` | The framing difference: RIM-ONE-DL has no black border. |
| `domain_shift_dx_global/intensity_overlay_fov_*_{glaucoma,non_glaucoma}.png` | The same split by diagnosis (§4). |
| `domain_shift/structure_scale.png` | Disc and cup size relative to the frame (§5) — the axis that does explain the results. |

**One correction to the main figure.** Its legend weights every *pixel* equally, so the datasets with more images dominate. The experiment trains on 40 images from each dataset, so each should count equally. Recomputing that way is what this document does; it changes Zeiss most (0.1273 → 0.0995). The figure's choice of all four datasets is correct here, since all four were run.
