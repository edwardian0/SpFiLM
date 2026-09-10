# Where the project stands, and what to do next

**One page, 10 Sep 2026.** Read against `Edward_Project_Brief.pdf`. Everything below is finished and checked; the detail sits in `s3_fixed_report.md`, `s3_domain_shift_brief.md` and the two intensity analyses.

---

## 1. Progress against the brief's ten steps

| Step | Status |
|---|---|
| 1. Get oriented | done |
| 2. In-domain baseline | done — all three datasets |
| 3. Measure the shift | **done, and gone well past what was asked** |
| 4. Global FiLM | **not started** |
| 5. SpFiLM | **not started** |
| 6. Full comparison, 5 seeds × all folds | not started |
| 7. RIGA | not started |
| 8–10. Interpretation, extension, write-up | not started |

Steps 4 and 5 are the whole point of the project and neither has begun. Everything so far is the baseline the brief asks for before touching the method.

---

## 2. Final results

**a. The shift is large, and one dataset breaks completely.** Train on three datasets, test on the fourth:

| Held-out | In-domain disc | Cross-domain disc | Drop | Cross-domain cup |
|---|---:|---:|---:|---:|
| Drishti-GS | 0.952 | 0.785 | −0.17 | 0.646 |
| REFUGE-Canon | 0.955 | 0.837 | −0.12 | 0.649 |
| REFUGE-Zeiss | 0.955 | 0.821 | −0.13 | 0.573 |
| RIM-ONE-DL | 0.941 | **0.074** | **−0.87** | **0.017** |

(The in-domain runs used all available data and one seed; the cross-domain runs use 120 images and five seeds, so the drop mixes domain shift with a smaller training budget. The RIM-ONE collapse is far too large to be budget.)

**b. Brightness and colour explain none of it — they point backwards.** The two datasets that look most alike (Drishti and Zeiss, 2 grey levels apart) transfer to each other *worst*. The dataset that looks least like the others (REFUGE-Canon) is the *best*. RIM-ONE-DL is the closest to the training pool in colour and is the one that collapses. We re-checked this across 336 combinations of channel, metric, sample and diagnosis split; nothing reorders it.

**c. What is left of the colour difference is the thing FiLM already does.** 92–97% of every dataset's colour gap is a simple per-channel brightness-and-contrast rescale — the part ordinary image normalisation deletes anyway, and the part with no measurable link to accuracy.

**d. Physical size does explain it.** The optic disc fills 1.6–2.9% of the frame in the three full-photo datasets and **41%** in RIM-ONE-DL, which ships tight crops. Rank the datasets by how badly their scale mismatches the nearest training set and the accuracy ordering follows. RIM-ONE's mismatch is 4× and it is the only total failure.

**e. Pooling datasets is not reliably better than picking one good one.** 24 paired tests on identical test images: 9 favour training on three datasets, 10 favour training on one well-chosen dataset, 5 tie.

---

## 3. Conclusion

**The thing that breaks these models is how big the anatomy is in the frame, not what colour the camera makes it.** Global colour statistics are dead as an explanation and dead as a conditioning signal.

---

## 4. What this means for the project brief

The brief (§3) lists four conditions that must all hold for SpFiLM to beat Global FiLM. Our work answers the first one only partly, and it is the important one:

| Brief's condition | Where we stand |
|---|---|
| 1. The shift varies **across the image**, not a single global change | **Half-answered, and it needs care.** We measured the *whole-image* shift and found it is essentially a single global change per colour channel — and that it predicts nothing. We have **not** measured whether there is a *position-dependent* part left over. |
| 2. The pattern sits in the same place across images | untested |
| 3. Position is what disambiguates the correction | untested |
| 4. The correction is smooth and low-frequency | untested |

Three consequences worth deciding on:

1. **The brief's stated motivation — vignetting — is a spatial claim, and we have only killed the global one.** That is genuinely worse news for **Global FiLM** than for SpFiLM: we have removed the baseline's justification and left ours open. Say this in that order, before a reviewer says only the first half.

2. **RIM-ONE-DL will dominate the headline comparison, for a reason neither FiLM variant can fix.** It fails because of crop scale. Adding conditioning to a model that has never seen a disc at 41% of frame will not rescue it. Decide its role *before* Step 6, or the main table becomes a referendum on one broken fold.

3. **We have no answer to "why not just pick a good training set?"** Result (e) says pooling is a coin flip against one good source. If conditioning is to be worth its complexity, it has to beat that, and nobody has measured it.

---

## 5. Next steps

| | Do this | Cost | Why |
|---|---|---|---|
| **1** | **Measure the shift per image region instead of per whole image.** Same code, grid-wise instead of global. Does a position-aware descriptor separate Drishti from Zeiss where the global one cannot? | ~1 day, **no training** | Directly tests the brief's condition 1. If there is no spatial structure, SpFiLM's premise fails on this data and we should know that before spending the compute, not after. |
| **2** | **Decide RIM-ONE-DL's role.** Include as-is, re-crop the others to match its scale, or move it to a side analysis. | half a day of judgement | Stops one broken fold deciding the headline result. |
| **3** | **Step 4 + Step 5: build Global FiLM and SpFiLM.** | main compute | The actual project. Nothing else can substitute. |
| **4** | **Step 6: full run, 5 seeds, all folds, paired tests.** | main compute | The deliverable in the brief's §9. |

Optional insurance, one day each, only if time allows: a per-channel-normalised baseline (if accuracy is unchanged, that is a one-line proof that colour conditioning has no headroom), and a scale-matched control (shows whether geometry is *causal* rather than just correlated).

**Recommendation: do 1 and 2 this week, then go straight to 3.** Step 1 is cheap and can change what we build; step 2 is free and stops a foreseeable mess. Everything else waits for the FiLM arms, which are now the critical path and have not started.

---

## 6. Worth remembering

The brief says plainly that a clean negative result is a real outcome. Results (b), (c) and (e) are already clean negatives with 35 runs and five seeds behind them. If the FiLM arms tie, we have a coherent story rather than an empty one: on this task the shift is geometric, global photometric conditioning has nothing to grip, and we can show it four different ways.
