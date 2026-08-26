1. **polarity** — inversion NOT required, and correctly NOT applied. Verified post-decode, not just as-read.
2. **manifest** — provenance is substantively honest. The split predates `b29bc10` (added in `2092c7b`), but the recorded commit is `2092c7b`, not `b29bc10`, and the generator reproduces the exact assignments bit-for-bit from a clean checkout. One weakness: `working_tree_dirty: true`.
3. **HD95** — `refuge`/`drishti` in 512×512 letterboxed-grid px; `rim_one_dl` in per-image native px. Correctly labelled *within* each dataset, but the labelling is asymmetric — REFUGE/Drishti carry no `hd95_unit` key at all. Comparability trap for any pooling script.
4. **gates** — 8 of 8 negative tests raised `DatasetLayoutError`. None fail open.
5. **entry-point references** — STALE. `submit_refuge_s2.sh` and `submit_drishti_s2.sh` both invoke `run_refuge_s2.py`, which is deleted in the working tree. `README.md` (×6) and `STAGE2.md` (×5) reference `run_stage2.py`, also deleted.
6. **most serious remaining defect** — the two completed baselines' submit scripts invoke a deleted entry point and will fail immediately on `sbatch`. This is a working-tree defect, not a defect in `b29bc10` itself.

---

## Scope and what could not be run

**CREATE was unreachable.** `ssh create` requires interactive MFA (the connection returns a TOTP QR challenge); `BatchMode=yes` cannot complete it in a non-interactive session. Therefore the following were **not run**, and nothing weaker is substituted for them:

- Phase 2.5 — existence of every manifest stem on the CREATE `/scratch` tree
- Phase 4.3 — CREATE-side copies of submit scripts and configs
- Phase 7.1/7.2 — Mac↔CREATE divergence for files touched by `b29bc10`

Where I verified dataset-facing behaviour, I used the **local Mac dataset copy** at `~/Desktop/Projects/Research/SpFilm/datasets`, which carries a complete RIM-ONE-DL tree (485 images, 970 mask PNGs). That is a genuine independent check of the *code and the data contract*; it is **not** a check of CREATE path resolution.

All destructive work was done in temporary git worktrees and a symlink fixture under the session scratchpad. The repository working tree is byte-for-byte unchanged — `git status` at the end of the audit is identical to `git status` at the start.

---

## Phase 1 — Polarity

### 1.1 Verbatim from the agent's report

On what polarity was found, and whether inversion is applied (opening summary line):

> Polarity: foreground-high (white); containment selected it decisively, so no inversion is required.

In the body:

> Therefore white/high is foreground and inversion is wrong.

> The loader performs no inversion. It outputs exactly two binary channels ordered
> Disc then Cup and asserts Cup ⊆ Disc after the pinned canonicalization.

And the containment table it based that on:

> | Candidate polarity | Cup-outside-Disc pixels |
> | --- | ---: |
> | as read, white/high foreground | 0 of 1,571,376 Cup pixels |
> | inverted, black/low foreground | 4,049,176 |

So the report *does* state both (a) and (b). The prompt's concern — that only as-read polarity was described — is not borne out.

### 1.2 The decode path

[data.py:853-878](src/spfilm/data.py#L853-L878) is the `rim_one_dl_foreground_high` branch. The operative lines are [data.py:867-868](src/spfilm/data.py#L867-L868):

```python
disc = disc_source >= 128
cup = cup_source >= 128
```

**No inversion.** High pixels are taken as foreground directly.

Inversion exists in this file only at [data.py:851-852](src/spfilm/data.py#L851-L852), inside the `separate_binary_foreground_low` branch:

```python
disc = disc_source < 128
cup = cup_source < 128
```

That branch is guarded by `record.mask_encoding in {"separate_binary_foreground_high", "separate_binary_foreground_low"}` and is reachable only from `load_rim_one_r3_manifest` ([data.py:718](src/spfilm/data.py#L718), the RIM-ONE-**r3** loader — a different dataset). Every RIM-ONE-DL record is constructed with `mask_encoding="rim_one_dl_foreground_high"` at [data.py:432](src/spfilm/data.py#L432).

Confirmation that the other two datasets are untouched — the full set of encoding assignments in the file:

```
154:            mask_encoding="refuge_0_cup_128_disc_255_background",
207:                    mask_encoding="drishti_softmap_three_of_four",
432:                mask_encoding="rim_one_dl_foreground_high",
718:                mask_encoding=f"separate_binary_{encoding}",
```

REFUGE decodes at [data.py:822-823](src/spfilm/data.py#L822-L823) (`disc = source <= 128; cup = source == 0`) and Drishti at [data.py:837-838](src/spfilm/data.py#L837-L838) (soft-map consensus threshold). Both are separate `elif` arms. **The RIM branch is applied to `rim_one_dl` only.**

### 1.3 Post-decode foreground fractions

Run through the real `discover_rim_one_dl` → `decode_mask_channels` path, 25 images sampled every 20th record so all six release×class cells are represented:

```
discovered records: 485
sampled: 25
POST-DECODE disc frac: mean=0.4048 min=0.2806 max=0.4965
POST-DECODE cup  frac: mean=0.1193 min=0.0299 max=0.2894
cup-outside-disc pixels post-decode: total=0 max_per_image=0
releases in sample: ['r1_glaucoma', 'r1_normal', 'r2_glaucoma', 'r2_normal', 'r3_glaucoma', 'r3_normal']
```

Against the stated expectation for this pre-cropped-ONH dataset — disc ≈ 0.25–0.6, cup ≈ 0.08–0.3:

| Structure | Expected | Post-decode measured | Verdict |
| --- | --- | --- | --- |
| disc | 0.25–0.6 | mean 0.4048, range 0.2806–0.4965 | in range |
| cup | 0.08–0.3 | mean 0.1193, range 0.0299–0.2894 | in range |

Disc is **0.4048, not ~0.9**. The failure signature described in the prompt (inversion detected but not applied) is absent.

The single low cup value (0.0299) is a genuinely small cup, not a polarity artefact — an inverted cup channel would read ≈ 0.97, not 0.03.

### 1.4 Cup ⊆ disc post-decode

Over the same 25 images: **0 pixels** where cup is foreground and disc is not. Not a pass/fail flag — a literal count of zero.

This is enforced, not incidental. [data.py:886-891](src/spfilm/data.py#L886-L891) raises unconditionally on any violation after the pinned repair. The five pinned source defects (`r2_Im319`, `r2_Im347`, `r2_Im357`, `r2_Im422`, `r2_Im427`, 2,505 px total) are repaired via `cup &= disc` at [data.py:876-877](src/spfilm/data.py#L876-L877) **only** when the decoded count exactly equals the count discovery recorded; any drift raises ([data.py:870-875](src/spfilm/data.py#L870-L875)). I confirmed that gate fires — see Phase 5, case 8.

### 1.5 Contact sheet

`artifacts/runs/rim_native_smoke_20260826/mask_contact_sheet.png`, read directly. Four samples (`r1_Im001`, `r1_Im028`, `r1_Im003`, `r1_Im004`), each as image / disc / cup / overlay.

What I actually saw: the disc panels are **solid white filled blobs on black background**, roughly circular, occupying a little under half the frame. The cup panels are **smaller solid white blobs**, concentric with and clearly inside the disc extent. The overlay panels show a green disc contour tracing the optic disc rim on the fundus and a blue cup contour inside it, both anatomically placed on the ONH.

There are **no frames with disc-shaped holes** — the inverted signature would show black discs on white frames, and green/blue contours hugging the image border. Nothing of that kind is present.

### Conclusion

**No inversion required — and none applied.** Correct in the code, correct in the report, and confirmed three independent ways (post-decode fractions, post-decode containment count, visual overlays).

---

## Phase 2 — Manifest integrity and provenance honesty

### 2.1 History

```
$ git log --oneline --follow -- splits/rim_one_dl.json
b29bc10 Revise RIM-ONE-DL Stage 2 evaluation contract
2092c7b Wire RIM-ONE-DL into Stage 2

$ git log --diff-filter=A --format="%H %ad %s" --date=iso -- splits/rim_one_dl.json
2092c7b79653b5a4c6680eedad5ba0fe98a23896 2026-08-24 20:18:06 +0100 Wire RIM-ONE-DL into Stage 2
```

**The manifest predates `b29bc10`** — confirmed. It was first committed in `2092c7b` on 2026-08-24.

`generate_rim_one_dl_split.py` **was** present at that point:

```
$ git ls-tree --name-only 2092c7b -- generate_rim_one_dl_split.py splits/
generate_rim_one_dl_split.py
splits/rim_one_dl.json
```

So the generator and the manifest entered the repository in the same commit. The prompt's premise — that the split may have been generated under an earlier, non-release-stratified policy — is testable, and I tested it directly in 2.2.

### 2.2 The provenance block, and whether it attests honestly

Verbatim, the entire block added by `b29bc10`:

```json
"provenance": {
  "generator_script": "generate_rim_one_dl_split.py",
  "git_commit": "2092c7b79653b5a4c6680eedad5ba0fe98a23896",
  "working_tree_dirty": true,
  "seed": 42,
  "generation_date_utc": "2026-08-26",
  "release_class_table": {
    "r1": { "glaucoma": 12, "normal": 86 },
    "r2": { "glaucoma": 108, "normal": 142 },
    "r3": { "glaucoma": 52, "normal": 85 }
  },
  "release_only_fallback_releases": [],
  "fellow_eye_caveat": "r3 filenames encode eye but not patient, so fellow-eye correlation is undetectable from filenames rather than known to be absent"
}
```

**The recorded commit is `2092c7b`, not `b29bc10`.** `2092c7b` is exactly the commit in which these split assignments first entered the repository. The provenance defect the prompt anticipated — a block attesting to `b29bc10` for a split `b29bc10` did not produce — **does not hold**.

The full diff of the manifest across the two commits is the schema bump plus that block and nothing else:

```
-  "schema_version": 1,
+  "schema_version": 2,
   "dataset": "rim_one_dl",
   "seed": 42,
   "policy": "one-time random 70/10/20 split across all 485 hospital-tree images; jointly stratified by release prefix and glaucoma/normal class",
   "source_record_count": 485,
+  "provenance": { ... }
```

And the assignments are byte-identical:

```
ASSIGNMENTS IDENTICAL 2092c7b vs b29bc10: True
  test: old=97 new=97 same_order=True
  train: old=340 new=340 same_order=True
  val: old=48 new=48 same_order=True
canonical hash old: 7870147cf3e5b9c2c13b861edb6dbced033fd10bdf6c82a1fb39a616b0543c8b
canonical hash new: 7870147cf3e5b9c2c13b861edb6dbced033fd10bdf6c82a1fb39a616b0543c8b
```

This independently reproduces the hash the report claims:

> The canonical JSON hash of the `partitions` object was
> `7870147cf3e5b9c2c13b861edb6dbced033fd10bdf6c82a1fb39a616b0543c8b`
> both before and after the provenance upgrade.

**"added provenance without changing any split assignments" is true as stated.**

**The stronger test — does the recorded generator actually produce the recorded split?** I checked out `b29bc10` into a detached worktree and re-ran the generator against the local dataset:

```
REGENERATED assignments == COMMITTED: True
committed hash  : 7870147cf3e5b9c2c13b861edb6dbced033fd10bdf6c82a1fb39a616b0543c8b
regenerated hash: 7870147cf3e5b9c2c13b861edb6dbced033fd10bdf6c82a1fb39a616b0543c8b
```

The generator named in the provenance, at the seed named in the provenance, deterministically reproduces the committed assignments exactly. The provenance describes how the split was generated **honestly and verifiably**.

Two residual weaknesses, both minor:

- `"working_tree_dirty": true` — the recorded commit hash alone does not pin the code that ran. In practice this is mitigated by the reproduction above (a clean `b29bc10` checkout yields the same result), but as an attestation it is weaker than a clean-tree hash.
- `"generation_date_utc": "2026-08-26"` sits alongside `git_commit: 2092c7b`, which was committed **2026-08-24**. The recorded commit is HEAD at the time of the provenance-writing regeneration, not the commit contemporaneous with the original assignment. Internally consistent once understood, but easy to misread.

### 2.3 Independent recomputation

Recomputed from the manifest against fresh discovery, not from any of the agent's outputs:

```
partition sizes: {'train': 340, 'val': 48, 'test': 97} sum: 485
union size: 485 == 485: True
no duplicates within/across (sum==union): True
  disjoint test/train: True (overlap=0)
  disjoint test/val: True (overlap=0)
  disjoint train/val: True (overlap=0)
manifest == discovery exactly: True | in manifest not disk: 0 | on disk not manifest: 0
```

Release × class contingency per partition:

| Partition | n | % | r1 G | r1 N | r2 G | r2 N | r3 G | r3 N |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 340 | 70.1 | 8 | 60 | 75 | 100 | 37 | 60 |
| val | 48 | 9.9 | 1 | 9 | 11 | 14 | 5 | 8 |
| test | 97 | 20.0 | 3 | 17 | 22 | 28 | 10 | 17 |

These match the report's tables cell for cell.

### 2.4 Release coverage and fallbacks

Every release appears in all three partitions with at least one image — and in fact every one of the six release×class **cells** is non-empty in all three partitions (smallest is `r1_glaucoma` in val at 1). `release_only_fallback_releases` is `[]`, and the regeneration independently emitted `release_only_fallback_releases=[]`. **No release fell back to release-only stratification.**

### 2.5 Stems on disk

Against the **local Mac copy**: all 485 manifest stems resolve to an existing image, disc mask, and cup mask (`all stems exist on disk: True`), and `manifest == discovery exactly` with zero in either direction.

**Not verified on CREATE** — see the scope note. This check must be repeated there.

### 2.6 Is it really proportional release stratification?

Arithmetic, per release×class cell, actual vs. the 70/10/20 proportional target:

| Cell | N | train (exp) | val (exp) | test (exp) |
| --- | ---: | --- | --- | --- |
| r1_glaucoma | 12 | 8 (8.4) | 1 (1.2) | 3 (2.4) |
| r1_normal | 86 | 60 (60.2) | 9 (8.6) | 17 (17.2) |
| r2_glaucoma | 108 | 75 (75.6) | 11 (10.8) | 22 (21.6) |
| r2_normal | 142 | 100 (99.4) | 14 (14.2) | 28 (28.4) |
| r3_glaucoma | 52 | 37 (36.4) | 5 (5.2) | 10 (10.4) |
| r3_normal | 85 | 60 (59.5) | 8 (8.5) | 17 (17.0) |

**Every cell is within ±1 of its proportional expectation.** That is the signature of exact largest-remainder allocation applied per joint stratum — it is not achievable by chance under an unstratified random split, where cells this small would scatter by several counts.

Ruled out explicitly — hospital partition reuse:

```
hospital groups: {'hospital_test_set': 174, 'hospital_training_set': 311}
  manifest test (n=97) vs hospital hospital_test_set: overlap=37
  manifest test (n=97) vs hospital hospital_training_set: overlap=60
```

The manifest test partition draws from **both** hospital folders (37 + 60). It is not the hospital split under another name.

**Conclusion: the split is consistent with proportional joint release×class stratification at 70/10/20, and is reproducible from the recorded generator and seed.**

---

## Phase 3 — HD95 and cross-dataset comparability

### 3.1 Units per dataset

| Dataset | HD95 unit | Set where |
| --- | --- | --- |
| `refuge` | 512×512 letterboxed-grid pixels | default `batch_hd95_unit` at [engine.py:308](src/spfilm/engine.py#L308) |
| `drishti` | 512×512 letterboxed-grid pixels | same default |
| `rim_one_dl` | per-image native source pixels | [engine.py:310-320](src/spfilm/engine.py#L310-L320) |

The conversion is at [engine.py:319](src/spfilm/engine.py#L319):

```python
hd95_multipliers = [1.0 / scale for scale in scales]
batch_hd95_unit = "native pixels"
```

triggered only when `"letterbox_scale" in metadata`, which only RIM records carry. Mixed frames within one evaluation are rejected outright ([engine.py:323-324](src/spfilm/engine.py#L323-L324)): `RuntimeError("Evaluation mixed incompatible HD95 coordinate frames")`. There is a hard post-condition too — [engine.py:624-627](src/spfilm/engine.py#L624-L627) raises `"RIM-ONE-DL evaluation did not convert HD95 to native pixels"` if a RIM run somehow finishes on the grid frame.

### 3.2 Is the difference flagged and labelled? — partly, and there is a real trap

Labelling that **is** present, verified in the actual artifacts:

- RIM `test_metrics.json` → `report["test"]["hd95_unit"] == "native pixels"`
- RIM `metric_frame` is dataset-specific and explicit ([engine.py:616-623](src/spfilm/engine.py#L616-L623)): *"each HD95 value is divided by that scale and reported in native-source pixels, not letterboxed-grid pixels or millimetres"*
- RIM per-image CSV carries `hd95_unit=native_px` **on every row**, plus `native_width`, `native_height`, `letterbox_scale`
- Summary column header is dataset-switched at [run_stage_s2.py:512](run_stage_s2.py#L512): `"HD95 (native px)" if config.dataset == "rim_one_dl" else "HD95 (px)"`
- `_print_test_results` ([engine.py:683-696](src/spfilm/engine.py#L683-L696)) prints a matching per-dataset banner

The report's claims here are accurate. But the labelling is **asymmetric**, and that is the finding:

```
RIM    test.hd95_unit: native pixels
REFUGE test.hd95_unit: None
```

[engine.py:344-345](src/spfilm/engine.py#L344-L345):

```python
if hd95_unit == "native pixels":
    metrics["hd95_unit"] = hd95_unit
```

The key is written **only** in the native case. REFUGE and Drishti emit no `hd95_unit` key at all, and their per-image CSVs have no unit column:

```
RIM:    image_id,release_prefix,hospital_split,diagnosis_class,native_width,native_height,letterbox_scale,hd95_unit,structure,dice,iou,hd95,...
REFUGE: image_id,structure,dice,iou,hd95,acc,tp,fp,fn,tn
```

Their unit survives only as **prose inside `metric_frame`**. So any script that pools the three datasets by reading `hd95_unit` sees `"native pixels"` for RIM and `None` for the other two, and the natural — wrong — reading of `None` is "no conversion needed, therefore comparable". **Reported as a finding, per the prompt's instruction, independent of the code being otherwise correct.**

### 3.3 Per-image scale, both structures

**Per image, not per dataset** — [engine.py:407](src/spfilm/engine.py#L407) computes it from each record's own native size:

```python
"letterbox_scale": f"{image_size / max(record.native_size):.12g}",
```

**Applied to both structures** — in [metrics.py:186-203](src/spfilm/metrics.py#L186-L203) the multiplier is indexed by image (`hd95_multipliers[index]`) *inside* the `for channel, name in enumerate(CHANNEL_NAMES)` loop, so disc and cup both receive their image's factor.

The unit test for this ([test_model_metrics.py:89](tests/test_model_metrics.py#L89)) uses a **single** image with a single multiplier `[0.5]`, and the smoke run had exactly **one** test image (one distinct `letterbox_scale`, `0.977099236641`). So the per-image *variation* — index > 0 receiving a different factor — was never exercised end-to-end. I exercised it directly with two images and different multipliers:

```
  imgA   disc  hd95=4.0000
  imgA   cup   hd95=4.0000
  imgB   disc  hd95=12.0000
  imgB   cup   hd95=12.0000
```

Identical geometry, multipliers `[1.0, 3.0]` → 4.0 and 12.0 on **both** channels. **The indexing is correct.** It is correct but under-tested, which is a test-coverage finding, not a correctness one.

Sanity check on direction: `r1_Im003` is natively 524×524, `512/524 = 0.977099236641` — matches the CSV. HD95 is divided by that scale, i.e. scaled *up* by 1.023, which is right: native pixels are finer than grid pixels here.

### 3.4 REFUGE and Drishti numerically unchanged — evidence

The `metrics.py` diff from `a630fa2` (the last commit before RIM existed) to `b29bc10` is strictly additive. The only change to a computed value:

```diff
                         "hd95": hd95(
                             prediction_masks[index, channel],
                             target_masks[index, channel],
-                        ),
+                        )
+                        * float(hd95_multipliers[index]),
```

with the default established immediately above:

```python
if hd95_multipliers is None:
    hd95_multipliers = [1.0] * dice.shape[0]
```

For REFUGE and Drishti no multipliers are passed, so every value is multiplied by `1.0` — exact in IEEE-754, not merely close. Dice, IoU, accuracy and the confusion counts are untouched by the diff.

Demonstrated rather than asserted — the same seeded input run through `OverlapAccumulator` in detached worktrees at both commits:

```
--- a630fa2 ---
c3d46f72ec8cdc478da8f1d84834ce0144903e438dfa079796fe4e9effb761d8
--- b29bc10 ---
c3d46f72ec8cdc478da8f1d84834ce0144903e438dfa079796fe4e9effb761d8
```

**Bit-identical SHA-256 over the full per-image row set, pre-RIM vs post-RIM.** REFUGE and Drishti metric outputs are numerically unchanged.

I also independently reproduced the report's discovery-invariance precondition: local REFUGE discovery returns **400** records and Drishti **101** — so those digests were taken over real, populated trees (see Phase 7.4).

### 3.5 Dice, IoU, vCDR

- **Dice, IoU, accuracy, tp/fp/fn/tn** — unaffected. Only the `"hd95"` key is multiplied; confirmed by the diff above and by the bit-identical hash in 3.4.
- **vCDR** — not affected, and not on this path at all. It is computed in [mask_audit.py:126](src/spfilm/mask_audit.py#L126) (`s["vcdr"] = (cv / dv) if dv else np.nan`) and [preprocess.py:268](src/spfilm/preprocess.py#L268). Neither file appears in `b29bc10`'s diffstat. It is a ratio of two areas measured on the same grid, so it is scale-invariant by construction — confirmed rather than assumed.

---

## Phase 4 — Entry-point rename fallout

### 4.1 Every `run_stage2` hit

```
$ grep -rn "run_stage2" . --exclude-dir=.git --exclude-dir=__pycache__ --exclude-dir=.spfilm
STAGE2.md:148:../learning/.spfilm2/bin/python run_stage2.py audit
STAGE2.md:163:../learning/.spfilm2/bin/python run_stage2.py inspect --dataset refuge
STAGE2.md:164:../learning/.spfilm2/bin/python run_stage2.py inspect --dataset drishti
STAGE2.md:181:../learning/.spfilm2/bin/python run_stage2.py train \
STAGE2.md:197:../learning/.spfilm2/bin/python run_stage2.py all \
README.md:19:../learning/.spfilm2/bin/python run_stage2.py audit
README.md:22:../learning/.spfilm2/bin/python run_stage2.py inspect --dataset refuge
README.md:25:../learning/.spfilm2/bin/python run_stage2.py inspect --dataset drishti
README.md:28:../learning/.spfilm2/bin/python run_stage2.py train \
README.md:32:../learning/.spfilm2/bin/python run_stage2.py all \
README.md:57:├── run_stage2.py                       # audit / inspect / train / all CLI
run_reports/s2_rim_one_dl_wiring_report.md:139:(prose acknowledging the rename — not a stale invocation)
```

**11 stale documentation references** across `README.md` and `STAGE2.md`. No `.py`, `.json` or `.sh` file references `run_stage2`. The only file referencing the new name is `submit_rimone_s2.sh`.

### 4.2 The submit scripts

| Script | Line 42 invokes | Exists on disk? | Verdict |
| --- | --- | --- | --- |
| `submit_rimone_s2.sh` | `run_stage_s2.py` | yes | **works** |
| `submit_refuge_s2.sh` | `run_refuge_s2.py` | **no** | **would fail on submission** |
| `submit_drishti_s2.sh` | `run_refuge_s2.py` | **no** | **would fail on submission** |

```
$ for f in run_stage2.py run_refuge_s2.py run_stage_s2.py; do ... done
  MISSING run_stage2.py
  MISSING run_refuge_s2.py
  EXISTS  run_stage_s2.py
```

An important distinction the prompt's framing invites getting wrong: **the deletions are unstaged.** At commit `b29bc10` all three entry points are still tracked:

```
$ git ls-tree --name-only b29bc10 | grep -E "^run_"
run_refuge_s2.py
run_stage2.py
run_stage_s2.py

$ git status --porcelain -- run_stage2.py run_refuge_s2.py
 D run_refuge_s2.py
 D run_stage2.py
```

So a **fresh clone of `b29bc10` would run all three submit scripts successfully**. The breakage lives in the working tree — which is what gets rsynced to CREATE. The hazard is real for deployment, but it is not a defect introduced by the commit under review.

Also note `submit_drishti_s2.sh` invoking `run_refuge_s2.py` is doubly wrong: even before the deletion, the Drishti wrapper was calling the REFUGE entry point. `a630fa2` ("Generalise Step 2 entry point to Drishti-GS") generalised the entry point but the wrapper was evidently never repointed.

Cosmetic conflicts, matching the agent's own account:

- `submit_rimone_s2.sh:2` — `#SBATCH --job-name=drishti_s2`
- `submit_rimone_s2.sh:37` — banner reads `starting refuge_s2`
- `submit_drishti_s2.sh:37` — banner reads `starting refuge_s2`

The one uncommitted change to `submit_drishti_s2.sh` is a user comment fix, unrelated:

```diff
-# Smoke: sbatch --time=0-00:20:00 /users/k23123868/edward/spfilm/submit_refuge_s2.sh --smoke
+# Smoke: sbatch --time=0-00:20:00 /users/k23123868/edward/spfilm/submit_drishti_s2.sh --smoke
```

### 4.3 CREATE-side copies

**Not run.** CREATE requires interactive MFA. This must be repeated on the cluster before any submission — especially since the deletions are working-tree-only and the deployed checkout is on a different base commit.

---

## Phase 5 — Do the gates fail closed?

### 5.1 What the 22 tests actually assert

The suite passes:

```
$ .spfilm/bin/python -m pytest tests/ -q
......................                                                   [100%]
22 passed in 1.23s
```

| Group | Count | What they assert |
| --- | ---: | --- |
| `test_model_metrics.py` model/metrics | 2 | U-Net preserves spatial shape, emits 2 channels; disc and cup Dice stay separate |
| HD95 scaling | 2 | multiplier converts grid→native on **both** channels; invalid scale (0.0) raises `ValueError` |
| hand-computed overlap | 4 | Dice/IoU, confusion counts and pixel accuracy, HD95 = the 4-px offset, HD95 symmetric and zero on identical masks — all against hand-checkable fixtures |
| degenerate cases | 3 | empty prediction / empty target / both empty behave per the documented policy |
| CSV↔summary | 3 | CSV rows and summary agree, exclusions counted, all-excluded reports `None` not `0` |
| `test_data.py` decode | 3 | REFUGE nesting, Drishti 3-of-4 consensus, cup-outside-disc raises |
| RIM decode | 1 | pinned source defect repaired, others not |
| RIM manifest | 2 | 485 stems mapped exactly once; missing provenance raises |
| split logic | 2 | REFUGE 256/64/80 disjoint; Drishti provider test locked |

These are **real assertions against hand-checkable expected values**, not tautologies. I found no test that merely checks a function returns, and none that asserts something trivially true.

**But there is a coverage gap, and it is exactly the one the prompt suspected.** The RIM manifest tests (`RimOneManifestTests`) build 485 **synthetic** `FundusRecord`s with fabricated ids (`sample_000`…) and non-existent paths (`Path("/sample_000.png")`). They never touch the filesystem. Consequently **none of the eight disk-facing negative conditions below is covered by the suite.** "22 tests passed" is true and the tests are sound, but it does not speak to the disk gates.

### 5.2 The eight negative tests

Run in a scratch symlink fixture (485 images / 970 mask PNGs mirroring the real tree), never against the repo or the real dataset. Baseline first: `discover_rim_one_dl(fixture)` → `OK records: 485`.

| # | Injected fault | Exception | Message |
| ---: | --- | --- | --- |
| 1 | manifest stem absent on disk | `DatasetLayoutError` | `Expected 485 RIM-ONE-DL hospital-tree images, found 484 with breakdown {('test_set','glaucoma'):56, ('test_set','normal'):117, ...}` |
| 2a | disk stem dropped from manifest | `DatasetLayoutError` | `RIM-ONE-DL manifest partition 'train' must contain 340 stems, found 339` |
| 2b | manifest stem swapped for a bogus id (counts preserved) | `DatasetLayoutError` | `RIM-ONE-DL manifest does not exactly match discovery: unlisted_discovered=['r1_Im001'], listed_missing_on_disk=['r9_ImFAKE']` |
| 3 | same stem in two partitions | `DatasetLayoutError` | `RIM-ONE-DL manifest partition 'val' must contain 48 stems, found 49` |
| 4 | one mask file removed | `DatasetLayoutError` | `Expected 970 RIM-ONE-DL mask PNGs, found 969` |
| 5 | mask dims ≠ image dims | `DatasetLayoutError` | `RIM-ONE-DL image/mask size mismatch for r1_Im003: image=(524,524), disc=(517,517), cup=(524,524)` |
| 6 | mask filename fails end-anchored regex | `DatasetLayoutError` | `Unparseable RIM-ONE-DL mask filenames: [...]` |
| 7 | mask resolving in the opposite class folder | `DatasetLayoutError` | `Expected 970 RIM-ONE-DL mask PNGs, found 971` |
| 8 | decoded cup not contained in disc | `DatasetLayoutError` | `RIM-ONE-DL source boundary defect changed for r1_Im004: discovery recorded 0, decode found 144 cup-outside-disc pixels` |

**8 of 8 raise. No case fails open.** Every one raises `DatasetLayoutError` — a hard exception, not a warning, and not a silent continue.

Two honest notes on strictness:

- Cases 1, 4 and 7 are caught by the **count** gate before the more specific gate they were aimed at. The count gate is strict enough to catch them, so they do fail closed — but the specific mechanism (e.g. "this stem's mask is in the wrong class folder") is not what fires. Case 2b was constructed precisely to defeat the count gate, and the join gate caught it with the exact right diagnosis.
- My first attempt at case 2 reported a false fail-open. That was my error, not the code's: I filtered `r1_Im003` out of `train`, but `r1_Im003` is in `test`, so nothing was removed and the manifest was unmodified. Re-run against a stem genuinely in `train` (2a) and against a count-preserving substitution (2b), both raise.

### 5.3 Is `partitioned_randomly` reachable?

It exists on disk — this is not hypothetical:

```
$ ls .../RIM-ONE_DL_images/
LICENSE.txt
partitioned_by_hospital
partitioned_randomly
```

The literal-string grep is clean (only a docstring at [data.py:228](src/spfilm/data.py#L228)), but as the prompt says, that is necessary and not sufficient. The enumeration primitives in `data.py` are:

```
86:        for path in root.rglob("*")          <- recursive
252:                for path in directory.iterdir()
274:        for path in (mask_root / diagnosis_class).iterdir()
```

Lines 252 and 274 are non-recursive `iterdir()` on fixed, fully-qualified paths rooted at `RIM-ONE_DL_images/partitioned_by_hospital/{split}/{class}` and `RIM-ONE-DL_masks/{class}`. They cannot ascend.

Line 86 is `_files()`, which **is** recursive. Its only RIM-reachable caller is [data.py:767](src/spfilm/data.py#L767) inside `inspect_rim_download` — and that call sits *after* an early return taken whenever `discover_rim_one_dl` succeeds. More decisively, **`inspect_rim_download` has no callers anywhere in the repository**; it is dead diagnostic code. (Its own text betrays that the rglob does reach both trees: it tests `len(images) == 970`, i.e. 485 × 2.)

Rather than rest on that reasoning, I instrumented `Path.iterdir`, `Path.rglob`, `Path.glob`, `builtins.open` and `PIL.Image.open`, then ran the full live pipeline — discovery, manifest load, and decoding:

```
total filesystem accesses recorded: 5067
accesses touching 'partitioned_randomly': 0
any rglob/glob calls: NONE
distinct directories enumerated via iterdir:
    .../RIM-ONE-DL_masks/glaucoma
    .../RIM-ONE-DL_masks/normal
    .../RIM-ONE_DL_images/partitioned_by_hospital/test_set/glaucoma
    .../RIM-ONE_DL_images/partitioned_by_hospital/test_set/normal
    .../RIM-ONE_DL_images/partitioned_by_hospital/training_set/glaucoma
    .../RIM-ONE_DL_images/partitioned_by_hospital/training_set/normal
```

**Zero glob calls of any kind, zero accesses to `partitioned_randomly`, exactly six directories enumerated.** Established empirically, not by grep.

---

## Phase 6 — Hospital-partition removal and residue

### 6.1 No evaluation path

```
$ grep -rn "secondary_hospital\|hospital_test_per_image\|hospital_partition\|hospital_eval" \
    --include="*.py" --include="*.json" --include="*.sh" . --exclude-dir=.git --exclude-dir=.spfilm
(no hits outside run_reports)
```

Confirmed removed.

### 6.2 No re-enablement surface

- **Config keys** — `configs/stage2_rimone_create.json` holds no hospital key. Full key set: `base_channels, batch_size, brightness_contrast, data_root, dataset, epochs, experiment_name, horizontal_flip_probability, image_size, learning_rate, min_epochs, num_workers, output_dir, patience, requested_device, rim_manifest, rotation_degrees, seed, test_fraction, threshold, val_fraction, weight_decay`.
- **Dataclass** — `Stage2Config` has no hospital field; the only RIM-specific addition is `rim_manifest: str | None = None`.
- **CLI flags** — `run_stage_s2.py` exposes only `--seed`, `--epochs`, `--num-workers` (plus config/out-dir/smoke). No hospital flag.
- **Output columns / report fields** — the only hospital-named CSV column is `hospital_split`, which is provenance (see 6.3). No hospital-named artifact is produced; `artifacts/runs/rim_native_smoke_20260826/` contains no such file.

One low-severity dead-key observation, not hospital-related: `test_fraction` and `val_fraction` remain in the RIM config but are **ignored** for `rim_one_dl`, because `build_splits` returns at [engine.py:159-165](src/spfilm/engine.py#L159-L165) before any fraction logic. Someone editing them would expect the split to change; it will not.

### 6.3 Hospital label is carried, not used

Carried as metadata:

- [data.py:436](src/spfilm/data.py#L436) — `hospital_split=hospital_split` on the record
- [data.py:433](src/spfilm/data.py#L433) — `split_hint=f"hospital_{hospital_split}"`
- [engine.py:358,403](src/spfilm/engine.py#L358) — `hospital_split` as a per-image CSV provenance column
- [data.py:1185](src/spfilm/data.py#L1185) — audit output
- `generate_rim_one_dl_split.py:186` and `preflight_rim_one.py:161` — distribution reporting only

Not used to partition: `build_splits` dispatches `rim_one_dl` to `load_rim_one_dl_split_manifest` and returns immediately ([engine.py:159-165](src/spfilm/engine.py#L159-L165)), so the `split_hint` path — which *does* lock the provider test set for Drishti — is never reached for RIM. Partitioning is by manifest stem only, which Phase 5 cases 2a/2b confirm is enforced exactly.

Not used to evaluate: the single evaluation pass is over `splits["test"]`; there is no second model pass.

### 6.4 Does the report explain why it was cut?

Yes — the "Why the hospital result was cut" section under Findings. Its core argument, verbatim:

> The hospital `test_set` has 174 images. Under the frozen random primary manifest,
> 118 are in primary training, 19 in validation, and 37 in primary test. A metric
> over all 174 would therefore include 137 training/validation images and mostly
> measure seen-data performance while being labelled as generalisation.

I verified those four numbers independently in Phase 2.6: hospital `test_set` = 174, with manifest overlaps of 118 / 19 / 37 (118 + 19 = 137). **The arithmetic in the justification is correct**, and the reasoning follows from it. The section goes on to explain why it is not an r1 holdout either, and defers the generalisation question to Step 3 leave-one-domain-out.

---

## Phase 7 — Deployment readiness

### 7.1 Mac↔CREATE divergence for files touched by `b29bc10`

**Not run** — CREATE unreachable (MFA). No substitute offered.

What I can state from the Mac side: `b29bc10` touched 10 files, and one of them is **dirty in the working tree relative to the commit** — `src/spfilm/engine.py`:

```diff
+from torch.optim.lr_scheduler import ReduceLROnPlateau
...
-    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
+    scheduler = ReduceLROnPlateau(
         optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
+        # Can adjust patience (by increasing) and increas the factor to increas the time taken
     )
```

An import-style refactor plus a comment. **No behavioural change** — same class, same arguments. It is a user edit, not agent residue.

The agent's own account of the deployed checkout, which I could not verify:

> The deployed checkout is based on `ec384d27ec3a1c5f1273bd516ab9f3e1d7eeee7d` with the RIM integration untracked, whereas the local checkout is based on `2092c7b` plus this task's changes.

If accurate, CREATE is **two commits behind** and carries the RIM integration as untracked files — so the schema-2 manifest, the HD95 native conversion and the hardened gates are all absent there. Must be confirmed on the cluster.

### 7.2 Which config `submit_rimone_s2.sh` reads, and the skipped preflight

The wrapper reads `configs/stage2_rimone_create.json`. Whether the CREATE copy matches the Mac copy: **not verified** (MFA).

The Mac copy is dirty and **wrong**, confirming the agent's disclosure:

```diff
-  "data_root": "~/datasets/glaucoma_datasets",
+  "data_root": "/scratch/prj/bc_ca_segmentation_in_tb_anatomy/datasets/RIM-ONE_DL_images",
```

`discover_rim_one_dl` documents and requires the **parent** of both sibling trees ([data.py:227-233](src/spfilm/data.py#L227-L233)):

```python
"""``root`` is the parent containing the two release trees. ..."""
image_root = root / "RIM-ONE_DL_images" / "partitioned_by_hospital"
mask_root = root / "RIM-ONE-DL_masks"
```

With the dirty value, it would look for `.../RIM-ONE_DL_images/RIM-ONE_DL_images/partitioned_by_hospital` and fail closed with `RIM-ONE-DL layout is incomplete; missing directories: [...]`. Correct root: `/scratch/prj/bc_ca_segmentation_in_tb_anatomy/datasets`.

The two baseline configs are also dirty, tracking the dataset directory move — note the asymmetry:

```diff
refuge:  -/scratch/.../bc_ca_segmentation_in_tb_anatomy/glaucoma_datasets/REFUGE
         +/scratch/.../bc_ca_segmentation_in_tb_anatomy/datasets/REFUGE
drishti: -/scratch/.../bc_ca_segmentation_in_tb_anatomy/glaucoma_datasets/DRISHTI-GS
         +/scratch/.../bc_ca_segmentation_in_tb_anatomy/datasets/glaucoma_datasets/DRISHTI-GS
```

REFUGE landed at `datasets/REFUGE`, Drishti at `datasets/glaucoma_datasets/DRISHTI-GS`. None of these three paths has been resolved on CREATE in this audit.

**What `--skip-runtime` skipped**, exactly ([preflight_rim_one.py:332-338](preflight_rim_one.py#L332-L338)):

- `check_torch_cuda()` — `torch.version.cuda`, `torch.cuda.is_available()`, and `torch.cuda.get_device_properties(0)`
- `check_amp_numerics()` — a `PlainUNet(base_channels=8)` forward/backward on `cuda` under `torch.amp.autocast("cuda")` with `GradScaler`, asserting a finite loss

**Remaining unverified:** CUDA availability on the target node, the cu121 torch build resolving against the driver, and AMP producing finite losses. The preflight prints this honestly rather than claiming a pass: `"All data gates passed; CUDA runtime gates remain to be run on CREATE."`

### 7.3 Dirty and untracked files left outside the commit

| Path | Status | Attribution |
| --- | --- | --- |
| `configs/stage2_refuge_create.json` | M | **User** — dataset move (`glaucoma_datasets/REFUGE` → `datasets/REFUGE`) |
| `configs/stage2_drishti_create.json` | M | **User** — dataset move |
| `configs/stage2_rimone_create.json` | M | **User** — dataset move, but root is one level too deep (defect, disclosed by the agent) |
| `run_stage2.py` | D | **User** — the known rename tidy |
| `run_refuge_s2.py` | D | **User** — same tidy; breaks two submit scripts |
| `src/spfilm/engine.py` | M | **User** — import refactor + comment, no behavioural change |
| `submit_drishti_s2.sh` | M | **User** — smoke-command comment fix |
| `lodo/lodo.py` | ?? | **User** — Step 3 leave-one-domain-out work, dated 2026-08-25, predates the agent's task |

**No agent residue in the tracked tree.** The agent's run outputs live under `artifacts/runs/` (`rim_native_smoke_20260826/`, `rim_one_wiring_smoke/`, `rim_one_wiring_smoke_final/`), which `.gitignore` excludes via `artifacts/`. That is the right place for them, though the two earlier `rim_one_wiring_smoke*` directories from 2026-08-24 are now superseded and could be cleared.

### 7.4 Was the REFUGE digest comparison vacuous?

**No — it was run against an intact tree.** I re-ran the precondition independently:

```
REFUGE records: 400
Drishti records: 101
```

Both discoveries return their full expected populations, so both sides of the digest comparison were computed over real, populated record lists. Had the tree been mid-transfer, discovery would have failed closed (REFUGE's count gate) rather than returning 400. **The match is not vacuous**, and the agent's stated precondition holds.

**But the prompt's distinction is the important one, and it stands.** What was tested:

- ✅ **Code invariance on Mac** — that `b29bc10` did not change REFUGE/Drishti discovery output. This is what the matching digests establish, and Phase 3.4 independently confirms it at the metrics level with a bit-identical hash.
- ❌ **CREATE path resolution after the dataset directory move** — **not tested at all.** The configs show REFUGE moving from `glaucoma_datasets/REFUGE` to `datasets/REFUGE` and Drishti to `datasets/glaucoma_datasets/DRISHTI-GS`. Whether those paths resolve on CREATE is unverified, and cannot be verified from here.

The agent stated this limit accurately rather than overclaiming:

> This is discovery invariance evidence. It is not a claim that the completed
> historical runs can be replayed from their stale resolved paths.

One related note it also flagged, which I confirmed: the completed REFUGE and Drishti runs' `resolved_config.json` files still record pre-move paths. Those historical results stand, but are not rerunnable verbatim.

---

## Where the original report is contradicted

Very little. The report is unusually accurate, and its major claims survived independent checking. The discrepancies are these:

**1. Mask tree file count — resolved in the report's favour.**

> The mask tree contains 344 glaucoma masks and 626 normal masks, for 970 total.

My first count found 1941 files. On inspection the tree holds `970 png + 971 txt`; all 970 PNGs match the end-anchored regex and zero fail it. **The report is right**; the extra files are `.txt` contour files it correctly ignores. Noted only because the raw file count is a trap for a future reader. (The 971st `.txt` having no PNG partner is a curiosity, not a defect — nothing reads them.)

**2. Foreground fraction ranges differ slightly between the report's two runs**, e.g. cup mean `0.116031` (60-pair sample) vs `0.103307` (full 485 in preflight). Both are internally consistent — different sample sizes — and my own 25-image post-decode figure (`0.1193`) is compatible with both. Not a contradiction, just noted so nobody reads it as one.

**3. The prompt's hypothesised provenance defect does not exist.** The prompt anticipated:

> Does the commit hash it records correspond to the commit that actually generated the split assignments, or to `b29bc10`? If the latter, that is a provenance defect

It records `2092c7b` — the commit that first contained the split — and the generator reproduces those assignments exactly. Reported here because an audit should record what it *ruled out*, not only what it found.

---

## Proposed fixes — ordered by severity

**Not applied.** Nothing in the repository was modified by this audit.

### 1 — HIGH: two submit scripts invoke a deleted entry point

`submit_refuge_s2.sh:42` and `submit_drishti_s2.sh:42` both call `run_refuge_s2.py`, which no longer exists in the working tree. Both would fail immediately on `sbatch`.

- `submit_refuge_s2.sh:42` — change `run_refuge_s2.py` → `run_stage_s2.py`
- `submit_drishti_s2.sh:42` — change `run_refuge_s2.py` → `run_stage_s2.py`

Then verify each resolves its own config (`stage2_refuge_create.json`, `stage2_drishti_create.json`) and confirm `run_stage_s2.py` dispatches `refuge` and `drishti` identically to the deleted script — the digests in Phase 3.4 suggest it does, but the CLI surface should be diffed against `run_refuge_s2.py` at `b29bc10` before submitting.

### 2 — HIGH: RIM config `data_root` is one level too deep

`configs/stage2_rimone_create.json` (dirty, uncommitted):

- change `"/scratch/prj/bc_ca_segmentation_in_tb_anatomy/datasets/RIM-ONE_DL_images"` → `"/scratch/prj/bc_ca_segmentation_in_tb_anatomy/datasets"`

Fails closed rather than silently, so it is a blocker and not a correctness risk — but it blocks the run.

### 3 — MEDIUM: HD95 unit labelling is asymmetric

`src/spfilm/engine.py:344-345` writes `hd95_unit` only in the native case, so REFUGE and Drishti emit no unit key and no CSV unit column.

- `engine.py:344-345` — write `metrics["hd95_unit"]` **unconditionally**, using `"letterboxed_grid_px"` for the grid frame and `"native_px"` for RIM
- `engine.py:357-365` — add `hd95_unit` to the per-image CSV context for **all** datasets, not just RIM

Every downstream consumer then sees an explicit unit and cannot silently pool incompatible columns. This changes only an emitted label, not a computed value, so the Phase 3.4 invariance still holds — but re-run that digest check to confirm.

### 4 — MEDIUM: stale entry-point references in documentation

11 references to the deleted `run_stage2.py`:

- `README.md` lines 19, 22, 25, 28, 32, 57
- `STAGE2.md` lines 148, 163, 164, 181, 197

Replace with `run_stage_s2.py`. Line 57 of `README.md` is a directory-tree comment and needs the filename updated in place.

### 5 — MEDIUM: no disk-facing negative tests for the RIM gates

All eight disk gates fail closed today (Phase 5.2), but nothing in the suite would catch a regression, because `RimOneManifestTests` uses synthetic records with non-existent paths.

- `tests/test_data.py` — add a `tmp_path` fixture building a minimal RIM tree (a handful of images per hospital/class plus matching masks is enough) and add the eight negative cases, asserting `DatasetLayoutError` each time

Case 2b (bogus stem substitution with counts preserved) is the most valuable: it is the only one that isolates the join gate from the count gate.

### 6 — LOW: per-image HD95 multiplier variation is untested

`tests/test_model_metrics.py:89` uses one image and one multiplier, so `hd95_multipliers[index]` for `index > 0` is never exercised.

- extend that test to two images with different multipliers (e.g. `[1.0, 3.0]`) and assert both disc and cup on both images

The behaviour is correct today — I verified it directly — this just pins it.

### 7 — LOW: cosmetic Slurm metadata conflicts

- `submit_rimone_s2.sh:2` — `--job-name=drishti_s2` → `rimone_s2`
- `submit_rimone_s2.sh:37` — banner `starting refuge_s2` → `starting rimone_s2`
- `submit_drishti_s2.sh:37` — banner `starting refuge_s2` → `starting drishti_s2`

Misleading in `squeue` and in log headers; no functional effect.

### 8 — LOW: dead config keys in the RIM config

`test_fraction` and `val_fraction` are present but ignored for `rim_one_dl` (the manifest wins). Either drop them from `configs/stage2_rimone_create.json`, or have `build_splits` raise at `engine.py:159` when they are set alongside `rim_manifest`, so an edit that will have no effect fails loudly instead of silently.

### 9 — LOW: provenance records a dirty tree

`splits/rim_one_dl.json` records `"working_tree_dirty": true` against `git_commit: 2092c7b`, so the hash alone does not pin the code that ran. Since a clean `b29bc10` checkout reproduces the assignments exactly, regenerate the provenance block from clean `b29bc10` — the assignments will not move (verified) and the attestation becomes a genuine pointer. Optionally add a `partitions_sha256` field carrying `7870147c…` so tampering is detectable without a regeneration run.

### 10 — LOW: dead diagnostic code carrying the only recursive glob

`inspect_rim_download` (`src/spfilm/data.py:738`) has no callers and contains the only `rglob` that could reach `partitioned_randomly` (via `_files` at `data.py:86`, called at `data.py:767`). It is unreachable today — proven empirically — but it is a live hazard if anyone wires it up.

Either delete it, or add an explicit guard rejecting any path under `partitioned_randomly` before the `_files` call.
