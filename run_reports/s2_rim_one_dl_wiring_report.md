Polarity: foreground-high (white); containment selected it decisively, so no inversion is required.
485-image join: all 485 images joined to one Cup and one Disc mask with class-folder agreement.
Final split counts: 340 train / 48 validation / 97 test; every release appears in every partition.
Existing datasets: REFUGE and Drishti discovery JSON is unchanged by SHA-256; local REFUGE was verifiable at 400 images.
Remaining blockers: the final changes are not deployed to CREATE, the local dirty RIM config has the wrong root level, and CUDA preflight is prohibited/pending.

# RIM-ONE-DL Step 2 wiring report

## Phase 0 — live CREATE data and polarity

Phase 0 was run read-only over the actual CREATE dataset before the integration
was edited. The logical `/scratch` paths resolve to:

- Images: `/cephfs/volumes/hpc_data_prj/bc_ca_segmentation_in_tb_anatomy/3ca7561a-652a-49af-b067-58fe0d979778/datasets/RIM-ONE_DL_images/partitioned_by_hospital`
- Masks: `/cephfs/volumes/hpc_data_prj/bc_ca_segmentation_in_tb_anatomy/3ca7561a-652a-49af-b067-58fe0d979778/datasets/RIM-ONE-DL_masks`

Only `partitioned_by_hospital` was enumerated. The alternative
`partitioned_randomly` tree was not read.

### Counts, joins, and filename structure

| Hospital folder | Class | Images |
| --- | --- | ---: |
| `test_set` | glaucoma | 56 |
| `test_set` | normal | 118 |
| `training_set` | glaucoma | 116 |
| `training_set` | normal | 195 |
| **Total** | | **485** |

The mask tree contains 344 glaucoma masks and 626 normal masks, for 970 total.
The end-anchored expression `-1-(Cup|Disc)-[A-Za-z]\.png$` parsed all mask
names by stripping from the end, including hyphenated r3 stems.

- joined image stems: 485
- unparseable masks: 0
- duplicate images or masks: 0
- missing Cup/Disc masks: 0
- orphan masks: 0
- image/mask class-folder disagreements: 0
- stems resolving in both mask class folders: 0
- image/mask dimension mismatches: 0 across all 485 triplets
- `right_half` images: 0
- unparseable r3 eye/case stems: 0
- duplicated r3 case identifiers: 0
- r3 eye tokens: 65 L and 72 R

The r3 checks do not establish patient independence. The required caveat is:
**r3 filenames encode eye but not patient, so fellow-eye correlation is
undetectable from filenames rather than known to be absent**.

### Release × class table

| Release | Glaucoma | Normal | Total |
| --- | ---: | ---: | ---: |
| r1 | 12 | 86 | 98 |
| r2 | 108 | 142 | 250 |
| r3 | 52 | 85 | 137 |
| **Total** | **172** | **313** | **485** |

Every cell has at least three images, so no release needed the documented
release-only fallback.

### Mask format and polarity

A balanced sample of 60 pairs contained 10 images from each release/class cell.
A 12-mask format subsample was PIL mode `L`, two-dimensional NumPy `uint8`, and
had exactly `{0,255}` values. Header-only size reads were used for all 485
dimension triplets; mask arrays were decoded only for content checks.

Containment on the 60-pair sample was:

| Candidate polarity | Cup-outside-Disc pixels |
| --- | ---: |
| as read, white/high foreground | 0 of 1,571,376 Cup pixels |
| inverted, black/low foreground | 4,049,176 |

Therefore white/high is foreground and inversion is wrong. Full preflight later
found five small source-boundary defects totalling 2,505 pixels:

| Stem | Raw Cup pixels outside Disc |
| --- | ---: |
| `r2_Im319` | 46 |
| `r2_Im347` | 677 |
| `r2_Im357` | 7 |
| `r2_Im422` | 1,655 |
| `r2_Im427` | 120 |

These exact defects are pinned. The canonical decoder applies `cup &= disc` only
to those measured source defects and then asserts binary output and Cup ⊆ Disc.
A changed stem or pixel count fails closed.

Foreground fractions are corroboration only, not a polarity gate:

| Structure | Minimum | Median | Mean | Maximum |
| --- | ---: | ---: | ---: | ---: |
| Cup, 60-pair sample | 0.002515 | 0.104049 | 0.116031 | 0.309215 |
| Disc, 60-pair sample | 0.260019 | 0.401709 | 0.401712 | 0.596242 |

All 60 sampled Cup masks and all 60 Disc masks were single 4-connected
components. Every sampled centroid was within 0.25 of the frame diagonal from
the frame centre. Median normalized offsets were 0.0315 for Cup and 0.0079 for
Disc.

### Native resolutions

All 485 sources are square. There are 269 distinct native edge lengths from 274
through 793 pixels:

| Native edge | Images |
| --- | ---: |
| 250–299 | 6 |
| 300–349 | 44 |
| 350–399 | 62 |
| 400–449 | 113 |
| 450–499 | 55 |
| 500–549 | 25 |
| 550–599 | 39 |
| 600–649 | 58 |
| 650–699 | 44 |
| 700–749 | 24 |
| 750–799 | 15 |
| **Total** | **485** |

## Phase 1 — dispatch audit

| Site | REFUGE / Drishti behavior | RIM-ONE-DL branch or requirement |
| --- | --- | --- |
| `src/spfilm/data.py::discover_refuge_training`, `discover_drishti` | Fail-closed dataset-specific pairing | `discover_rim_one_dl` names only the hospital tree, joins sibling class folders, validates 485/970, release/class/hospital structure, dimensions, r3 case uniqueness, and source defects |
| `src/spfilm/data.py::decode_mask_channels` | REFUGE combined low-valued mask; Drishti 3-of-4 soft-map consensus | Foreground-high `L` masks, exact `{0,255}`, canonical `[disc,cup]` `uint8`, pinned source repair, binary and containment assertions |
| `src/spfilm/data.py::load_rim_one_dl_split_manifest` | Not used by existing datasets | Schema-2 provenance gate plus exact 340/48/97 counts, disjointness, exact discovered-stem union, release presence, and no listed-but-missing stems |
| `src/spfilm/engine.py::discover_config_records` | Dispatches existing discovery unchanged | Dispatches `rim_one_dl` to the sibling-tree adapter |
| `src/spfilm/engine.py::build_splits` | REFUGE seeded in-domain split; Drishti locks provider test | RIM reads the committed manifest and never regenerates a runtime split |
| `run_stage_s2.py` pool/split gates | 400 → 256/64/80; 101 → 40/10/51 | 485 → 340/48/97 with a committed-manifest label |
| `run_stage_s2.py::labels_for_dataset`, input notes, artifact names | REFUGE/Drishti disk-facing labels | RIM provenance, run prefix, and already-ONH-cropped full-source input note |
| `src/spfilm/engine.py::evaluate` and RIM CSV annotation | Existing datasets retain 512-grid HD95 and their CSV schema | RIM uses each `letterbox_scale` to convert HD95 to native pixels; CSV carries release, hospital/class metadata, native size, scale, and unit |
| `preflight_rim_one.py` | Dedicated existing preflights remain unchanged | Discovery, manifest provenance, polarity, containment, dimensions, class folders, runtime, and disk gates |
| `generate_rim_one_dl_split.py` | No runtime role | One-shot seeded largest-remainder generator; prints the six-cell table and records fallbacks/provenance |

The prompt names `run_stage2.py`, but that tracked file was already deleted in
the user's dirty working tree before this task. It was not restored. The active
submission wrapper invokes `run_stage_s2.py`, which is the implemented path.

### Existing config and wrapper

The local dirty `configs/stage2_rimone_create.json` currently contains:

- dataset: `rim_one_dl`
- data root: `/scratch/prj/bc_ca_segmentation_in_tb_anatomy/datasets/RIM-ONE_DL_images`
- manifest: `splits/rim_one_dl.json`
- image size 512, batch size 8, workers 8, epochs 300
- patience 20, minimum epochs 30, requested device `cuda`

That local data root is one level too deep: the adapter needs the parent that
contains both sibling trees. The config was not silently rewritten.

The deployed CREATE config differs and correctly uses
`/scratch/prj/bc_ca_segmentation_in_tb_anatomy/datasets`. The deployed checkout
is based on `ec384d27ec3a1c5f1273bd516ab9f3e1d7eeee7d` with the RIM integration
untracked, whereas the local checkout is based on `2092c7b` plus this task's
changes. Deployment reconciliation remains required.

`submit_rimone_s2.sh` expects:

- config: `configs/stage2_rimone_create.json`
- output directory: `artifacts/runs/rim-one_s2_${SLURM_JOB_ID}`
- partition: `interruptible_gpu`
- one node, one task, nine CPUs, 32 GiB RAM, one GPU
- wall time: 30 minutes
- exclusions: `erc-hpc-comp[048,054,170-175,177,178,196]`
- stdout/stderr: `/users/k23123868/edward/logs/rimone_s2_%j.{out,err}`

Its Slurm job name remains the copied `drishti_s2`, and its start banner says
`starting refuge_s2`; both are cosmetic conflicts. The wrapper and config were
not edited, as required.

The completed REFUGE and Drishti runs' `resolved_config.json` files retain their
pre-move data paths. Those historical results stand, but the recorded configs are
not rerunnable verbatim. This should be stated in any focused commit message so
it is not later misread as a result bug.

## Phase 2 — implementation

### Discovery and masks

`discover_rim_one_dl` returns one record per image with image, Disc, and Cup
paths; release prefix; hospital folder; diagnosis class; native `(W,H)`; and the
pinned source-repair count. It enumerates four explicit hospital/class image
directories and two explicit mask-class directories, never a broad image-tree
walk.

It fails closed on missing directories, non-PNG hospital rasters, changed counts,
unparseable suffixes or releases, duplicate/ambiguous/misplaced masks, missing or
orphan masks, duplicate image stems, any `right_half`, duplicate/unparseable r3
case identifiers, dimension mismatch, non-square sources, non-binary masks,
changed release/hospital totals, or changed source containment defects.

The loader performs no inversion. It outputs exactly two binary channels ordered
Disc then Cup and asserts Cup ⊆ Disc after the pinned canonicalization.

### Frozen manifest

`splits/rim_one_dl.json` is now schema 2 and includes:

- generator: `generate_rim_one_dl_split.py`
- generation commit: `2092c7b79653b5a4c6680eedad5ba0fe98a23896`
- generation working tree: dirty, recorded explicitly
- seed: 42
- generation date: 2026-08-26 UTC
- the full release × class table
- release-only fallbacks: none
- the exact fellow-eye caveat

Regeneration preserved every stem assignment. The canonical JSON hash of the
`partitions` object was `7870147cf3e5b9c2c13b861edb6dbced033fd10bdf6c82a1fb39a616b0543c8b`
both before and after the provenance upgrade.

| Split | Total | r1 | r2 | r3 | Glaucoma | Normal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 340 | 68 | 175 | 97 | 120 | 220 |
| Validation | 48 | 10 | 25 | 13 | 17 | 31 |
| Test | 97 | 20 | 50 | 27 | 35 | 62 |

| Split | r1 G | r1 N | r2 G | r2 N | r3 G | r3 N |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 8 | 60 | 75 | 100 | 37 | 60 |
| Validation | 1 | 9 | 11 | 14 | 5 | 8 |
| Test | 3 | 17 | 22 | 28 | 10 | 17 |

Every release and every release/class cell appears in all three partitions.

### Per-image metrics and boundary units

The RIM per-image CSV fields are:

`image_id,release_prefix,hospital_split,diagnosis_class,native_width,native_height,letterbox_scale,hd95_unit,structure,dice,iou,hd95,acc,tp,fp,fn,tn`

For RIM, `letterbox_scale = 512 / max(native_width,native_height)`. HD95 measured
on the resampled grid is multiplied by `1 / letterbox_scale` before it is written
or summarized, so the reported unit is `native_px`. REFUGE and Drishti keep their
existing letterboxed-grid HD95 behavior and CSV schemas.

The release prefix is present on every RIM per-image row, so r1/r2/r3 results can
be filtered without reparsing filenames.

### Hospital evaluation removal

The previous `secondary_hospital_partition` model pass, report block, and
`hospital_test_per_image_metrics.csv` artifact have been removed. Hospital folder
metadata remains provenance only. A source search found no
`secondary_hospital`, `hospital_test_per_image`, or hospital-result formatting
path, and the final smoke produced no hospital-named artifact.

## Phase 3 — verification

### Final-code data preflight

Command:

```bash
MPLCONFIGDIR=/private/tmp/spfilm-rim-mpl .spfilm/bin/python -u \
  preflight_rim_one.py \
  --config configs/stage2_rimone_create.json \
  --data-root /Users/edwardian0/Desktop/Projects/Research/SpFilm/datasets \
  --out-parent /private/tmp \
  --skip-runtime
```

Full output:

```text
host: Edwards-MacBook-Air.local
cwd:  /Users/edwardian0/Desktop/Projects/Research/SpFilm/code/spfilm

=== config / discovery / manifest ===============================
  config                /Users/edwardian0/Desktop/Projects/Research/SpFilm/code/spfilm/configs/stage2_rimone_create.json
  data_root (config)    /scratch/prj/bc_ca_segmentation_in_tb_anatomy/datasets/RIM-ONE_DL_images
  data_root (override)  /Users/edwardian0/Desktop/Projects/Research/SpFilm/datasets
  image tree            /Users/edwardian0/Desktop/Projects/Research/SpFilm/datasets/RIM-ONE_DL_images/partitioned_by_hospital
  mask tree             /Users/edwardian0/Desktop/Projects/Research/SpFilm/datasets/RIM-ONE-DL_masks
  split manifest        /Users/edwardian0/Desktop/Projects/Research/SpFilm/code/spfilm/splits/rim_one_dl.json
  paired images         485 (expected 485)
  paired mask PNGs      970 (expected 970)
  hospital/class        {('test_set', 'glaucoma'): 56, ('test_set', 'normal'): 118, ('training_set', 'glaucoma'): 116, ('training_set', 'normal'): 195}
  class-folder agreement PASS (enforced during discovery)
  release totals        {'r1': 98, 'r2': 250, 'r3': 137}
  manifest counts       {'train': 340, 'val': 48, 'test': 97}
  manifest generator    generate_rim_one_dl_split.py
  generation commit     2092c7b79653b5a4c6680eedad5ba0fe98a23896
  generation date       2026-08-26
  generation seed       42
  release/class table   {'r1': {'glaucoma': 12, 'normal': 86}, 'r2': {'glaucoma': 108, 'normal': 142}, 'r3': {'glaucoma': 52, 'normal': 85}}
  release-only fallback []
  fellow-eye caveat     r3 filenames encode eye but not patient, so fellow-eye correlation is undetectable from filenames rather than known to be absent
  train               release={'r1': 68, 'r2': 175, 'r3': 97} class={'glaucoma': 120, 'normal': 220} joint={'r1_glaucoma': 8, 'r1_normal': 60, 'r2_glaucoma': 75, 'r2_normal': 100, 'r3_glaucoma': 37, 'r3_normal': 60}
  val                 release={'r1': 10, 'r2': 25, 'r3': 13} class={'glaucoma': 17, 'normal': 31} joint={'r1_glaucoma': 1, 'r1_normal': 9, 'r2_glaucoma': 11, 'r2_normal': 14, 'r3_glaucoma': 5, 'r3_normal': 8}
  test                release={'r1': 20, 'r2': 50, 'r3': 27} class={'glaucoma': 35, 'normal': 62} joint={'r1_glaucoma': 3, 'r1_normal': 17, 'r2_glaucoma': 22, 'r2_normal': 28, 'r3_glaucoma': 10, 'r3_normal': 17}
  duplicate image hash groups  0

=== polarity / containment / dimensions =========================
  mask modes            {('cup', 'L'): 485, ('disc', 'L'): 485}
  mask dtypes           {('cup', 'uint8'): 485, ('disc', 'uint8'): 485}
  unique values         {'disc': [0, 255], 'cup': [0, 255]}
  cup white fraction    min=0.002296 median=0.084576 mean=0.103307 max=0.349258
  disc white fraction   min=0.260019 median=0.407895 mean=0.406741 max=0.596242
  polarity              white/high pixels are foreground (no inversion)
  raw cup repairs       {'r2_Im319': 46, 'r2_Im347': 677, 'r2_Im357': 7, 'r2_Im422': 1655, 'r2_Im427': 120}
  raw repair pixels     2505
  inverted containment  39381669 violating pixels
  checked dimensions    485 image/disc/cup triplets
  native resolutions    269 distinct; min=(274, 274) max=(793, 793)

=== runtime probes skipped ======================================
  SKIP: rerun without --skip-runtime on a CREATE GPU compute node

=== disk ========================================================
  checked               /private/tmp
  free                  71.4 GiB

=== summary =====================================================
  PASS  discovery/manifest
  PASS  masks/polarity

All data gates passed; CUDA runtime gates remain to be run on CREATE.
```

The deployed CREATE checkout's data-only preflight also exited 0 against the
actual `/scratch` root: 485/970, identical split distributions, the same five
source defects, 39,381,669 inverted violations, and 269 native resolutions. It
used the older deployed code; final schema-2 provenance must be rechecked after
deployment. No CUDA probe was run.

### Tests and executable smoke

- Python compilation passed for every changed Python file.
- Unit suite: 22 tests passed, including native-HD95 scaling and invalid-scale
  failure paths.
- A two-epoch CPU smoke completed train/validation/test, checkpoint reload,
  per-image CSV, JSON, and human-readable summary paths.
- The smoke CSV used `hd95_unit=native_px`, and the summary column was
  `HD95 (native px)`.
- No hospital evaluation key, CSV, print block, or artifact was produced.
- Smoke scores are plumbing evidence only, not a scientific result.

### REFUGE and Drishti invariance digests

The precondition passed locally: REFUGE discovery returned a plausible 400
records and Drishti returned 101. Each record list was serialized to JSON with
sorted keys and absolute paths. A temporary detached worktree at clean `HEAD`
was compared with the working tree:

| Dataset | Clean HEAD SHA-256 | Working tree SHA-256 | Verdict |
| --- | --- | --- | --- |
| REFUGE | `30db73e5929c473d6098c2c3ad49ab91386afdde91dfdf769ac9d118198d9a96` | `30db73e5929c473d6098c2c3ad49ab91386afdde91dfdf769ac9d118198d9a96` | identical |
| Drishti-GS | `f4b056636a4f7c6c5e781e755343b9e3065e4d8b16cb86c8426eb7dad36b3f3a` | `f4b056636a4f7c6c5e781e755343b9e3065e4d8b16cb86c8426eb7dad36b3f3a` | identical |

This is discovery invariance evidence. It is not a claim that the completed
historical runs can be replayed from their stale resolved paths.

## Findings

### Why the hospital result was cut

The hospital `test_set` has 174 images. Under the frozen random primary manifest,
118 are in primary training, 19 in validation, and 37 in primary test. A metric
over all 174 would therefore include 137 training/validation images and mostly
measure seen-data performance while being labelled as generalisation.

It is not an r1 holdout either, because r1 is deliberately included across the
primary train/validation/test split. The uncontaminated intersection is just the
37 primary-test images already present in the primary CSV and filterable by
release prefix. The r1 subset of the primary test contains 20 images and is a
weak read. The r1 generalisation question belongs to leave-one-domain-out Step 3,
not this in-domain Step 2 run.

### Boundary units

Per-image scale factors were plumbed through. RIM HD95 is reported in native
pixels, with the scale and unit carried on every CSV row. Boundary metrics were
not nulled. REFUGE and Drishti remain on their existing grid-pixel convention.

### Pre-crop ROI asymmetry

RIM-ONE-DL is already tightly cropped to the optic nerve head, so the ONH occupies
a much larger fraction of the input than it does in full-fundus REFUGE and
Drishti images. A model trained on these inputs is segmenting a pre-localised
region. Applying it to full-fundus input is a different task. This is an open
question for Pushpendra when considering the proposed cross-dataset ground-truth
ROI crop; no decision was made here.

### Open items and exact handoff

Could not determine or complete under this task boundary:

- CUDA availability, AMP numerics, and an unskipped GPU preflight; GPU/Slurm work
  was explicitly prohibited.
- Final-code behavior on CREATE after deployment; the cluster checkout is dirty,
  older, and contains the prior RIM integration as untracked files.
- Which local dirty config path should win during reconciliation. The deployed
  CREATE parent root is valid; the local image-tree root is not valid for the
  sibling-tree adapter.

After the final code and schema-2 manifest are deliberately deployed, the config
root is reconciled to the parent dataset directory, and an unskipped GPU
preflight exits 0, the exact training submission command is:

```bash
sbatch /users/k23123868/edward/spfilm/submit_rimone_s2.sh
```

No GPU job or Slurm submission was made.
