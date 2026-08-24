Polarity: foreground-high (white); no inversion is required.
485-image join: all 485 images joined cleanly to one Cup and one Disc PNG.
Final split counts: 340 train / 48 validation / 97 primary test.
REFUGE/Drishti unaffected: legacy discovery projections are byte-identical before and after.
Remaining blocker: none locally; the CREATE data root and unskipped CUDA preflight still require live verification before `sbatch`.

# RIM-ONE-DL Step 2 wiring report

## Phase 0 — data and polarity

The verified local roots are:

- Images: `/Users/edwardian0/Desktop/Projects/Research/SpFilm/datasets/RIM-ONE_DL_images/partitioned_by_hospital`
- Masks: `/Users/edwardian0/Desktop/Projects/Research/SpFilm/datasets/RIM-ONE-DL_masks`

Only `partitioned_by_hospital` was enumerated. `partitioned_randomly` contains an
alternative split of the same images and is not touched by discovery.

### Counts and joins

| Hospital partition | Class | Images |
| --- | --- | ---: |
| `test_set` | glaucoma | 56 |
| `test_set` | normal | 118 |
| `training_set` | glaucoma | 116 |
| `training_set` | normal | 195 |
| **Total** | | **485** |

There are 970 raster mask PNGs. The end-anchored expression
`^(?P<stem>.+)-1-(?P<kind>Cup|Disc)-(?P<suffix>[A-Za-z])\.png$`
parsed all of them, including stems containing hyphens. Results:

- unparseable PNG names: 0
- duplicate image or mask keys: 0
- missing Cup/Disc pairs: 0
- orphan mask PNGs: 0
- expert/type suffixes: `T` for all 970 PNGs
- image/mask dimension mismatches: 0 across all 485 triplets

The text contour sidecars and `LICENSE.txt` are not raster masks and are not
included in the 970-mask gate.

### Release and source-partition distribution

| Release | Images | Hospital location |
| --- | ---: | --- |
| r1 | 98 | all in `test_set` |
| r2 | 250 | 76 in `test_set`, 174 in `training_set` |
| r3 | 137 | all in `training_set` |

This confirms why the provider hospital folders are unsuitable as the primary
in-domain split.

### Polarity and containment

Every mask is PIL mode `L`, NumPy dtype `uint8`, with exactly `{0, 255}` values.
White/high pixels are foreground:

| Structure | Minimum | Median | Mean | Maximum |
| --- | ---: | ---: | ---: | ---: |
| Cup | 0.002296 | 0.084576 | 0.103307 | 0.349258 |
| Disc | 0.260019 | 0.407895 | 0.406741 | 0.596242 |

Using low/black pixels as foreground produces 39,381,669 Cup-outside-Disc
pixels, so inversion is decisively wrong.

Five source masks contain small foreground-high boundary inconsistencies:

| Stem | Cup pixels outside Disc |
| --- | ---: |
| `r2_Im319` | 46 |
| `r2_Im347` | 677 |
| `r2_Im357` | 7 |
| `r2_Im422` | 1,655 |
| `r2_Im427` | 120 |
| **Total** | **2,505** |

These are pinned exactly in discovery. The RIM-ONE-DL decoder applies
`cup &= disc` only for those measured defects; a changed stem or pixel count
raises instead of being silently repaired.

A 12-image contact sheet sampled two records from every release/class stratum.
The rendered Disc and Cup contours align with the visible optic nerve head, and
the Cup is visibly the smaller inner foreground region, corroborating the
foreground-high decode.

### Native resolutions

All sources are square. There are 269 distinct native edge lengths, spanning
274 through 793 pixels. The complete binned distribution is:

| Native edge (pixels) | Images |
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

| Site | Existing behavior | RIM-ONE-DL requirement/result |
| --- | --- | --- |
| `src/spfilm/data.py::discover_refuge_training` / `discover_drishti` | Dataset-specific fail-closed discovery | Added `discover_rim_one_dl`; hospital tree only, 485/970 gates, anchored join, dimensions, releases and source defects |
| `src/spfilm/data.py::decode_mask_channels` | REFUGE combined-mask and Drishti consensus decoding | Added foreground-high binary decode and pinned source canonicalization; output remains binary `[disc, cup]` |
| `src/spfilm/engine.py::discover_config_records` | Dispatches `refuge`, `drishti`, and explicit r3 manifest | Added `rim_one_dl` dispatch |
| `src/spfilm/engine.py::build_splits` | REFUGE seeded split; other datasets use provider split | Added committed JSON-manifest branch; no runtime regeneration or provider fall-through |
| `run_stage_s2.py` count gates | REFUGE 400 and Drishti 101 | Added pool 485 and exact 340/48/97 gate |
| `run_stage_s2.py::labels_for_dataset` | Dataset-facing run labels | Added `rim_one_s2`, `rim_one_dl`, and RIM-ONE-DL provenance |
| `src/spfilm/engine.py` data audit | REFUGE-specific or provider split wording | Added explicit committed-manifest policy text |
| Per-image metrics | Fixed overlap columns | RIM-only post-write context adds release, hospital/class provenance, native dimensions, and letterbox scale; legacy CSV schemas are unchanged |
| Evaluation | One primary test result | Added a separately named hospital `test_set` evaluation and CSV |
| `preflight_drishti.py` | REFUGE/Drishti gates | Added the dedicated `preflight_rim_one.py` polarity, containment, dimension, discovery and manifest gates |
| `inspect_rim_download` | Stale claim that DL has no masks | It now recognises the paired local image/mask trees through the same adapter |

The active entry point is `run_stage_s2.py`; the inherited worktree had already
deleted `run_stage2.py` and `run_refuge_s2.py`. The RIM submit wrapper invokes the
active entry point.

`submit_rimone_s2.sh` expects:

- config: `configs/stage2_rimone_create.json`
- run directory: `artifacts/runs/rim-one_s2_${SLURM_JOB_ID}`
- partition: `interruptible_gpu`
- resources: one GPU, nine CPUs, 32 GiB RAM, 30 minutes
- exclusions: `erc-hpc-comp[048,054,170-175,177,178,196]`

Its copied `drishti_s2` job name and `starting refuge_s2` echo are cosmetic. The
wrapper was not rewritten, as required.

## Phase 2 — implementation

### Discovery and canonical masks

`discover_rim_one_dl` returns one `FundusRecord` per image with image, Disc and
Cup paths plus explicit release prefix, hospital split, diagnosis class, native
size and joint stratification key. It uses
`rim_one_dl_foreground_high`; no inversion is applied.

Discovery fails on a missing tree, non-PNG hospital raster, wrong 485/970 count,
bad release prefix, unparseable mask name, duplicate or orphan mask, missing pair,
dimension mismatch, non-square image, non-binary mask, changed release layout, or
changed source containment defect.

### Materialised split

`generate_rim_one_dl_split.py` was run once at seed 42. It sorts stems before
shuffling and uses integer largest-remainder allocation, then writes
`splits/rim_one_dl.json`. A second generation to `/private/tmp` was byte-identical:

`sha256 3746d094c0ac65cc0d3a6dd149a76881812bae76b955109b6956e8a27c7f7ca5`

The runtime only reads the committed stem lists. It verifies exact counts,
within- and across-partition uniqueness, exact union with discovery, absence of
listed-but-missing files, and all three releases in every partition.

| Split | Total | r1 | r2 | r3 | Glaucoma | Normal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 340 | 68 | 175 | 97 | 120 | 220 |
| Validation | 48 | 10 | 25 | 13 | 17 | 31 |
| Primary test | 97 | 20 | 50 | 27 | 35 | 62 |

Joint strata are:

| Split | r1 G | r1 N | r2 G | r2 N | r3 G | r3 N |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 8 | 60 | 75 | 100 | 37 | 60 |
| Validation | 1 | 9 | 11 | 14 | 5 | 8 |
| Primary test | 3 | 17 | 22 | 28 | 10 | 17 |

### Metric provenance and secondary evaluation

RIM-ONE-DL per-image metric CSVs now include:

`release_prefix,hospital_split,diagnosis_class,native_width,native_height,letterbox_scale`

The scale is `canvas_edge / max(native_width, native_height)`. This preserves the
information needed to interpret boundary distances across variable native sizes;
HD95 itself remains reported in letterboxed-grid pixels.

The full provider hospital `test_set` (174 images) is evaluated a second time and
written to `hospital_test_per_image_metrics.csv`. It is printed and stored under
`secondary_hospital_partition`, never merged with the primary result. Its overlap
with the primary manifest is 118 train / 19 validation / 37 primary-test images,
so it is explicitly labeled descriptive and not an independent holdout.

## Phase 3 — verification

### RIM-ONE-DL preflight

The local command was:

```bash
MPLCONFIGDIR=/private/tmp/spfilm-mpl .spfilm/bin/python -u preflight_rim_one.py \
  --data-root /Users/edwardian0/Desktop/Projects/Research/SpFilm/datasets \
  --skip-runtime
```

Full result:

```text
host: Edwards-MacBook-Air.local
cwd:  /Users/edwardian0/Desktop/Projects/Research/SpFilm/code/spfilm

=== config / discovery / manifest ===============================
  config                /Users/edwardian0/Desktop/Projects/Research/SpFilm/code/spfilm/configs/stage2_rimone_create.json
  data_root (config)    /Users/edwardian0/datasets/glaucoma_datasets
  data_root (override)  /Users/edwardian0/Desktop/Projects/Research/SpFilm/datasets
  image tree            /Users/edwardian0/Desktop/Projects/Research/SpFilm/datasets/RIM-ONE_DL_images/partitioned_by_hospital
  mask tree             /Users/edwardian0/Desktop/Projects/Research/SpFilm/datasets/RIM-ONE-DL_masks
  split manifest        /Users/edwardian0/Desktop/Projects/Research/SpFilm/code/spfilm/splits/rim_one_dl.json
  paired images         485 (expected 485)
  paired mask PNGs      970 (expected 970)
  hospital/class        {('test_set', 'glaucoma'): 56, ('test_set', 'normal'): 118, ('training_set', 'glaucoma'): 116, ('training_set', 'normal'): 195}
  release totals        {'r1': 98, 'r2': 250, 'r3': 137}
  manifest counts       {'train': 340, 'val': 48, 'test': 97}
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
  checked               /Users/edwardian0/Desktop/Projects/Research/SpFilm/code/spfilm/artifacts/runs
  free                  74.1 GiB

=== summary =====================================================
  PASS  discovery/manifest
  PASS  masks/polarity

All data gates passed; CUDA runtime gates remain to be run on CREATE.
```

### Regression and executable checks

- Unit tests: 19 passed.
- Final-code CPU smoke: two epochs, 2/1/1 manifest-derived split, primary and
  separate hospital metric paths both completed; no GPU or Slurm job was used.
- Primary smoke CSV contains the six RIM-only provenance/scale fields.
- Secondary smoke result is stored separately with explicit overlap metadata.
- Manifest regeneration was byte-identical.
- Python compilation passed for every changed Python file.

Legacy discovery was hashed over the original nine `FundusRecord` fields before
and after implementation:

| Dataset | Records | Before SHA-256 | After SHA-256 |
| --- | ---: | --- | --- |
| REFUGE | 400 | `4523f2f579fe7a583aa515904047a4888a8a1a746bbd21922df5567834236744` | identical |
| Drishti-GS | 101 | `d15e999633d27cc03d68d6e15f5b319ef56c5b595795feb67d50862af53b04ae` | identical |

### CREATE commands

After the two data trees exist under the configured
`~/datasets/glaucoma_datasets`, run the unskipped gate on a GPU compute node:

```bash
srun -p interruptible_gpu --gres=gpu:1 --time=0:10:00 \
  bash -l /users/k23123868/edward/spfilm/oncompute.sh \
  python -u /users/k23123868/edward/spfilm/preflight_rim_one.py \
  --config /users/k23123868/edward/spfilm/configs/stage2_rimone_create.json
```

If that passes, the exact training submission command is:

```bash
sbatch /users/k23123868/edward/spfilm/submit_rimone_s2.sh
```

No GPU job or Slurm submission was made during this wiring work.
