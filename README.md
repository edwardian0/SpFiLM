# SpFiLM fundus project

This directory is the implementation workspace for the fundus-domain SpFiLM
project. Stage 2 establishes a plain U-Net baseline before any FiLM or SpFiLM
code is introduced.

The raw downloads stay in `../../datasets`; they are not copied into this
directory. Generated checkpoints, reports, split manifests, and figures go in
`artifacts/`.

## Stage 2 quick start

Use the existing learning environment from this directory:

```bash
cd /Users/edwardian0/Desktop/Projects/Research/SpFiLM/code/spfilm

# 1. Check all downloaded datasets and normalize every available mask.
../learning/.spfilm2/bin/python run_stage2.py audit

# 2. Inspect twelve normalized REFUGE masks before training.
../learning/.spfilm2/bin/python run_stage2.py inspect --dataset refuge

# Drishti-GS is not used by the first baseline, but its adapter is checked too.
../learning/.spfilm2/bin/python run_stage2.py inspect --dataset drishti

# 3. Exercise decoding, forward/backward, checkpointing, and evaluation cheaply.
../learning/.spfilm2/bin/python run_stage2.py train \
  --config configs/stage2_refuge.json --smoke

# 4. Run the real Stage 2 experiment from a fixed config and seed.
../learning/.spfilm2/bin/python run_stage2.py all \
  --config configs/stage2_refuge.json
```

Do not treat the smoke-test Dice as a result. It uses one 128 x 128 batch and
exists only to prove that the pipeline is connected correctly.

The detailed protocol, decisions, score sanity range, and exit gate are in
[`STAGE2.md`](STAGE2.md).

## Stage 3 LODO quick start

Stage 3 reuses each domain's locked Stage 2 train/validation/test roles. For a
given held-out domain, training and validation are the unions of the other
domains' train and validation partitions; testing is only the held-out domain's
locked test partition. Source-domain tests and held-out train/validation samples
are excluded.

Prepare the membership file once, inspect it, then run the full validation gate:

```bash
.spfilm/bin/python run_stage3_lodo.py \
  --config configs/stage3_lodo.json prepare

git diff -- splits/lodo/lodo_manifest.json

.spfilm/bin/python run_stage3_lodo.py \
  --config configs/stage3_lodo.json check
```

`prepare` refuses to replace changed membership unless `--force` is explicit.
`check` re-discovers all four datasets, recomposes every fold from the config,
proves exact manifest coverage, and decodes every mask. The lighter
`check --skip-mask-audit` still checks paths and membership but is not the full
pre-training gate. Commit the reviewed `splits/lodo/lodo_manifest.json`; `run`
does not create it automatically.

Run one plumbing rehearsal, then one real fold/seed combination:

```bash
.spfilm/bin/python run_stage3_lodo.py \
  --config configs/stage3_lodo.json run \
  --held-out-domain refuge_zeiss --seed 42 --smoke --device cpu

.spfilm/bin/python run_stage3_lodo.py \
  --config configs/stage3_lodo.json run \
  --held-out-domain refuge_zeiss --seed 42
```

Smoke output is deliberately marked as non-scientific. A non-smoke run refuses
to overwrite a non-empty run directory. `--all` explicitly runs all 20
domain/seed combinations sequentially; on CREATE, submit them as independent
jobs instead:

```bash
sbatch submit_lodo_stage3.sh refuge_zeiss 42
sbatch --time=0-00:20:00 submit_lodo_stage3.sh refuge_zeiss 42 --smoke
```

The CREATE wrapper uses `configs/stage3_lodo_create.json`. Prepare, review, and
commit the same manifest before submitting. Its current wall time must cover the
configured 300 epochs: `early_stopping_mode: monitor` selects a checkpoint but
does not shorten training.

## Stage 4 (Step 4) quick start: Global FiLM over three domains

Step 4 adds the honest baseline: channel-wise (global) FiLM after each encoder
block of the *same* U-Net, trained with the true source-domain code. It runs
leave-one-domain-out over **three domains, RIM-ONE-DL dropped for now**: each
fold trains on two domains (80 train / 20 val) and tests on the third (50).
RIM-ONE-DL stays under `domains` so the locked four-domain manifests still
validate; the protocol's `held_out_domains` list is the *active* set and folds
are composed from it alone. Each domain's budgeted partitions are unchanged, so
a held-out domain's 50 test images are the same ones Stage 3 scored.

Under leave-one-domain-out the held-out domain has no code the model was
trained with, so at test time each image gets the code of the source domain
whose training appearance it is nearest to (`src/spfilm/film/conditioning.py`;
the policy agreed with the supervisor). The FiLM layer itself is
`src/spfilm/film/global_film.py`, matches the reference implementation
(`p-singh-kcl/spatial_film_parcellation`, `models/film_mlp.py`) block for block,
and is the K=0 case SpFiLM must reduce to.

Because the Stage 3 plain runs trained on RIM-ONE-DL, they are **not** the pair
for this arm. Step 4 therefore has two configs that differ only in `arm` (a test
asserts this): `stage4_plain_3dom*.json` re-runs the plain U-Net under the
three-domain folds and `stage4_global_film_3dom*.json` adds FiLM. Both use the
same `run_stage3_lodo_3_1_fixed.py`:

```bash
.spfilm/bin/python run_stage3_lodo_3_1_fixed.py \
  --config configs/stage4_global_film_3dom.json check --skip-mask-audit

.spfilm/bin/python run_stage3_lodo_3_1_fixed.py \
  --config configs/stage4_global_film_3dom.json run \
  --held-out-domain drishti_gs --seed 42 --smoke --device cpu

sbatch --time=0-00:20:00 submit_stage4_global_film.sh drishti_gs 42 --smoke
sbatch submit_stage4_plain.sh drishti_gs 42
sbatch submit_stage4_global_film.sh drishti_gs 42
```

The full protocol is 3 domains x 5 seeds x 2 arms = 30 submissions. Put the
arms next to each other (FiLM minus plain on identical test images; the tool
refuses to pair arms that trained on different source sets, and a partial grid
pairs against the same seeds of the other arm):

```bash
.spfilm/bin/python aggregate_stage4_film.py --expected-seeds 42 \
  --report-out run_reports/stage4_global_film.md
```

`aggregate_stage3_fixed.py` now needs `--arm <experiment_name>` once a run root
holds more than one arm; it refuses to mix them, and refuses runs that disagree
on the active domain set.

### Stage 4 output contract

A Global FiLM run writes the Stage 3 artifacts plus:

- `domain_selector.json`: the fold's code vocabulary and the nearest-domain
  reference statistics (also stored inside both checkpoints).
- `test_conditioning_per_image.csv`: per held-out image, the code used, the
  distance to every source centroid, and the descriptor.
- `val_selector_per_image.csv`: the selector run on source validation images,
  whose true domain is known; `conditioning.selector_validation` in
  `test_metrics.json` gives its accuracy and confusion.
- `test_fixed_code_<domain>_per_image_metrics.csv`: the held-out set scored
  once under each source code; `conditioning.fixed_code_sweep` summarises it
  and `nearest_domain_minus_best_fixed_code_dice` says whether the selector
  found the best code. If the sweep is flat, the conditioning is inert.
- `test_metrics.json["arm"]` and `["conditioning"]`; `parameter_count` includes
  the FiLM generators. Plain runs record `arm: "plain"` and
  `conditioning: null` and are otherwise unchanged.

## Directory map

```text
spfilm/
├── configs/
│   ├── stage2_refuge.json              # frozen first-baseline settings
│   └── rim_one_r3_manifest.example.csv # explicit RIM pairing contract
├── src/spfilm/
│   ├── data.py                         # discovery, decoding, splits, Dataset
│   ├── engine.py                       # train/validate/test orchestration
│   ├── lodo.py                         # immutable partitions/folds/manifest
│   ├── stage3.py                       # Stage 3 config and record resolution
│   ├── losses.py                       # BCE + soft Dice training objective
│   ├── metrics.py                      # per-image disc/cup Dice and IoU
│   ├── model.py                        # plain 2D U-Net and its FiLM-conditioned wrapper
│   ├── film/global_film.py             # channel-wise FiLM layer (Step 4)
│   ├── film/conditioning.py            # domain codes; nearest-source-domain rule at test
│   └── visualization.py                # mask and prediction QA figures
├── tests/                              # fast contract and shape tests
├── run_stage2.py                       # audit / inspect / train / all CLI
├── run_stage3_lodo.py                  # prepare / check / run LODO CLI
├── run_stage3_lodo_3_1_fixed.py        # fixed-budget LODO runner, plain and global_film arms
├── aggregate_stage4_film.py            # FiLM next to plain: paired test + selector diagnostics
├── submit_lodo_stage3.sh               # one CREATE fold/seed submission
├── STAGE2.md                           # research and execution protocol
└── artifacts/                          # generated locally
```

## Stage 2 output contract

A real run writes the following under `artifacts/stage2_refuge/`:

- `data_audit.json`: decoded-mask and source-layout checks.
- `split_manifest.csv`: the exact, disjoint sample IDs for train/validation/test.
- `mask_contact_sheet.png`: twelve source images with normalized masks.
- `best_model.pt`: checkpoint selected by validation loss, not test Dice.
- `history.csv` and `training_curves.png`: epoch-level training evidence.
- `test_metrics.json`: disc and cup Dice/IoU reported separately.
- `test_predictions.png`: targets, predictions, false positives, and false negatives.
- `resolved_config.json`: the settings that actually ran.

## Stage 3 output contract

Each run writes the normal engine artifacts plus `lodo_run.json` and
`resolved_stage3_config.json`. `test_metrics.json` also records the held-out
domain, run seed, manifest/config hashes, locked and executed split counts, and
whether the run was a smoke rehearsal. Disc and cup metrics remain separate.
