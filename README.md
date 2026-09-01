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
│   ├── model.py                        # plain 2D U-Net
│   └── visualization.py                # mask and prediction QA figures
├── tests/                              # fast contract and shape tests
├── run_stage2.py                       # audit / inspect / train / all CLI
├── run_stage3_lodo.py                  # prepare / check / run LODO CLI
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
