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

## Directory map

```text
spfilm/
├── configs/
│   ├── stage2_refuge.json              # frozen first-baseline settings
│   └── rim_one_r3_manifest.example.csv # explicit RIM pairing contract
├── src/spfilm/
│   ├── data.py                         # discovery, decoding, splits, Dataset
│   ├── engine.py                       # train/validate/test orchestration
│   ├── losses.py                       # BCE + soft Dice training objective
│   ├── metrics.py                      # per-image disc/cup Dice and IoU
│   ├── model.py                        # plain 2D U-Net
│   └── visualization.py                # mask and prediction QA figures
├── tests/                              # fast contract and shape tests
├── run_stage2.py                       # audit / inspect / train / all CLI
├── STAGE2.md                           # research and execution protocol
└── artifacts/                          # generated locally
```

## Output contract

A real run writes the following under `artifacts/stage2_refuge/`:

- `data_audit.json`: decoded-mask and source-layout checks.
- `split_manifest.csv`: the exact, disjoint sample IDs for train/validation/test.
- `mask_contact_sheet.png`: twelve source images with normalized masks.
- `best_model.pt`: checkpoint selected by validation loss, not test Dice.
- `history.csv` and `training_curves.png`: epoch-level training evidence.
- `test_metrics.json`: disc and cup Dice/IoU reported separately.
- `test_predictions.png`: targets, predictions, false positives, and false negatives.
- `resolved_config.json`: the settings that actually ran.

## Current RIM-ONE finding

`../../datasets/RIM-ONE_DL_images` is not the release required by the brief. It
contains 485 unique classification images repeated under two alternative split
schemes (970 PNG files total) and no optic-disc/cup masks. The required
RIM-ONE-r3 segmentation release has 159 images and a published 99/60 split.

The code therefore refuses to invent RIM masks. After the correct release is
present, make a 159-row manifest from
`configs/rim_one_r3_manifest.example.csv`, explicitly choosing the averaged
annotations (or another documented policy). This does not block the first
REFUGE-only baseline, but it must be resolved before the multi-domain stage.

