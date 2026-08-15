# Stage 2 - plain U-Net on one fundus domain

## Outcome and boundary

Stage 2 answers one question: can a plain, unconditioned U-Net learn optic-disc
and optic-cup segmentation on one camera domain through a script that can be
rerun from raw files to a locked test report?

This stage does not contain Global FiLM, SpFiLM, leave-one-domain-out training,
or RIGA contour conversion. Those changes would make a broken data pipeline and
a broken research method indistinguishable.

The exit gate is met only when a full run produces:

1. a passing raw-data audit;
2. a visually accepted 12-sample mask contact sheet;
3. a saved, disjoint split manifest;
4. a plain U-Net checkpoint selected without using the test set;
5. separate test Dice values for disc and cup; and
6. prediction/error overlays that look anatomically consistent.

This protocol implements Step 2 of
[`Edward_Project_Brief.pdf`](../../Edward_Project_Brief.pdf). The brief's exact
move-on criterion is a sensible Dice score from a script that runs from start
to finish.

## Decisions made for the first run

### 1. Use REFUGE Training400 as the one domain

REFUGE's training and validation sets were captured by different scanners.
DoFE therefore treats them as separate domains. Testing `Training400` against
`Validation400` would already measure domain shift and belongs in Stage 3.

The first run uses only the 400-image `Training400` camera pool and makes a
seeded, diagnosis-stratified split inside it:

| Split | Images | Purpose |
|---|---:|---|
| Train | 256 | Gradient updates |
| Validation | 64 | Checkpoint selection and early stopping |
| Test | 80 | One final, locked same-domain report |

The test count preserves DoFE's 320/80 development/test ratio. Validation is
taken only from the 320-image development side. The exact sample IDs are saved
to `split_manifest.csv` so later methods use the identical evidence.

### 2. Normalize every source to one mask contract

The model never sees a dataset-specific label convention. Every adapter returns
a binary `float32` tensor shaped `[2, H, W]`:

| Channel | Meaning |
|---:|---|
| 0 | optic disc, including the cup |
| 1 | optic cup |

The cup must be a subset of the disc. Empty masks, unexpected source values,
shape mismatches, missing pairs, or cup pixels outside the disc stop the run.

Source decoding is explicit:

| Dataset | Raw convention | Internal conversion |
|---|---|---|
| REFUGE | one mask: `0=cup`, `128=disc ring`, `255=background` | `disc = raw <= 128`; `cup = raw == 0` |
| Drishti-GS | separate soft maps with values `0,64,128,191,255` from four readers | `disc/cup = raw >= 191`, meaning at least three of four readers |
| RIM-ONE-r3 | separate disc/cup annotations with multiple possible annotation policies | explicit 159-row manifest and explicit foreground polarity; no filename or expert-policy guessing |

### 3. Do not use a ground-truth crop at test time

DoFE localizes an 800 x 800 optic-disc region before resizing it to 256 x 256.
Using the test mask to obtain that crop would leak ground truth. Stage 2 instead
letterboxes the full fundus image and mask to 512 x 512. This is slower, but it
keeps the first baseline honest and leaves ROI localization as an explicit
future experiment.

Only the training split receives mild horizontal flips, rotations, brightness,
and contrast changes. Validation and test preprocessing is deterministic.

### 4. Keep the model and report plain

- Architecture: four-level 2D U-Net, RGB input, two output logits.
- Capacity/input: 16 base channels, 512 x 512 full-frame letterboxed input,
  batch size 2 (a practical default for the current local runtime).
- Conditioning: none.
- Objective: equal-weight binary cross-entropy plus channel-wise soft Dice.
- Optimizer: Adam, learning rate `1e-3`, weight decay `1e-5`.
- Schedule: up to 40 epochs, reduce learning rate on validation-loss plateaus,
  early stop after eight non-improving epochs.
- Checkpoint rule: lowest validation loss.
- Test threshold: `0.5`, fixed before the run.
- Report: per-image mean, standard deviation, and median Dice for disc and cup
  separately; IoU is also retained separately.

The training objective may reduce two channel losses to one scalar because
backpropagation requires a scalar. The scientific report never hides disc and
cup performance inside one average.

## Read DoFE's table before interpreting the run

The closest prior work is Wang et al.,
[DoFE: Domain-oriented Feature Embedding for Generalizable Fundus Image
Segmentation on Unseen Datasets](https://arxiv.org/abs/2010.06208). Its Table I
uses these optic-disc/cup domains and splits:

| DoFE domain | Dataset | Train + test |
|---:|---|---:|
| 1 | Drishti-GS | 50 + 51 |
| 2 | RIM-ONE-r3 | 99 + 60 |
| 3 | REFUGE training camera | 320 + 80 |
| 4 | REFUGE validation camera | 320 + 80 |

DoFE Table II reports the following **leave-one-domain-out** vanilla baseline,
trained on the other three domains and tested on the named unseen domain:

| Held-out domain | Cup Dice | Disc Dice |
|---|---:|---:|
| Drishti-GS | 0.7703 +/- 0.0066 | 0.9496 +/- 0.0033 |
| RIM-ONE-r3 | 0.7821 +/- 0.0123 | 0.8969 +/- 0.0158 |
| REFUGE training camera | 0.8028 +/- 0.0403 | 0.8933 +/- 0.0149 |
| REFUGE validation camera | 0.8474 +/- 0.0113 | 0.9009 +/- 0.0367 |

These values are a sanity range, not an acceptance threshold. DoFE used a
DeepLabV3+/MobileNetV2 model, ImageNet initialization, a learned ROI crop,
256 x 256 inputs, and multi-source cross-domain training. This Stage 2 run uses
a from-scratch full-frame U-Net on one domain, so an exact numerical comparison
would be false precision.

Before training, the working expectation is nevertheless concrete:

- a converged same-domain baseline should plausibly approach the rough region
  of 0.90 disc Dice and 0.80 cup Dice;
- disc below about 0.80 or cup below about 0.60 is a debugging signal, not a
  publishable negative result; inspect masks, class channels, resizing, loss,
  and predictions first;
- unusually perfect scores require a duplicate/split-leakage audit.

Those last thresholds are project debugging heuristics inferred from DoFE's
cross-domain table, not values claimed by the paper.

## Exact execution sequence

Run all commands from `code/spfilm` with the existing learning Python.

### Step 2.1 - audit what is actually on disk

```bash
../learning/.spfilm2/bin/python run_stage2.py audit
```

Expected current status:

- REFUGE: 400 image/mask pairs, three valid mask values, usable.
- Drishti-GS: provider split 50/51 and two soft maps per image, usable.
- RIM-ONE: blocked because the current RIM-ONE DL download has no segmentation
  masks. This does not block a REFUGE-only Stage 2 run.

Do not continue if REFUGE is not `ok` in `artifacts/data_audit.json`.

### Step 2.2 - inspect a dozen normalized masks by eye

```bash
../learning/.spfilm2/bin/python run_stage2.py inspect --dataset refuge
../learning/.spfilm2/bin/python run_stage2.py inspect --dataset drishti
```

For every tile, confirm:

- the green disc boundary surrounds the optic nerve head;
- the blue cup boundary is inside the disc;
- neither channel is inverted into the background;
- image and mask are spatially aligned after letterboxing; and
- no sample is blank, clipped, or paired with a different eye.

This is a human gate. A passing array assertion does not prove anatomical
alignment.

### Step 2.3 - run the cheap end-to-end smoke test

```bash
../learning/.spfilm2/bin/python run_stage2.py train \
  --config configs/stage2_refuge.json --smoke
```

This must complete image decoding, one forward/backward update, validation,
checkpoint save/reload, test metrics, and prediction export. Ignore its Dice.

### Step 2.4 - freeze the experimental inputs

Review `configs/stage2_refuge.json`. Do not tune it after looking at test Dice.
The run saves the resolved config and sample manifest. Keep the same manifest
for later plain U-Net, Global FiLM, and SpFiLM comparisons.

### Step 2.5 - train the real baseline

```bash
../learning/.spfilm2/bin/python run_stage2.py all \
  --config configs/stage2_refuge.json
```

`all` repeats the data audit before training. On a CPU this can be slow; use a
CUDA or MPS device when available. `num_workers` remains zero by default because
the earlier macOS learning pipeline showed shared-memory worker failures.

### Step 2.6 - interpret the result in the right order

1. Check `history.csv`: training and validation loss should both move, and disc
   and cup curves must be visible separately.
2. Check `test_predictions.png`: do not accept a number when contours are
   systematically shifted, inverted, empty, or spilling into background.
3. Read `test_metrics.json`: quote `test.disc.dice_mean` and
   `test.cup.dice_mean` separately, with their standard deviations.
4. Compare only broadly with DoFE's range and record all preprocessing
   differences.

### Step 2.7 - apply the exit gate

Stage 2 is complete when the real, non-smoke artifacts exist, the visual checks
pass, and both separate Dice values are sensible. Only then begin Stage 3's
leave-one-domain-out shift measurement. Do not add SpFiLM while any loader,
split, checkpoint, or metric question remains unresolved.
