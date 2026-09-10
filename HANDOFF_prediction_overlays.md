# Session handoff: predicted-segmentation overlays on original images

**Written 2026-09-10.** Task: a script that draws predicted disc/cup contours on
the **original-resolution** fundus image, for **chosen** instances rather than
arbitrary ones. Nothing has been written yet — this is a specification plus the
facts you need so you do not rediscover them.

---

## 1. Why this task exists

The domain-shift analysis (`run_reports/s3_domain_shift_brief.md`) concluded that
the shift breaking cross-domain transfer is **geometric, not photometric**:
intensity distance anti-correlates with Dice, while disc-to-frame scale tracks it.
That conclusion rests entirely on aggregate statistics. Nobody has looked at what
the failures actually *look like*.

That is the gap this script fills, and it is why instance selection matters more
than the drawing does. If the geometric account is right, the failure modes should
be visibly **boundary and scale errors** — contours the right shape in the wrong
place or at the wrong size — not colour-driven confusion. If instead the failures
look like the model mistaking bright retina for disc, the account needs revisiting.

Pick instances that can discriminate between those, not a random sample.

---

## 2. Read this before anything else: the checkpoints are not here

**All 107 real runs under `artifacts/runs/` were synced back from CREATE without
their weights.** Only 17 `.pt` files exist locally and every one belongs to a
smoke run (`*_smoke*`, `seed_42_smoke`). There is no local checkpoint for any run
whose numbers appear in the reports.

Each run records where its weights live on CREATE, e.g. in
`artifacts/runs/<run>/single_source_run.json`:

```
"checkpoint": "/cephfs/volumes/hpc_home/k23123868/627d4c89-cd29-4f24-b3ae-4f85744f010c/edward/spfilm/artifacts/runs/<run>/best_model.pt"
```

So step 0 is one of:

- **pull the checkpoints you need** down from CREATE (a handful, not all 107 — one
  per cell you want to visualise), or
- **run the script on CREATE** and bring back only the PNGs.

Pulling is probably better: the images are local, the models are ~small
(`base_channels=16`), and inference on 6–20 images is fine on the Mac.

Do not start by writing the drawing code. Confirm you can load one real checkpoint
first, or you will build something you cannot run.

---

## 3. What already exists — reuse it, do not rewrite it

### `src/spfilm/visualization.py`

| symbol | line | what it does |
|---|---|---|
| `_boundary(mask)` | 18 | 4-neighbour erosion → 1px contour. Correct, reuse it |
| `_overlay(image, masks)` | 28 | paints disc contour green `(0.1,1.0,0.2)`, cup blue `(0.1,0.5,1.0)` |
| `save_prediction_gallery(...)` | 118 | the existing 4-panel gallery |

`save_prediction_gallery` already draws image / target contours / prediction
contours / error map (red FP, cyan FN). **Its two limitations are exactly the two
things you are asked to fix:**

1. It renders the **letterboxed 512×512 network input**, not the original image.
2. It picks rows with `np.linspace` over the dataset — evenly spaced indices, not
   instances chosen for any reason.

It also needs a live `model` object, so it only runs inside training. The new
script is standalone: checkpoint + manifest + selection → PNGs.

Every real run already contains its output as `test_predictions.png` and
`test_<domain>_predictions.png` (~4.5 MB each). Look at one before you start —
it shows you the panel layout that already works, and makes the two gaps obvious.

---

## 4. The technical crux: inverting the letterbox

`_resize_and_pad` (`src/spfilm/data.py:1183`) does aspect-preserving resize onto a
black square canvas:

```python
scale   = size / max(width, height)
resized = (max(1, round(width * scale)), max(1, round(height * scale)))
offset  = ((size - resized[0]) // 2, (size - resized[1]) // 2)
```

To map a 512×512 prediction back onto the original image:

```python
# crop away the black padding, then resize to native
pred_crop = pred[oy : oy + resized_h, ox : ox + resized_w]
pred_native = np.array(
    Image.fromarray(pred_crop.astype(np.uint8) * 255)
         .resize((native_w, native_h), Image.Resampling.NEAREST)
) > 127
```

You do not need to recompute the geometry — the dataset already hands it to you.
`FundusSegmentationDataset.__getitem__` returns metadata (`data.py:1288`) carrying
`image_path`, `native_width`, `native_height` and `letterbox_scale`. Recompute
`resized`/`offset` from those with the formula above so the two stay consistent.

**Use NEAREST, never BILINEAR**, for masks — bilinear on a binary mask produces
fractional values and a contour that drifts by a pixel or two.

Two sanity checks worth writing as you go:

- **Round trip.** Letterbox a ground-truth mask, invert it, compare against the
  original mask. Should agree to within NEAREST resampling — a few boundary pixels,
  not a systematic offset. If it is offset, your `offset` has x/y swapped (PIL is
  `(x, y)`, numpy is `[y, x]` — this is the easiest bug to write here).
- **Dice recomputed** from the inverted prediction against the native mask should
  land near the value in the per-image CSV. It will not match exactly — the CSV is
  computed in the 512 grid — but a large gap means the inversion is wrong.

---

## 5. Choosing instances

Every run has per-image metrics. Cross-domain ones are
`test_<target_domain>_per_image_metrics.csv`; the source domain's own test set is
plain `test_per_image_metrics.csv`.

```
image_id,structure,dice,iou,hd95,acc,tp,fp,fn,tn
drishtiGS_001,disc,0.4235,0.2686,207.79,0.9646,3406,2331,6942,249465
drishtiGS_001,cup,0.2620,0.1508,45.80,0.9781,1020,0,5745,255379
```

Two rows per image (`disc`, `cup`). Support at least:

- `--worst N` / `--best N` / `--median N` by Dice on a chosen structure
- `--image-ids a,b,c` for naming specific instances directly
- a fixed `--seed` for any random sampling, so figures are reproducible

**Cells worth visualising, tied to the argument in §1:**

| cell | why |
|---|---|
| `refuge_zeiss → drishti_gs` | worst cell in the matrix (0.727 disc / 0.475 cup), and the pair that is photometrically almost identical (2 grey levels). If failure here looks geometric, that is the strongest single image-level support for the whole conclusion |
| `refuge_canon_val → refuge_zeiss` vs `refuge_zeiss → refuge_canon_val` | same pair, both directions, 0.891 vs 0.733 disc. A symmetric photometric story cannot explain that; show what differs |
| `rim_one_dl` held out (LODO) | the 0.074 collapse. Expect predictions at entirely the wrong scale |
| a good cell, e.g. `refuge_canon_val → drishti_gs` | a control. Failure figures alone are not evidence |

---

## 6. Gotchas that will cost you time

**`base_channels` defaults wrong.** `PlainUNet.__init__` (`model.py:56`) defaults
to `32`, but every Stage 3 run used **16** (`resolved_config.json`). Construct the
model from the run's own config, not the class default — otherwise
`load_state_dict` fails with a shape mismatch that reads like a corrupted file.

**Checkpoints are wrapped, not bare state dicts.**

```python
ckpt = torch.load(path, map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])   # keys: model_state_dict,
                                                  # optimizer_state_dict, epoch,
                                                  # validation_metrics, ...
```

**Manifest paths are CREATE paths.** `split_manifest.csv` carries absolute
`/cephfs/volumes/hpc_data_prj/.../datasets/REFUGE/...` paths. Local data is at
`datasets/{DRISHTI-GS,REFUGE,RIM-ONE-DL_masks,RIM-ONE_DL_images}` under the repo
root. Take a `--data-root` and remap everything up to and including `/datasets/`.
Fail loudly on a missing file — a silent skip gives you a figure of whatever
happened to load.

**`image_size` is 512 and `threshold` is 0.5** in every Stage 3 config. Read both
from `resolved_config.json` rather than hardcoding, so the figure cannot silently
disagree with the metrics it is placed next to.

**Do not letterbox the original image to draw on it.** The whole point is native
resolution. Load it with PIL at full size and draw the inverted mask on that.

**RIM-ONE-DL HD95 is in native pixels; every other domain is in grid pixels.**
Irrelevant to drawing, but do not put both in one caption as if comparable.

**Augmentation must be off.** `FundusSegmentationDataset(..., augment=False)`.
The existing gallery gets this right; it is easy to lose.

---

## 7. Environment

```bash
cd code/spfilm
.spfilm/bin/python <script>.py ...
```

`.spfilm` is the project venv (torch 2.13.0, numpy, PIL, matplotlib). The bare
`python3` on this Mac has **no numpy** — if you see `ModuleNotFoundError: numpy`
you are using the wrong interpreter. `uv run --with numpy` also works for
throwaway analysis but use the venv for anything touching torch.

Write outputs under `artifacts/` like everything else in the repo, and take an
explicit `--output-dir` rather than writing to the process working directory.
`matplotlib.use("Agg")` is already set at the top of `visualization.py`; keep it,
and never call `plt.show()` before `savefig` (it can blank the figure).

---

## 8. Suggested shape

A top-level CLI script next to `analyze_domain_shift.py`, e.g.
`visualize_predictions.py`, that:

1. takes `--run-dir` (a directory under `artifacts/runs/`), reads
   `resolved_config.json` and `split_manifest.csv` from it;
2. takes `--checkpoint` (defaulting to `<run-dir>/best_model.pt`) and errors with a
   clear message naming the CREATE path if it is absent — see §2, this will happen;
3. takes `--target-domain` to choose which per-image CSV drives selection;
4. selects instances per §5;
5. runs inference at 512, inverts the letterbox per §4;
6. writes one PNG per instance (not a contact sheet — these are for looking at
   closely) plus an `index.csv` recording image_id, cell, and the Dice from the
   per-image CSV, so a figure can always be traced back to its number.

Reuse `_boundary` and `_overlay`; add a native-resolution path beside the existing
letterboxed one rather than changing `save_prediction_gallery`, which training
still calls.

---

## 9. Related

- `run_reports/s3_domain_shift_brief.md` — the one-page conclusion this serves
- `run_reports/s3_1_3_intensity_shift_analysis.md`, `s3_3_1_intensity_shift_analysis.md` — full analyses
- `HANDOFF_histograms.md` — the earlier handoff, same house style
- Memory: `domain-shift-is-geometric-not-photometric`, `pooling-domains-is-a-coin-flip`
