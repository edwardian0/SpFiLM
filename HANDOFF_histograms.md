# Session handoff: pixel-intensity histograms and the domain-shift analysis

**Written 2026-09-06.** Covers the finished analysis in `analyze_domain_shift.py`
and the in-progress `src/spfilm/global_histograms.py`.

---

## 1. Why this task exists

Pushpendra asked for "normalised histogram distributions across the datasets to
show the domain shift — the distribution shift of the pixel intensities — and see
what I find", plus a follow-up: work out why holding out REFUGE gives better
Stage 3 results than holding out the others.

Both questions are answered. The answer to the second one is **not** the answer
anybody expected, and it is the single most important thing to carry forward.

---

## 2. The finding (carry this forward)

**Pixel-intensity shift does not explain the Stage 3 results. It runs backwards.**

Wasserstein-1 distance from each domain to the pooled other three (field-of-view
pixels, gray channel) against that domain's held-out disc Dice in the 3→1 arm:

| domain | W1 to rest | held-out disc Dice |
|---|---:|---:|
| `rim_one_dl` | 0.0377 — closest | 0.0537 — worst |
| `drishti_gs` | 0.0923 | 0.7589 |
| `refuge_canon_val` | 0.1168 | 0.8684 |
| `refuge_zeiss` | 0.1273 — furthest | 0.8834 — best |

REFUGE-Zeiss is photometrically the *most* distinct domain and scores best.
RIM-ONE-DL is the *least* distinct and collapses.

**What does explain it is structure scale.** Median disc area as a fraction of
the frame: `refuge_zeiss` 0.0166, `refuge_canon_val` 0.0163, `drishti_gs` 0.0291,
`rim_one_dl` 0.4079. RIM-ONE-DL's disc is 14–25× larger relative to the frame,
because it ships optic-nerve-head crops while the others are full posterior-pole
images. Its field-of-view pixel fraction is 0.972 against 0.72–0.79.

REFUGE scores best when held out because the two REFUGE subsets are near
geometric twins (0.0166 vs 0.0163 disc fraction; 0.1432 vs 0.1440 disc diameter
over longest edge), so holding one out always leaves its twin in the training
pool. Drishti has no twin and sits ~1.8× away, giving the intermediate score.

**The train-on-one runs independently confirmed this.** RIM-ONE-DL fails
*symmetrically*: as a target it scores 0.086 disc against 0.56–0.60 for the
others; as a source its row mean is 0.096 against 0.51–0.61. A photometric cause
could not produce that symmetry. Train on crops → fail on full frames; train on
full frames → fail on crops.

Practical consequence: intensity normalisation or colour augmentation would not
have moved the RIM-ONE-DL result at all. Treat it as a field-of-view/scale
mismatch when writing up, and when choosing what SpFiLM conditions on.

This is also stored in memory as `domain-shift-is-geometric-not-photometric`.

---

## 3. Done: `analyze_domain_shift.py`

Top-level CLI script, committed in `1aa9c6f`. Uses PIL, parallelised with
`ProcessPoolExecutor`. Sweeps all 1,386 images in ~2 minutes on the Mac.

```bash
.spfilm/bin/python analyze_domain_shift.py                 # full sweep
.spfilm/bin/python analyze_domain_shift.py --from-cache    # redraw in seconds
.spfilm/bin/python analyze_domain_shift.py --limit 8 --workers 4   # quick rehearsal
```

Flags: `--config`, `--output-dir`, `--working-size` (default 256),
`--limit`, `--workers`, `--from-cache`, `--bars` (default 64, must divide 256),
`--skip-geometry`.

### Outputs, all in `artifacts/domain_shift/`

| file | what it is |
|---|---|
| `intensity_histograms_bars_fov.png` | **the headline figure** — real bar histograms, 4 domains × 4 channels, retinal FOV only |
| `intensity_histograms_bars_all.png` | same but all pixels; shows RIM-ONE-DL has *no* spike at zero while the others do — the framing difference, visible directly |
| `intensity_histograms.png` | 4-domain overlay curves; better for judging *how far apart* domains sit, worse as a "histogram" |
| `structure_scale.png` | box plots of disc/cup area fraction and disc diameter — the figure that actually explains the results |
| `leave_one_out_distance.png` | each domain's W1 distance to the pooled other three |
| `densities.npz` | cached accumulated densities + geometry, so `--from-cache` skips rescanning |
| `domain_shift.json` | everything machine-readable, including method parameters |
| `domain_intensity_summary.csv`, `domain_pairwise_distances.csv`, `domain_leave_one_out_distances.csv`, `domain_structure_scale.csv` | tables |

### Design decisions worth knowing

- **Two pixel populations, never pooled.** `all` includes the black surround
  (the framing difference, and what the network's input tensor actually
  contains); `fov` masks it out at luminance > 0.10 (the photometric difference).
  A gap in `fov` means the cameras disagree about colour; a gap only in `all`
  means the images are cropped differently. Different remedies, so they are kept
  apart.
- **Densities, not counts.** Per-image counts are summed per domain, then
  divided by the domain's total pixels and the bin width. Each curve integrates
  to 1, so domains of very different size are comparable, and a domain is not
  skewed by a few unusually large frames.
- **256 bins internally, rebinned to 64 for the bar figures.** At 256 bins the
  bars merge into a solid block. Rebinning also averages the black-surround spike
  down enough that the `all` figure fits on a linear axis; the overlay figure
  uses a log y-axis for the same reason.
- **`working_size=256` via PIL `draft`.** Lets the JPEG decoder downscale while
  decoding, which is what makes a full-dataset sweep affordable. The intensity
  *distribution* is insensitive to this; the exact pixel grid is not needed.
- **Geometry is measured on native-resolution decoded masks**, and
  `disc_diameter_fraction` is divided by the image's **longest edge** — the edge
  the training pipeline letterboxes by — so it is the scale the network sees.

---

## 4. In progress: `src/spfilm/global_histograms.py`

Untracked, 41 lines, actively being edited. An OpenCV-based `GlobalHistogram`
dataclass that accumulates one grayscale histogram across a list of image paths.

### Current state — mid-edit, does not yet parse

The file was being actively typed while this handoff was written, so it ends in a
partial statement and `import spfilm.global_histograms` fails with a
`SyntaxError`. That is expected churn, not a defect to chase — finish the edit
first. The issues below are the durable ones that survive whatever the current
half-written line happens to be.

| issue | why it matters |
|---|---|
| `global_hist += hist.astype(np.int64)` in `build_histogram` refers to a **bare local**, not `self.global_hist` | `UnboundLocalError` the moment the file parses. Nothing is ever accumulated |
| `plt.show()` immediately before `plt.savefig(...)` | `show()` can clear the figure, so the saved PNG may come out blank. `show()` is also useless headless; the rest of the repo sets `matplotlib.use("Agg")` and saves to a path |
| `global_hist: np.array` | `np.array` is a function, not a type; should be `np.ndarray` |
| `if img is not None:` with no else | unreadable files are skipped in silence. A dataset that half-fails to decode produces a plausible-looking histogram over whatever loaded |
| `FundusRecord`, `FundusSegmentationDataset`, `glob`, `os` imported but unused | leftovers from the earlier shape of the file |
| `savefig` writes to the process working directory | everything else in the repo writes under `artifacts/`; take an output path instead |

Verified **not** a problem: `cv2.imread` accepts a `Path` fine on the installed
OpenCV 5.0.0, so no `str()` wrapping is needed.

### Dependency gap

`cv2` is installed in the local `.spfilm` venv (5.0.0) but is **not declared in
`pyproject.toml`**. Anything importing `spfilm.global_histograms` will fail on
CREATE unless opencv is added to the conda env. Since this file lives inside the
package (`src/spfilm/`), an unguarded `import cv2` at module top level makes the
whole package fragile. Either add the dependency, or keep OpenCV out of the
package and use PIL as the rest of the code does.

---

## 5. The open question: what is `global_histograms.py` for?

This is the decision to make first, before fixing the bugs above. Right now it
would be a **second, disagreeing implementation** of something
`analyze_domain_shift.py` already does:

| | `analyze_domain_shift.py` | `global_histograms.py` |
|---|---|---|
| library | PIL | OpenCV |
| normalisation | densities (comparable across domains) | raw counts |
| black surround | separated into `all` vs `fov` | included, will dominate |
| channels | gray + R/G/B | gray only |
| scope | per domain, four compared | one list of paths |
| output | files under `artifacts/` | `plt.show()` |

Two implementations with different conventions will give different-looking
answers for the same data, and the raw-count version's zero spike will swamp
everything else — that is exactly the problem the `fov` split and the log axis
were added to solve.

Three plausible intents, with what each implies:

1. **A reusable in-package API** (most likely). Then don't reimplement — move
   `accumulate_domain` / `DomainHistograms` out of the script into
   `src/spfilm/global_histograms.py`, and have `analyze_domain_shift.py` import
   them. One implementation, one set of conventions, and the script keeps its
   CLI. Keep PIL to avoid the OpenCV dependency.
2. **One "global" histogram per dataset**, rather than a four-way comparison —
   e.g. a single curve for a whole dataset in the write-up. That is a genuinely
   different figure and worth having, but it should still be a density and
   should still say whether the surround is included.
3. **A simpler figure for the thesis.** If so, the existing
   `intensity_histograms_bars_fov.png` may already be it, and this file can be
   dropped.

**Recommendation: (1).** It gets the reusable API without a second set of
conventions, and it is mostly a move rather than new code.

---

## 6. Immediate next steps

1. Decide the intent in §5. Everything else follows from it.
2. Fix the syntax error on line 41 so the module imports; then the
   `self.global_hist` bug on line 28.
3. If OpenCV stays, add `opencv-python` to `pyproject.toml` dependencies and to
   the CREATE conda env. If not, switch to PIL.
4. Replace `plt.show()` with a `savefig` to a caller-supplied path, and set the
   `Agg` backend, matching `visualization.py` and `analyze_domain_shift.py`.
5. There are no tests for either histogram module. If `global_histograms.py`
   becomes package API, it should get some — a synthetic image with a known
   intensity distribution is enough to pin the accumulation and the
   normalisation.

## 7. Repo state at handoff

Committed: `analyze_domain_shift.py`, all its outputs, and the memory entry.

Untracked and needing `git add` when convenient:

```
aggregate_stage3_1_3.py
tests/test_aggregate_stage3_1_3.py
src/spfilm/global_histograms.py
run_reports/s3_1_3_report.md
run_reports/s3_1_3_cells.csv
run_reports/s3_1_3_findings_no_rim_one.md
```

Test suite is green: **237 passed, 80 subtests**
(`.spfilm/bin/python -m pytest tests -q`). Neither histogram module is covered by
it.

One stray edit was reverted during this session: line 1 of
`analyze_domain_shift.py` had picked up leading whitespace on the shebang, which
compiles fine but stops the file being directly executable.
