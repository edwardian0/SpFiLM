# Session handoff: complete Spatial FiLM (SpFiLM) and run it on the two existing protocols

**Written 2026-09-28.** Task: finish the SpFiLM arm and run it through the **same two
protocols, splits, seeds and runners** that the plain U-Net and Global FiLM arms have
already completed, so SpFiLM is compared with both on identical test images. SpFiLM is
Step 5 of `Edward_Project_Brief.pdf` ("drop in the layer, head-to-head against Global
FiLM"); Step 6 is the same comparison at five seeds per held-out domain.

All paths are relative to `code/spfilm/` (the git repo, `origin` =
`github.com/edwardian0/SpFiLM`). Python is `.spfilm/bin/python`; tests are
`.spfilm/bin/python -m unittest discover tests` (**482 pass** as of writing). HEAD is
`12ad85a`; the working tree holds Edward's uncommitted `src/spfilm/film/spfilm.py`
(the layer, see §3) and the untracked Step 5 reports `run_reports/stage5_lodo{.md,_cells.csv}`.
Ask Edward whether to commit the layer as its own commit before building on it.

`PROMPT_step5_spatial_film.md` is an earlier *guide-mode* prompt (Edward writing, a
session reviewing). This handoff supersedes it for implementation. Its method section
(§2) and gotchas (§6) are still correct and worth reading. These parts are stale: it says
`spfilm.py` is empty, gives 449 tests, and routes SpFiLM LODO through the Step 4 runner.
LODO now has its own Step 5 runner (§4 here).

---

## 1. Terminology: the conditioning signal

What earlier code, logs and reports call the domain *code* is the **conditioning
signal** `s`: a frozen one-hot vector (64-d, `DomainOneHot`) indexing one of the
fold's training domains. Use "conditioning signal" in all **new prose, docstrings, log
lines and report text**.

**Do not rename existing identifiers or data keys**: `domain_index`, `DomainVocabulary`,
`fixed_code_sweep`, `best_fixed_code`, `domain_decision`, `used_minus_worst_fixed_code_dice`
and similar. The 40 completed runs on CREATE store their results under those keys, and the
aggregators read them. Renaming would orphan every existing result.

---

## 2. The two protocols SpFiLM must run (already run for plain + Global FiLM)

Both use the same locked budgeted partitions (`splits/single_source/single_source_manifest.json`,
sha `af4fa639…`): per domain **40 train / 10 val / 50 test**, three active domains
(`drishti_gs`, `refuge_canon_val`, `refuge_zeiss`). RIM-ONE-DL stays under `domains` in
every config because the locked manifests cover it and are revalidated from that block,
but it never enters a fold. A domain's 50 test images are identical in every protocol and
arm, which is what makes per-image paired tests valid.

### Protocol A — train on all three, test on each (runner `run_stage4_all_domains.py`)

- **One model per seed**, trained on all three domains pooled: 3 × 40 = **120 train**,
  3 × 10 = **30 val** (checkpoint selection).
- **Training uses each image's true conditioning signal.** Yes, SpFiLM gets exactly what
  Global FiLM got. The engine does this for every conditioned arm (`OracleCondition` for
  train and val whenever `arm != "plain"`); nothing to build.
- **Scored on each domain's own 50 test images separately**: three results per model
  (Drishti, Canon, Zeiss). No test image was seen in training, but no domain is held
  out. **Each test domain is scored with its own true conditioning signal**: config
  `film.test_conditioning: "oracle"`, which the runner enforces for any conditioned
  arm (`_require_arm_policy`).
- **Wrong-signal check.** Each domain's test set is also scored under every *other*
  training domain's signal. The drop is the wrong-signal penalty, which shows the
  conditioning layers are actually used. For Global FiLM the correct signal was best in
  30 of 30 seed-by-domain cells. **Yes, apply it to SpFiLM for a fair comparison.** The
  engine already runs this sweep for any conditioned arm on every named test set
  (`engine._conditioning_report`, per-test-set keys `fixed_code_sweep`,
  `used_minus_worst_fixed_code_dice`), and `aggregate_stage4_all_domains.py` turns it into
  the penalty table. SpFiLM gets it for free; the only work is letting the report show
  it for two conditioned arms side by side (§5, piece E).
- Grid: **5 seeds** (42–46) → 5 SpFiLM jobs. Existing arms to compare against (all
  COMPLETED on CREATE, report `run_reports/s4_all_domains_report.md`):
  `stage4_all_domains_fixed_budget_plain_unet_3dom`,
  `stage4_all_domains_fixed_budget_global_film_3dom`.

### Protocol B — train on two, test on the left-out third (runner `run_stage5_lodo.py`)

- Train on the two other domains: 2 × 40 = **80 train**, 2 × 10 = **20 val**. Test only
  on the held-out domain's **50** images. The training signal is each image's true one.
- The held-out domain has no signal the model was trained with. It gets **one signal for
  the whole domain**: the training domain whose colour centroid (FOV RGB mean/std) is
  nearest to the mean of the held-out domain's *unlabelled* reference sample (its
  budgeted train partition, never trained on, disjoint from the test images). This is
  `film.test_conditioning: "nearest_domain"`, the rule agreed with Pushpendra on
  2026-09-12; the runner refuses `oracle`. SpFiLM must use the same rule. The engine also
  scores the held-out set under each training signal (the sweep), so "signal used vs best
  available" is reported.
- Grid: **3 held-out domains × 5 seeds = 15 SpFiLM jobs**. Existing arms (all 30
  COMPLETED 2026-09-26; job ids in memory note `step5-lodo-runner`):
  `stage5_lodo_fixed_budget_plain_unet_3dom`, `stage5_lodo_fixed_budget_global_film_3dom`.

---

## 3. What the results so far say (read before interpreting SpFiLM)

**Protocol A (train on all), 5 + 5 seeds**, from `run_reports/s4_all_domains_report.md`,
Δ = Global FiLM − plain:

| Test domain | disc Δ | cup Δ | Wrong-signal penalty (disc / cup) |
|---|---:|---:|---:|
| Drishti | −0.006 (sig.) | −0.014 (sig.) | +0.10 / +0.26 |
| Canon | −0.001 | +0.005 | +0.05 / +0.11 |
| Zeiss | −0.005 (sig.) | −0.005 | +0.07 / +0.21 |

Global FiLM never beats plain, but it clearly *uses* the signal.

**Protocol B (LODO), 15 + 15 runs**, from `run_reports/stage5_lodo.md` (findings
section not yet written):

| Held out | Plain disc / cup | Global FiLM disc / cup | Δ disc / cup | Signal given | Best signal in sweep |
|---|---|---|---|---|---|
| Drishti | 0.866 / 0.622 | 0.811 / 0.571 | −0.055* / −0.051* | Zeiss | Canon (5/5) |
| Canon | 0.879 / 0.678 | 0.731 / 0.676 | −0.148* / −0.002 | Zeiss | Drishti |
| Zeiss | 0.863 / 0.632 | 0.854 / 0.639 | −0.010* / +0.008 | Drishti | Canon (5/5) |

`*` = significant after Holm.

**Global FiLM is worse than plain under LODO.** For the disc, the nearest-source rule
handed every held-out domain the *worse* of its two available signals in every seed
("used − best" equals the full sweep spread). For the cup, Canon was mixed.
This was predicted by the train-on-all sweep and by `domain-shift-is-geometric-not-photometric`:
the colour descriptor pairs Drishti with Zeiss, which are geometrically the least alike.
SpFiLM will be given signals by the same rule, which is fair to both arms. But the
LODO comparison will be partly a comparison of how badly each arm degrades under a
bad signal. Read each run's sweep (best-signal Dice) next to the Dice difference.

**Why SpFiLM is worth testing here.** The domain shift is geometric. The disc diameter as
a fraction of the frame is 1.25× larger in Drishti than in REFUGE (`run_reports/`, memory
note `domain-shift-is-geometric-not-photometric`). A per-channel affine cannot say *where*
to modulate; SpFiLM's spatial term can.

**Parameter counts at full size** (base 16, 5 conditioned levels):

| Model | Parameters |
|---|---:|
| Plain U-Net | 1,944,066 |
| Global FiLM | 2,611,170 |
| SpFiLM K=2 | 3,128,618 |
| SpFiLM K=8 | 4,667,042 |
| SpFiLM K=16 | 6,718,274 |

Smoke runs use base 8, so their counts are smaller (plain 487,298). The brief's own
logic: if SpFiLM only ties Global FiLM, the spatial idea is not what helped, and any gain
over plain could be extra parameters. Report parameter counts in every table.

---

## 4. What exists — the seams SpFiLM plugs into

| Piece | Where | State |
|---|---|---|
| Layer `SpatialBasis`, `SpatialFiLM` | `src/spfilm/film/spfilm.py` | **Written by Edward, uncommitted, untested.** Verified this session: K=0 with `GlobalFiLM.generator` weights reproduces `GlobalFiLM` exactly (`allclose`, same 90,656 params at C=16). K=8 fields vary over pixels; dtype preserved under autocast. Keep it; add tests (piece A). |
| Global FiLM layer, one-hot | `src/spfilm/film/global_film.py` (`GlobalFiLM`, `DomainOneHot`) | Done; must stay byte-identical. |
| Network | `src/spfilm/model.py` | `ConditionedUNet` done. `class SpatialFiLMUNet(): pass` is a **stub**. `ARMS = ("plain", "global_film")`. `build_model` has a stray `if arm == "spatial_film": return SpatialFiLMUNet` that returns the class. |
| Engine | `src/spfilm/engine.py` | Branches only on `arm != "plain"`. `_forward(model, images, condition)` calls `model(images, condition.indices)`: the network gets the image, so the basis maps need **no engine change**. Needs config plumbing only (piece C). |
| Signal providers | `src/spfilm/film/conditioning.py` | `OracleCondition`, `DomainCondition` (per-domain nearest), `NearestCondition`, `FixedCondition` (sweep). Reused unchanged. `fov_descriptor` holds the FOV luminance rule (`global_histograms.FOV_LUMINANCE_THRESHOLD = 0.10`, `LUMA_WEIGHTS`). |
| Config parser | `src/spfilm/stage3_single_source.py` (`Stage3SingleSourceConfig`, `_parse_film_block`, `FILM_BLOCK_DEFAULTS`) | One parser serves every runner. `from_json` checks `arm in ARMS`, so adding the arm to `ARMS` is what makes configs load. |
| Protocol A | `run_stage4_all_domains.py`, `configs/stage4_all_domains_global_film_3dom{,_create}.json`, `submit_stage4_all_domains_film.sh` (job `allf_s4`, 3 h), `aggregate_stage4_all_domains.py` | Runner needs nothing. |
| Protocol B | `run_stage5_lodo.py`, `configs/stage5_lodo_global_film_3dom{,_create}.json`, `submit_stage5_lodo_global_film.sh` (job `gfilm_s5`, 2 h), `aggregate_stage5_lodo.py` (thin wrapper over `aggregate_stage4_film.py`) | Runner needs nothing. |
| Tests to mirror | `tests/test_global_film.py` (`GlobalFiLMTests`, `ConditionedUNetTests`, `Stage4ConfigTests`, `FingerprintTests`), `tests/test_stage4_all_domains.py`, `tests/test_stage5_lodo.py` | |

Reference implementation (the "how"): Pushpendra's
<https://github.com/p-singh-kcl/spatial_film_parcellation>, `models/spatial_film.py`
(`SpatialFiLM3d`, `UNetSpatialFiLMOneHot3D.forward_encoder`) and
`configs/spatial_film_onehot_k{1,2,4,8,16}.yaml`. Edward's layer already mirrors
`SpatialFiLM3d` in 2D.

---

## 5. The work, in order

Match the surrounding style: `from __future__ import annotations`, type hints, comments
that say *why*, `ValueError` for contract violations in layers, `unittest` with tests
named as sentences. **Every arm must differ from its comparator in exactly one thing.**
Plain and Global FiLM code paths, configs and fingerprints stay byte-identical.

### A. Tests for the existing layer — new `tests/test_spatial_film.py`

Classes `SpatialBasisTests` and `SpatialFiLMTests`. Invariants:

- **Basis maps:** output is `(N, K, H', W')` and lies in `[-1, 1]`. The maps differ for
  different images, do not depend on the conditioning signal, and `rank < 1` is refused.
- **K=0 identity:** copy `GlobalFiLM(C).generator.state_dict()` into
  `SpatialFiLM(C, rank=0).generator` and assert `allclose(atol=1e-6)` on random
  inputs. With K=0, `basis_gamma` and `basis_beta` are `None`, there are no basis
  state-dict keys, and the parameter count equals `GlobalFiLM`'s.
- **Zero generator output ⇒ identity** (mirror `test_gamma_and_beta_zero_is_the_identity`).
- **Spatial variation:** for K≥1 with a nonzero `A`, γ varies over pixels. This is the
  contrast with the global test `…uniform_over_pixels`.
- **Clamp after assembly:** push the bars and coefficients so the assembled field would
  exceed `clamp`, and check the output field is inside `[-clamp, clamp]`.
- **Autocast:** output dtype preserved and finite with large inputs.
- **FOV gating:** pixels with mask 0 come out exactly equal to the input features. With
  gating off, the mask is ignored.
- **Parameter count** equals the closed form: generator
  `(E·H + H) + (H·H + H) + (H·2C(1+K) + 2C(1+K))`, plus, for K≥1,
  `2 × [9·in·16 + 2·16 + 9·16·K + 2·K]`.
- **Coefficient layout** is `[γ̄ | A | β̄ | B]`: perturb one slice of the last `Linear`
  bias and check only the intended tensor moves.
- **Shape and batch errors:** a features/image spatial mismatch raises; so does a batch
  mismatch.

Only fix the layer where a test exposes a real defect. Its docstrings say "domain code";
change those to "conditioning signal".

### B. Network — replace the stub in `src/spfilm/model.py`

```python
class SpatialFiLMUNet(nn.Module):
    """PlainUNet with SpFiLM after each encoder block; its K=0 case is ConditionedUNet."""
    ENCODER_LEVELS = 5
    def __init__(self, num_domains, in_channels=3, out_channels=2, base_channels=32,
                 film_levels=ENCODER_LEVELS, embedding_dim=64, hidden_dim=256, clamp=5.0,
                 rank=DEFAULT_RANK, fov_gating=False) -> None
    def forward(self, inputs: torch.Tensor, domain_index: torch.Tensor) -> torch.Tensor
```

- Same skeleton and validation as `ConditionedUNet`: a real `PlainUNet` as `backbone`
  (state-dict keys under `backbone.` equal the plain keys), `DomainOneHot`, and
  `num_domains` in `[1, embedding_dim]`. `domain_index >= num_domains` raises
  `ValueError`. `film_levels` counts from the shallowest, and 5 = four encoder blocks plus
  the bottleneck.
- Level 0 feeds `inputs` itself to the layer. Deeper levels feed
  `F.interpolate(inputs, size=features.shape[-2:], mode="bilinear", align_corners=False)`.
  **Resample the image, never the features.**
- FOV gating (default off): compute the mask once from `inputs` with a new helper
  `fov_mask(images) -> (N, 1, H, W)` in `film/conditioning.py`, next to
  `fov_descriptor`, using the existing threshold and weights; do not re-derive them. Pass
  the mask to each layer, which nearest-resamples it.
- `ARMS = ("plain", "global_film", "spatial_film")`. `build_model(...)` gains
  `rank: int = 0, fov_gating: bool = False` and returns a `SpatialFiLMUNet` **instance**
  for `"spatial_film"`, requiring `num_domains` like `global_film`. Fix the stray
  `return SpatialFiLMUNet`.

Tests (`SpatialFiLMUNetTests`, mirroring `ConditionedUNetTests`):
- Shape preserved and 2 output channels on 3×32×32 CPU inputs.
- Backbone keys equal the plain keys, and the parameter count is backbone + Σ films.
- `film_levels=2` conditions only two levels.
- With every `generator` zeroed and K≥1, the output equals `PlainUNet` with the same
  backbone weights exactly.
- With `rank=0` and generator weights copied from a `ConditionedUNet`, the output equals
  that `ConditionedUNet`.
- An untrained signal is refused.
- `build_model("spatial_film", 8, num_domains=2, rank=4)` is a `SpatialFiLMUNet`; without
  `num_domains` it raises `ValueError`.
- `tests/test_global_film.py::test_build_model_maps_arms` still asserts `"spatial"` is
  unknown; leave it passing.

### C. Config plumbing (four small places)

1. **`engine.Stage2Config`:** add `film_rank: int = 0` and `film_fov_gating: bool = False`
   next to the other `film_*` fields. `0` means "no spatial term", which is literally true
   of every non-spatial arm. Also:
   - Append both to `FILM_CONFIG_FIELDS`.
   - Pass `rank=config.film_rank, fov_gating=config.film_fov_gating` into the
     `build_model(...)` call (~line 1307).
   - Add `"rank"` and `"fov_gating"` to the `"film"` block in `_conditioning_report`
     (~line 1097).
   - Add `rank=` to the `conditioning | arm=… | film_levels=…` log line.
2. **Resume fingerprint (not optional).** `_resume_fingerprint` hashes `asdict(config)`,
   and plain already pops every `FILM_CONFIG_FIELDS` entry. Add
   `SPATIAL_FILM_CONFIG_FIELDS = ("film_rank", "film_fov_gating")` and pop them when
   `config.arm == "global_film"`. Every existing Global FiLM fingerprint then stays
   byte-identical, so a relaunched Global FiLM run still resumes. Tests:
   - The plain fingerprint is unchanged (existing test).
   - The Global FiLM fingerprint is unchanged: build it from a config without the fields
     and with the defaults; the two must be equal.
   - The spatial fingerprint changes with `rank`.
3. **`stage3_single_source.Stage3SingleSourceConfig`:**
   - Add fields `film_rank: int = 0` and `film_fov_gating: bool = False`, and have
     `training_config` forward them.
   - In `_parse_film_block`, allow `rank` and `fov_gating`.
   - `global_film`: refuse `rank` / `fov_gating` ("the global arm has no spatial term").
   - `spatial_film`: **require** `rank` as a positive int. Refuse 0 with "use global_film";
     that equivalence lives in the tests, not in a run. `fov_gating` is an optional bool.
   - The plain arm must still carry no film block.
4. **Runners and engine flow:** nothing else. Both runners key on `arm != "plain"`.

Tests (`SpatialFiLMConfigTests`):
- `rank` is parsed and forwarded to `Stage2Config`.
- A missing `rank` is refused for the spatial arm.
- `rank` is refused for the global arm.
- `rank: 0` is refused.
- `fov_gating` must be a bool.
- The fingerprint tests from item 2.

### D. Configs and submit scripts

Derive each new config from its Global FiLM twin with a script that copies the file and
changes only the listed keys (as the Step 5 configs were derived; see the generator
pattern in `configs/stage5_lodo_*`). Keep `domains` complete (all four) and keep each
`_create` twin identical to its local config except the four `data_root` values.

| Protocol | New config | From | Changes |
|---|---|---|---|
| A | `configs/stage4_all_domains_spatial_film_k8_3dom{,_create}.json` | `stage4_all_domains_global_film_3dom*` | `arm: "spatial_film"`, `film.rank: 8`, `film.fov_gating: false` (stated explicitly), `experiment_name: "stage4_all_domains_fixed_budget_spatial_film_k8_3dom"`, `output_dir`, `protocol.policy`, `protocol.paired_arm` = the Global FiLM train-on-all arm, `protocol.secondary_comparison` = the plain arm. `test_conditioning` stays `"oracle"`. |
| B | `configs/stage5_lodo_spatial_film_k8_3dom{,_create}.json` | `stage5_lodo_global_film_3dom*` | Same set of changes. `experiment_name: "stage5_lodo_fixed_budget_spatial_film_k8_3dom"`; `paired_arm` = `stage5_lodo_fixed_budget_global_film_3dom`; secondary = the plain Step 5 arm. `test_conditioning` stays `"nearest_domain"`. |

The file prefix follows the **runner**: train-on-all configs are `stage4_all_domains_*`
because `run_stage4_all_domains.py` and the arms they pair with carry that name. The
`paired_arm` is Global FiLM because the brief names it SpFiLM's honest comparison; plain
is the secondary comparison.

Tests: copy `test_configs_differ_only_in_the_arm` /
`test_film_configs_differ_from_plain_only_in_the_conditioning`. Each new config must equal
its Global FiLM twin once the listed keys (plus `film.rank` / `film.fov_gating`) are
removed. Also check local vs `_create` differ only in `data_root`, and each runner's
`_require_arm_policy` accepts the new config.

Submit scripts: copy the current (hardened) Global FiLM scripts and change only the job
name, log names, config, output prefix and header comment.
- `submit_stage4_all_domains_spatial_film.sh`: job `allsf_s4`, output dir
  `artifacts/runs/allsf_s4_seed_<seed>_<job>`, args `<seed> [--smoke]`, 3 h.
- `submit_stage5_lodo_spatial_film.sh`: job `sfilm_s5`, output dir
  `artifacts/runs/sfilm_s5_<held-out>_seed_<seed>_<job>`, args
  `<held-out-domain> <seed> [--smoke]`, 2 h.

Keep `#!/bin/bash -l`, `--export=NONE`, and the exclude list (it includes the ECC-faulty
`comp223`). SpFiLM costs more per step (two extra 3×3 convs on the resampled image and a
`bmm` per level, plus a 16-channel field at 512² at level 0). The seed-42 real run's
`Elapsed` is the number to check before the rest. Smoke timing says nothing: smokes run
at 128 px and base 8.

### E. Aggregators — two conditioned arms side by side

Both aggregators already take `--plain-arm` (the comparator) and `--film-arm` (the
reference; a positive Δ reads "reference helped"). They run mechanically for SpFiLM vs
Global FiLM, but they hard-code "Plain" / "FiLM" labels, report titles that say
"Global FiLM against the plain U-Net", and conditioning diagnostics for the *second* arm
only. For SpFiLM vs Global FiLM both arms are conditioned, and both sets of diagnostics
are the point.

- **Labels from the runs, not the flags.** Map each arm's recorded `conditioning_arm`:
  `plain` → "Plain U-Net", `global_film` → "Global FiLM", `spatial_film` → "SpFiLM".
  Use the labels for column headers, `Δ (B − A)` and the title.
  - `AllDomainsRun` already has `conditioning_arm`.
  - For `FixedRun` (used by `aggregate_stage4_film.py` / `aggregate_stage5_lodo.py`), add
    a `conditioning_arm: str = "plain"` field at the end of the dataclass, filled from the
    metadata block in `build_fixed_run`. Both the Step 4 `fixed_lodo` and the Step 5
    `stage5_lodo` blocks record it.
- **Diagnostics for every conditioned arm.** Build the train-on-all wrong-signal penalty
  cells and the LODO conditioning cells for *each* arm with a conditioning block, and add
  an Arm column to those tables.
- New or changed report text says "conditioning signal". Data keys stay as they are (§1).
- **Rename the CLI flags to `--arm-a` and `--arm-b`**, keeping `--plain-arm` /
  `--film-arm` as aliases so every earlier command still works. The defaults stay plain
  vs Global FiLM.
- Existing tests that pin old strings (for example `"Plain arm: `…`. FiLM arm: `…`."`
  and the Step 4/5 titles in `tests/test_stage5_lodo.py`) are updated to the new labels.
  Do **not** regenerate the committed Step 4 report or overwrite hand-written findings.

Four comparisons result:

| Protocol | `--arm-a` | `--arm-b` |
|---|---|---|
| A | plain | SpFiLM |
| A | Global FiLM | SpFiLM |
| B | plain | SpFiLM |
| B | Global FiLM | SpFiLM |

Each is Holm-adjusted within its own table (6 tests: 3 domains × disc/cup), as today.

### F. Verify locally, in this order

1. Full test suite green, including every untouched Step 3/4/5 suite. Recompute a Global
   FiLM fingerprint before and after piece C; it must be identical.
2. `.spfilm/bin/python run_stage4_all_domains.py --config configs/stage4_all_domains_spatial_film_k8_3dom.json check --skip-mask-audit`
3. `.spfilm/bin/python run_stage5_lodo.py --config configs/stage5_lodo_spatial_film_k8_3dom.json check --skip-mask-audit`.
   Expect 80/20/50 per fold, with the conditioning reference = 40.
4. CPU smokes (write to `artifacts/runs/<name>` and delete the `_smoke` dirs afterwards):
   - `run_stage4_all_domains.py --config configs/stage4_all_domains_spatial_film_k8_3dom.json run --seed 42 --smoke --device cpu --out-dir artifacts/runs/spfilm_all_smoke`.
     Expect `arm=spatial_film`, three signals, `conditioning.film.rank == 8`, a three-entry
     sweep per test domain, and `parameter_count` = plain + Σ films at base 8.
   - `run_stage5_lodo.py --config configs/stage5_lodo_spatial_film_k8_3dom.json run --held-out-domain drishti_gs --seed 42 --smoke --device cpu --out-dir artifacts/runs/spfilm_lodo_smoke`.
     Expect two signals, the held-out signal decided from reference images, and a
     two-entry sweep.
5. Simulate the submit scripts end to end, as was done for Step 5 on 2026-09-24:
   - Make a sandbox copy of each script with only three lines changed: `CODE_ROOT` → the
     local repo, the log dir → scratchpad, and `_create.json` → the local config.
   - Stub `module`, `conda`, `nvidia-smi` and the CUDA probe (a `python` shim).
   - **Run the harness under `bash`, not the tool's zsh**, otherwise the exported shell
     functions never reach the job.
   - Pass `--smoke --device cpu`. For each job check that its own `split_manifest.csv`
     shows train/val only from the training domains and test only from the tested one,
     and that the sweep differs across signals.
6. Run the new aggregators on synthetic runs through `main`, including one seed per arm
   ("(1 seed)" cells) and a Global FiLM vs SpFiLM pair with both diagnostic tables.

### G. On CREATE (Edward runs these; the session prepares commands)

1. Commit and push. On CREATE (`~/edward/spfilm`), run `git status`; it should show only
   untracked run outputs. Then `git pull`. If git refuses because untracked files would
   be overwritten, move exactly those files aside and pull again.
2. Activate the env on the login node, then run `check` for both new `_create` configs.
3. Smoke each script: `sbatch --time=0-00:20:00 <script> … --smoke`. Then seed 42 for
   real: 1 train-on-all job, and 3 LODO jobs (one per held-out domain). Read their
   `Elapsed` and the seed-42 reports:
   - `aggregate_stage4_all_domains.py --arm-a <global> --arm-b <spfilm> --expected-seeds 42`
   - `aggregate_stage5_lodo.py --arm-a <global> --arm-b <spfilm> --expected-seeds 42`
   Only then submit seeds 43–46 (4 + 12 jobs). Submit from `erc-hpc-login1/2`.
4. Pull reports down on the Mac. Keep the quotes, or zsh globs locally:
   `rsync -avz 'create:edward/spfilm/run_reports/<prefix>*' ~/Desktop/Projects/Research/SpFilm/code/spfilm/run_reports/`.
   Run folders without weights:
   `rsync -avz --exclude='*.pt' 'create:edward/spfilm/artifacts/runs/<prefix>_*' …/artifacts/runs/`.

---

## 6. Gotchas (each has bitten this project or the reference)

- **Autocast:** the whole modulation, *including the basis convs*, runs with autocast off
  in fp32 (Edward's layer does this). One half-precision overflow in a `bmm` turns the
  field into NaN.
- **Clamp the assembled field**, not the bars and coefficients separately; bilinear
  upsampling can push values past the bars' range. **No bias** on the basis convs
  (InstanceNorm follows each).
- **Smallest grid:** a 128-px smoke's bottleneck is 8×8, and the stride-2 basis conv
  makes it 4×4. InstanceNorm over K channels at 4×4 is fine; nothing smaller ever runs.
- **Letterbox border:** it is black and inside what the basis generators see, and
  InstanceNorm includes it. Only Drishti is non-square (memory note
  `only-drishti-is-non-square`), so this is a Drishti-specific effect, and one reason FOV
  gating is a worthwhile ablation (off in the first run).
- **Conditioned-arm RNG:** the extra modules change parameter-initialisation order, so
  SpFiLM seed 42 is not "Global FiLM seed 42 plus a spatial term". The arms are paired on
  test images, not on training trajectories.
- **`domains` stays complete** (four entries) in every config; participation is the
  protocol list.
- **CREATE:**
  - ssh hangs about 40 s then dies until MFA is refreshed at
    <https://portal.er.kcl.ac.uk/mfa/>.
  - `--requeue` is set, but preempted jobs die. Resubmit the cell; it restarts from
    epoch 1 in a new directory.
  - An "uncorrectable ECC error" is a broken GPU, not the code. Exclude the node. A
    command-line `--exclude` **replaces** the script's list, so repeat the whole list plus
    the node. The node that killed `gfilm_s5_37503695` is still unidentified.
  - Aggregators ignore smoke runs and dead run directories (no `test_metrics.json`).
- Pushpendra reads three-line messages: result, conclusion, next step.

---

## 7. Out of scope here; decisions for Edward and Pushpendra

- **Rank K:** 8 first (the reference's default). A {2, 16} sweep only if budget allows.
  If gains track K, suspect parameters rather than space.
- **FOV gating ablation:** after the K=8 grid, on Drishti first.
- **The held-out selection rule:** it picked the worse signal in every LODO disc cell for
  Global FiLM. Changing it, for example to a geometric descriptor (disc-to-frame scale), is
  Pushpendra's call. Do not change it unilaterally; it would invalidate the pairing with
  the completed Global FiLM runs.
- **Step 8 figures** (visualising learned γ/β fields, and region-wise Dice) come after
  the grid.
- **The Step 6 main table** (brief §5): Dice and IoU for disc and cup, HD95, 95% CIs,
  paired tests, per held-out domain. The aggregators print mean ± seed SD. The CIs exist
  in `SeedInterval.low/high`, but a table showing them is not yet rendered.

## 8. Definition of done

- `SpatialFiLM`/`SpatialFiLMUNet` are tested. The K=0 identity holds to 1e-6 at both
  layer and network level. Plain and Global FiLM paths and fingerprints are unchanged,
  and every earlier test passes.
- Both new configs pass `check`, both CPU smokes run, and both submit scripts pass the
  local simulation.
- The aggregators show SpFiLM against Global FiLM and against plain, with diagnostics for
  both conditioned arms.
- On CREATE: 5 train-on-all and 15 LODO SpFiLM runs COMPLETED; reports pulled to
  `run_reports/`; findings sections written by hand; a three-line summary for Pushpendra.
