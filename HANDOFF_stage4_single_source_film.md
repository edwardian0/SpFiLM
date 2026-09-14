# Session handoff: Global FiLM in the train-on-one, test-on-two setting

**Status 2026-09-12 (later the same day): built and verified locally; not launched.**
Everything in Section 4 exists and passes (`tests/test_stage4_single_source_film.py`,
36 tests; full suite 420 green). `check` and the CPU smoke behave as Section 6
specifies. One deviation from the spec below: the film block uses
`test_conditioning: "nearest_image"`, not `"nearest_domain"`. While this was
being built the engine's `nearest_domain` became a *per-domain* rule that
decides one code from a single-domain primary test set plus a reference sample
(`run_experiment(..., conditioning_reference=...)`); this runner hands the
engine the pooled union of two target domains, so it cannot use that rule and
now refuses it with a message. With one code the per-image and per-domain rules
pick the same code for every image, so nothing is lost. Still waiting on
Pushpendra before any CREATE submission.

**Written 2026-09-12.** Task: make the Global FiLM arm runnable under the
*single-source* protocol — train on one domain, test on the other two active
domains — as a hedge. The brief (Section 5–6) defines Step 4 as
leave-one-domain-out (train on all-but-one), and that is what exists and is
tested. This variant is only worth running if Pushpendra says Stage 4 should be
train-on-one. **Do not launch the grid until he has.** Nothing for this variant
has been written yet; this is a specification plus the facts you need so you do
not rediscover them.

All paths are relative to `code/spfilm/` (the git repo). Python is
`.spfilm/bin/python`; tests are `.spfilm/bin/python -m unittest discover tests`
(378 pass as of writing).

---

## 1. Why this task exists, and why it is probably degenerate

The open thread with Pushpendra (Teams, 2026-09-11/12):

> Me: under LODO the held-out domain has no domain ID the model has seen — what
> should `s` be at test time?
> PS: "Let's try this. During inference supply the conditioning signal s of that
> domain whose distribution it's closest to."
> Me: per image or per held-out domain? *(asked with too much jargon)*
> PS: "I am not sure I understood that."

The follow-up being sent asks A (per image) vs B (per domain) in plain words.
There is a third possibility hiding in his answer: that he pictures Stage 4 as
**train on one domain, test on the others** (the Stage 3 single-source arm) with
FiLM added. This handoff prepares for that reading so we are not blocked if he
confirms it.

Be clear-eyed about what it can show. **With one source domain there is one
code.** The FiLM MLP then receives a constant input, so `(1+γ)·F+β` is a fixed
per-channel affine per level — exactly what `InstanceNorm2d(affine=True)` already
provides. The nearest-domain selector has one candidate and always picks it. The
fixed-code sweep has one entry. Selector validation accuracy is 100% by
construction. So Global FiLM(1 source) should equal the plain single-source
U-Net within seed noise; if it does not, that is a bug, not a finding. Say this
to Pushpendra before running 15 jobs. Its only legitimate uses:

- a sanity control ("FiLM with one code changes nothing", which it must not);
- as the plain half of the pair, the **existing** single-source runs already
  serve (Section 3), so only the film half costs compute.

---

## 2. Current state of Step 4 (what already works — read before touching anything)

Committed on 2026-09-12 (protocol change to three domains still uncommitted in
the working tree when this was written; check `git status`):

| Piece | Where |
|---|---|
| FiLM layer, frozen one-hot, MLP 64→256→256→2C, `(1+γ)F+β`, clamp ±5, fp32 under autocast | `src/spfilm/film/global_film.py` |
| Domain vocabulary, FOV RGB mean/std descriptor, nearest-domain selector (within-domain-scaled centroid distance), `OracleCondition` / `NearestCondition` / `FixedCondition` | `src/spfilm/film/conditioning.py` |
| `ConditionedUNet` composing an untouched `PlainUNet`; `build_model(arm, …)` | `src/spfilm/model.py` |
| Condition threaded through train/val/test; selector fitted before training; `test_conditioning_per_image.csv`, `val_selector_per_image.csv`, fixed-code sweep; `report["conditioning"]` | `src/spfilm/engine.py` (`run_experiment`, `_conditioning_report`) |
| Config parser: `arm ∈ {plain, global_film}`, optional `film` block, `protocol.paired_arm`; protocol domain list is the **active set** (may be a subset of `domains`) | `src/spfilm/stage3_single_source.py` |
| LODO runner composing folds from the active set | `run_stage3_lodo_3_1_fixed.py` (`fixed_lodo_folds(manifest, active_domains)`) |
| 3-domain LODO configs, both arms | `configs/stage4_plain_3dom*.json`, `configs/stage4_global_film_3dom*.json` |
| Submit scripts | `submit_stage4_plain.sh`, `submit_stage4_global_film.sh` |
| FiLM-vs-plain side-by-side + paired test + selector diagnostics | `aggregate_stage4_film.py` |
| Tests | `tests/test_global_film.py`, `tests/test_fixed_lodo.py::ActiveDomainTests` |

Design decisions already made and cross-checked against Pushpendra's own repo
(`https://github.com/p-singh-kcl/spatial_film_parcellation`, `models/film_mlp.py`;
the anonymous.4open.science link redirects there): identical block, identical
insertion (after every encoder ConvBlock, before pooling, decoder unconditioned).
We keep the frozen one-hot for both arms (his Global FiLM uses a learned
embedding; his SpFiLM uses one-hot). Do not reopen these.

**RIM-ONE-DL is dropped for now.** Active domains are `refuge_zeiss`,
`refuge_canon_val`, `drishti_gs`. RIM-ONE stays under `domains` in every config
because the locked manifests (`splits/lodo/lodo_manifest.json`,
`splits/single_source/single_source_manifest.json`) cover all four and are
rebuilt from that block during validation. Removing it from `domains` breaks
`check`. Restrict participation via the protocol list only.

---

## 3. What already exists for train-on-one — reuse it

`run_stage3_lodo_1_3.py` is the single-source runner (Stage 3.1). Facts:

- CLI: `run --source-domain <d> --seed <n> [--smoke] [--device] [--out-dir]`,
  plus `prepare` and `check`. Config: `configs/stage3_lodo_single.json` /
  `_create.json`, `stage: "single_source"`, domain list under
  `protocol.source_domains` (note: **not** `held_out_domains`).
- Folds come from the **locked manifest**: `manifest.folds` are
  `SingleSourceFold`s composed over all four budgeted partitions when the
  manifest was built, so `fold.target_domains` includes `rim_one_dl`
  (`src/spfilm/single_source.py:241`). The manifest is locked; filter targets in
  the runner, do not rebuild the manifest.
- `_run_one` (line ~377) hands the engine `split_records={"train","val","test": pooled union of targets}`
  and `extra_test_sets={target.value: records}`. The engine already threads the
  film condition into `extra_test_sets` (`evaluate_named_test_set(..., condition_fn=test_condition)`)
  and writes `test_<target>_conditioning_per_image.csv` per target. So **the
  engine side needs no change** for this variant.
- `main` (line ~575) **refuses `arm != "plain"`** — a guard I added on
  2026-09-12 precisely because one code is degenerate. Lifting it is the first
  code change; keep the reasoning as a printed warning rather than deleting it.
- Metadata block in `test_metrics.json` is `single_source` (not `fixed_lodo`);
  `arm` is `config.experiment_name`; per-target results live under
  `test_by_domain`; the pooled score is renamed away from `test` on purpose.
- **The 20 existing plain runs `artifacts/runs/single_s3_<source>_seed_<s>_*`
  are already the plain half of this pair.** Train-on-one training does not
  depend on the other domains, so for sources `refuge_zeiss`, `refuge_canon_val`,
  `drishti_gs` their per-target CSVs for the two non-RIM targets are exactly
  "train on 1, test on the remaining 2". Do not re-run plain. Check the runs are
  complete: `ls artifacts/runs | grep single_s3` (weights are on CREATE, not
  local — see `HANDOFF_prediction_overlays.md` §2).
- Aggregator: `aggregate_stage3_1_3.py` (`discover_runs`, `select_scientific_runs`).
  It dedups per (source, seed) by latest completion **before** checking for
  mixed arms (line ~296–314) — the same bug I fixed in `aggregate_stage3_fixed.py`
  on 2026-09-12. Once film single-source runs exist under `artifacts/runs`, it
  would silently pick whichever finished last. Add an `arm=` filter and move
  the arms check before the dedup, mirroring `aggregate_stage3_fixed.select_fixed_runs`.

---

## 4. What to build

Keep it small; most of it is configs and guards.

1. **Runner** (`run_stage3_lodo_1_3.py`):
   - Replace the hard refusal in `main` with: allow `global_film`, print one
     line — `"WARNING: single source ⇒ one code; Global FiLM reduces to a fixed
     per-channel affine and is a control, not a conditioning test"` — and record
     `"degenerate_conditioning": true` in the `single_source` metadata when
     `config.arm != "plain"`.
   - Restrict `target_domains` to `config.active_domains` (drop `rim_one_dl`)
     when the protocol list is a subset. Record `active_domains` and
     `inactive_domains` in the metadata, as the fixed runner now does. Refuse a
     source outside the active set (existing `--source-domain` check already
     does this via `config.source_domains`).
   - `paired_with`: `config.paired_arm or "stage3_lodo_fixed_budget_plain_unet"`
     (the historical default for plain must stay byte-identical; the parser
     already turns the legacy prose value into `None`).
2. **Configs**, derived from `configs/stage3_lodo_single{,_create}.json` the way
   `configs/stage4_*_3dom*.json` were derived from the fixed ones (see the
   generator pattern in this session's history: copy, change only the stated
   keys, keep `domains` complete):
   - `configs/stage4_single_source_global_film_3dom{,_create}.json`:
     `arm: "global_film"`, `film` block (levels 5, embedding 64, hidden 256,
     clamp 5.0, `test_conditioning: "nearest_domain"`),
     `experiment_name: "stage4_single_source_global_film_3dom"`,
     `output_dir: "artifacts/stage4_single_source_film_3dom"`,
     `protocol.source_domains: [refuge_zeiss, refuge_canon_val, drishti_gs]`,
     `protocol.inactive_domains: ["rim_one_dl"]`,
     `protocol.paired_arm: "stage3_single_source_plain_unet"`.
   - No plain config is needed (Section 3). If Pushpendra insists on a re-run,
     copy the film config with `arm: "plain"` and no `film` block.
3. **Submit script** `submit_stage4_single_source_film.sh`: copy
   `submit_lodo_stage3_single_source.sh`, point at the new `_create` config, job
   name `ssfilm_s4`, out-dir prefix `ssfilm_s4_`. 3 sources × 5 seeds = 15 jobs.
4. **Aggregation**: either extend `aggregate_stage4_film.py` with a
   `--protocol single_source` mode or write `aggregate_stage4_single_source_film.py`
   reusing `aggregate_stage3_1_3.discover_runs` + a fixed `select_scientific_runs(arm=…)`.
   Pair film-vs-plain **per (source, target)** on identical target images
   (50 each), Wilcoxon on seed-averaged per-image Dice, Holm over
   3 sources × 2 targets × 2 structures = 12 tests, FiLM as the reference so a
   positive Δ reads "FiLM helped". Reuse `aggregate_stage3_fixed.paired_tests(substrate, reference_arm=…)`
   and `holm_adjust`. Refuse to pair unless both arms' target sets match after
   dropping RIM-ONE from the plain runs' targets.
5. **Tests** (`tests/test_stage4_single_source_film.py`):
   - config loads with `arm == "global_film"`, active set is the three domains,
     `domains` still covers `set(Domain)`;
   - runner accepts the film arm and stamps `degenerate_conditioning: true`;
   - targets exclude `rim_one_dl` under the 3-domain protocol and match the
     4-domain fold's targets minus RIM-ONE (same 50 images per target);
   - `DomainVocabulary` of one domain, `NearestDomainSelector.fit` with one
     domain (within-domain scale is defined; falls back to total spread then 1.0);
   - `aggregate_stage3_1_3.select_scientific_runs` refuses mixed arms even when
     they share (source, seed) cells, and `arm=` filters.

---

## 5. Gotchas

- `Stage3SingleSourceConfig.from_json` takes `expected_stage` and `domains_key`;
  the 1_3 runner calls it with defaults (`"single_source"`, `"source_domains"`).
  The fixed runner uses `"lodo_fixed_budget"` / `"held_out_domains"`.
- `run_experiment` fits the selector on `splits["train"]` (one domain here) and
  reports `true_domains_in_vocabulary` per test set — for this variant it is
  `False` for every target, which is correct.
- The engine's fixed-code sweep runs over the **pooled** primary test set (the
  union of targets), because that is what `split_records["test"]` is in this
  runner. With one code it is one extra evaluation; harmless, but do not read
  the pooled number — the 1_3 runner already renames it away from `test`.
- Smoke mode shrinks to 128 px, base 8, 1 epoch; `select_single_source_smoke_splits`
  keeps every represented domain. Use `--out-dir artifacts/runs/<name>` and
  delete the `_smoke` directory afterwards; the aggregators ignore smoke runs
  but the directory listing gets noisy.
- CREATE: ssh hangs until you re-auth MFA at the KCL portal; preempted jobs do
  not requeue in practice (they die; a relaunch into the same `--out-dir` resumes
  from `resume_state.pt`). Test ssh early. Smoke wall time 20 min.
- Pushpendra reads short, actionable messages. If you need his decision, three
  lines: what A and B are in his domains, which one is built, "which do you
  want?" — no method detail.

---

## 6. Verification

1. `.spfilm/bin/python -m unittest discover tests` — everything green including
   the untouched Stage 3 suites.
2. `.spfilm/bin/python run_stage3_lodo_1_3.py --config configs/stage4_single_source_global_film_3dom.json check --skip-mask-audit`
   — targets per source are the two other active domains only.
3. CPU smoke: `… run --source-domain refuge_zeiss --seed 42 --smoke --device cpu --out-dir artifacts/runs/ssfilm_smoke_local`
   — expect `codes=['refuge_zeiss']`, `test_<target>_conditioning_per_image.csv`
   for `refuge_canon_val` and `drishti_gs` only, one fixed-code sweep entry,
   `degenerate_conditioning: true`.
4. Then stop and wait for Pushpendra's answer. If he confirms train-on-one for
   Stage 4: CREATE smoke, one real `refuge_zeiss 42`, then the aggregation
   against the existing `single_s3_*` plain runs with `--expected-seeds 42`.
   Expect FiLM ≈ plain; anything else is a bug to chase, not a result.

---

## 7. Related

- `Edward_Project_Brief.pdf` §5–6 (Step 4 is LODO), `literature/Spatial_Film_Draft__Final_ 1.pdf` §2.1.
- Plan file for the LODO Step 4 work: `~/.claude/plans/we-have-now-started-joyful-teapot.md`.
- `run_reports/` for the Stage 3 brief format Pushpendra actually read.
