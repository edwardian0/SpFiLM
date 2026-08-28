# Stage 2 Epoch Budget Audit

**Question:** the config asks for 300 epochs; runs terminate before reaching it. Why, and how do we retain early stopping as a monitor rather than a terminator?

| | |
| --- | --- |
| Audited commit | `a25e223ce3e396af2995e4b3aac104a5959ed03a` (working tree clean) |
| Date | 2026-08-28 |
| Entry point | [run_stage_s2.py](run_stage_s2.py) → [engine.run_experiment](src/spfilm/engine.py#L429) |
| Scope | Read-only. No repository source file was modified. |
| Patch | `run_reports/s2_epoch_budget_audit.patch` — built, executed, `git apply --check` verified. **Not applied.** |
| Artifact | https://claude.ai/code/artifact/7758c74a-598f-4106-bd5f-97f435714108 |

**Short answer:** the early-stopping `break` at [engine.py:595-597](src/spfilm/engine.py#L595-L597) fired in both completed runs. Nothing else did. But the more consequential finding is §3.5: under the current scheduler, 71% of a 300-epoch budget would run at the learning-rate floor.

---

## 0. Three premises in the brief that don't match the code

Stated first, because they change what the fix means.

1. **Stage 2 is not a three-arm study.** There is one arm: `PlainUNet` at [model.py:56-57](src/spfilm/model.py#L56-L57), explicitly "no FiLM or SpFiLM conditioning". No Global FiLM or SpFiLM module exists anywhere in the repository — the only hits for "FiLM" in code are comments and docstrings. The requirement that all three arms be treated identically is therefore *vacuously satisfiable today*, and is the thing to guard when the arms are written (see §3, "invariant to protect").

2. **Stage 2 is not leave-one-domain-out.** It is single-domain, in-domain train/val/test. `compose_lodo_fold` at [data.py:1202-1205](src/spfilm/data.py#L1202-L1205) and `compose_lodo_folds` at [lodo.py:3-4](src/spfilm/lodo.py#L3-L4) are both `pass`, with no callers anywhere.

3. **Runs stop for the reason suspected** — the early-stopping `break` — **but extending the loop is a bookkeeping win, not an optimisation one.** See §3.5.

---

## 1. Every path that can end a run before epoch 300

The epoch loop is [engine.py:528](src/spfilm/engine.py#L528):

```python
for epoch in range(1, config.epochs + 1):
```

Entry chain: `submit_*.sh:42` → [`main`](run_stage_s2.py#L587) → [`resolve_config`](run_stage_s2.py#L194) → [`run_experiment`](src/spfilm/engine.py#L429) → the loop.

### 1.1 Mechanisms present

| file:line | mechanism | trigger condition | currently active? |
| --- | --- | --- | --- |
| [engine.py:595-597](src/spfilm/engine.py#L595-L597) | **early-stop `break`** | `epoch >= min_epochs (30)` **and** `epochs_without_improvement >= patience (20)` | **YES — this is the one that fired** |
| [engine.py:563](src/spfilm/engine.py#L563) | hard-coded `min_delta` | `val_loss < best - 1e-5` defines "improvement"; feeds the counter above | YES, and it is a magic number |
| [engine.py:525,591](src/spfilm/engine.py#L525) | patience counter | incremented on every non-improving epoch | YES |
| [engine.py:549-550](src/spfilm/engine.py#L549-L550) | NaN/inf guard | non-finite `val_loss` → `RuntimeError` | present, never fired; crashes loudly rather than exiting clean |
| [engine.py:265-266](src/spfilm/engine.py#L265-L266), [:333-334](src/spfilm/engine.py#L333-L334) | per-batch `break` on `max_batches` | `max_batches` is `1` only when `smoke=True` | **inactive** — [run_stage_s2.py:635](run_stage_s2.py#L635) passes `smoke=False` deliberately |
| [engine.py:446-456](src/spfilm/engine.py#L446-L456) | engine smoke override forcing `epochs=1, patience=1` | `run_experiment(smoke=True)` | **inactive**, same reason |
| [run_stage_s2.py:219-220](run_stage_s2.py#L219-L220) | `--epochs` also shrinks `patience` and `min_epochs` | only when `--epochs` is passed | inactive in the SLURM jobs (no `--epochs`) |
| [run_stage_s2.py:206-214](run_stage_s2.py#L206-L214) | `--smoke` forces `epochs=2, patience=2` | only with `--smoke` | inactive in production |
| `submit_drishti_s2.sh:9` | **SLURM `--time=0-00:30:00`** | 30-minute wall | not yet fired, but see §4 — this is the next thing that will bite |
| `submit_refuge_s2.sh:9` / `submit_rimone_s2.sh:9` | SLURM wall 6 h / 5 h | | ample headroom |
| `submit_*.sh:4`, `:17-18` | `--partition=interruptible_gpu`, no `--requeue`, no SIGUSR1 | preemption kills the job dead | **live risk**, not the observed cause |
| [engine.py:515-518](src/spfilm/engine.py#L515-L518) | `ReduceLROnPlateau(min_lr=1e-6)` | LR floors; **does not stop the loop** | active, no exit — but see §3.5 |

### 1.2 Mechanisms explicitly not present

Searched; zero hits. Recorded as **not found** rather than assumed absent.

| candidate | result |
| --- | --- |
| `max_epochs` / `num_epochs` / `max_steps` / `max_time` / `limit_train_batches` / `check_val_every_n_epoch` / `Timer` | **not found** — there is no Lightning or framework layer; the loop is hand-written |
| `T_max` / `total_steps` / `milestones` / poly power | **not found** — `ReduceLROnPlateau` is the only scheduler and has no epoch horizon |
| step / iteration budget | **not found** |
| signal handlers, `SIGUSR1`/`SIGTERM`, checkpoint-and-exit | **not found** — only the comment at `submit_*.sh:17` |
| `drop_last` | **not found** — [`_make_loader`](src/spfilm/engine.py#L217-L233) leaves the default `False`, so no batches are dropped and no `steps_per_epoch` miscount is possible |
| resume logic | **not found** — the only `torch.load` in project code is [engine.py:599](src/spfilm/engine.py#L599), *after* the loop, to reload the best checkpoint for testing. No run can start from a high epoch counter |
| `try`/`except` around the epoch loop | **not present** — the only `try` blocks in engine.py are at [:89](src/spfilm/engine.py#L89) and [:387](src/spfilm/engine.py#L387), both unrelated. An exception cannot be swallowed into a clean-looking exit |
| env-var overrides | **not found** — the only `os.environ` in project code is `MPLCONFIGDIR` at [run_stage_s2.py:36](run_stage_s2.py#L36) |
| OOM / dataloader-worker death | no evidence, and no mechanism to hide it — a worker death raises and aborts the job before the test block |

### 1.3 Config shadowing

One key, `epochs`, read once at [engine.py:528](src/spfilm/engine.py#L528). No YAML. Precedence is JSON → CLI, resolved in one place at [run_stage_s2.py:202-228](run_stage_s2.py#L202-L228).

**No stage-1 inheritance** — there is no stage-1 config. Git history of the value:

| commit | epochs | patience | min_epochs |
| --- | ---: | ---: | ---: |
| `b771bca` | 40 | 8 | — |
| `6f4700a` | 150 | 20 | 30 |
| `ec384d2` | 300 | 20 | 30 |

**Stale documentation:** [STAGE2.md:88-89](STAGE2.md#L88-L89) still says "up to 40 epochs… early stop after eight non-improving epochs", contradicting every committed config.

---

## 2. Which one actually fired

**Artefact caveat, stated up front.** The completed full runs live on CREATE at `/users/k23123868/edward/logs/` and `artifacts/runs/refuge_s2_36631447/`. **Those are not accessible from this machine.** Everything local under `artifacts/` is a 1–2 epoch laptop smoke. The evidence below is the stdout committed to `run_reports/`, which includes the per-epoch table and the terminator line. There is no TensorBoard or W&B in this project — no `SummaryWriter`, no `wandb` import.

| run | terminal epoch | best epoch | trigger | evidence |
| --- | ---: | ---: | --- | --- |
| **REFUGE**, job 36631447 | **83 / 300** | 63 | early-stop `break` | [`s2_refuge_report.md:169`](run_reports/s2_refuge_report.md#L169) carries the literal `early_stopping best_epoch=63`, printed only from [engine.py:596](src/spfilm/engine.py#L596). Epoch table lines 87–168 run 1→83 unbroken. Arithmetic consistent: last improvement epoch 63, 63+20 = 83, `min_epochs=30` satisfied. |
| **Drishti-GS** | **146 / 300** | 126 | same `break` | [`s2_drishti_report.md`](run_reports/s2_drishti_report.md) §3: "Training stopped after epoch 146, with the best model selected at epoch 126". 126+20 = 146. **No epoch table was pasted into this report**, so the terminator line itself is not in the artefact — the arithmetic is the evidence, and it is consistent. |
| **RIM-ONE-DL** | **no full run exists** | — | — | Only a 2-epoch smoke: [`s2_rim_one_dl_wiring_report.md:337`](run_reports/s2_rim_one_dl_wiring_report.md#L337). Local `artifacts/runs/rim_one_wiring_smoke_final/history.csv` has 2 rows. |

**Verdict: the early-stopping `break` fired, in both completed runs. No wall-clock limit, no crash, no silent partial run.** SLURM was not implicated: REFUGE used ~325 s of a 6-hour wall.

---

## 3. Design spec

`early_stopping` becomes a nested, validated block. `patience` and `min_epochs` **move into it** rather than being duplicated — duplication across the CLI override at [run_stage_s2.py:219-220](run_stage_s2.py#L219-L220) would have been a shadowing landmine. Legacy top-level keys still parse, so every `resolved_config.json` already on disk stays readable; setting **both** forms raises rather than picking a silent winner.

```json
"early_stopping": {
  "mode": "monitor",        // monitor | terminate  <- the only rollback lever needed
  "metric": "val_loss",     // val_loss | val_disc_dice | val_cup_dice | val_*_iou
  "direction": "min",
  "min_delta": 1e-05,       // was hard-coded 1e-5 at engine.py:563
  "patience": 20,
  "min_epochs": 30
}
```

### 3.1 Requirements, and how each is met

| requirement | how |
| --- | --- |
| 1. Loop always runs the full budget | The `break` survives but is gated on `mode == "terminate"`. Under `monitor` it is unreachable. |
| 2. Monitor survives | The rule is evaluated in full every epoch regardless of mode; under `monitor` its only effect is to record `would_have_stopped_at_epoch` the first time it fires. `monitored_metric`, `epochs_without_improvement` and `would_have_stopped_at_epoch` are written to `history.csv` and to the stdout epoch line (two new columns, `patience` and `wh_stop`) every epoch. |
| 3. Checkpointing | `best_model.pt` on every improvement of the monitored metric; `last_model.pt` every epoch (new). One shared writer `_save_checkpoint`, so payload schemas cannot drift. **Test-time evaluation loads `best_model.pt`** at [engine.py:599](src/spfilm/engine.py#L599), unchanged. |
| 4. Config-driven, no hard-coded numbers | `mode`, `metric`, `direction`, `min_delta`, `patience`, `min_epochs` are all config. The `1e-5` at [engine.py:563](src/spfilm/engine.py#L563) is removed. |
| 5. LR schedule | See §3.5. |
| 6. Identical across arms | See below. |
| 7. Seeding, determinism, provenance | Untouched. [`seed_everything`](src/spfilm/engine.py#L115-L123), the loader `Generator` at [:505](src/spfilm/engine.py#L505) and `worker_init_fn` are not in the diff. Git-SHA logging at [run_stage_s2.py:128-134](run_stage_s2.py#L128-L134) and `resolved_config.json` are unchanged (the latter now carries the nested block). |

**Naming note on requirement 3.** The brief says `best.pt` / `last.pt`. The patch keeps `best_model.pt` and adds `last_model.pt`. Renaming would break `report["artifacts"]["checkpoint"]` and the two already-published reports. Rename later if you want, but do it as a separate commit.

### 3.2 The invariant to protect for requirement 6

Checkpoint selection has **exactly one implementation**: `run_experiment` does not branch on arm, dataset, or seed anywhere in the selection or reload path. That is what makes "same rule for all three arms and all seeds" true today.

**When Global FiLM and SpFiLM are added, keep them inside `run_experiment`.** Any arm that gets its own training entry point is how this requirement silently breaks. There is no such code path today — checked.

### 3.5 LR schedule — read this before re-running

**Scheduler horizon: there is none.** `ReduceLROnPlateau` at [engine.py:515-518](src/spfilm/engine.py#L515-L518) (`factor=0.5, patience=3, min_lr=1e-6`) is metric-driven, not epoch-driven. No `T_max`, no `total_steps`, no `milestones`.

Measured trajectory, from the REFUGE epoch table:

| first epoch at LR | 1 | 38 | 50 | 58 | 67 | 71 | 75 | 79 | 83 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| learning rate | 1e-3 | 5e-4 | 2.5e-4 | 1.25e-4 | 6.25e-5 | 3.13e-5 | 1.56e-5 | 7.81e-6 | 3.91e-6 |

**The good news.** Extending the loop **does not change the LR curve for any epoch that would have run anyway**. Given the same val-loss sequence, epochs 1–83 of a 300-epoch REFUGE run reproduce the completed run's LR trajectory exactly. **There is no confound in the region you already have results for.** This is the reason to leave the scheduler alone in this patch.

**The bad news.** One more halving clamps at `min_lr=1e-6`, landing around **epoch 87**. So epochs ~87–300 — **213 epochs, 71% of the budget** — train at 0.1% of the initial LR. Roughly 14 of the ~19 minutes per REFUGE run do nothing measurable. The plateau confirms it: over epochs 50–83, val disc Dice sd = 0.00084, val cup Dice sd = 0.00261.

**Recommendation — a decision, not a free win:**

- **(a) Recommended for this patch: change nothing about the scheduler.** The 300 epochs then serve a *bookkeeping* purpose — the full monitor trace and a recoverable final-epoch model — not an optimisation one. Zero confound; the REFUGE and Drishti numbers stay directly comparable.
- **(b) If you want the 300 epochs to do optimisation work**, that is a separate, deliberate change: `CosineAnnealingLR(T_max=config.epochs)` or raising `min_lr` to `1e-5`. **It alters the LR at every epoch and invalidates comparison with the two completed runs.** If you do it, do it once, before any arm is trained, for all three arms and all datasets — never mid-study.

**(b) is deliberately not in the patch.** Changing it silently is exactly the confound the brief warned about.

---

## 4. Cost and feasibility

### 4.1 Measured, from the pasted logs (A100)

| dataset | s/epoch | source | 300 epochs |
| --- | ---: | --- | ---: |
| REFUGE (256 train / 64 val) | **3.81** | epoch table, n=83; median 3.80; ep-1 warmup 6.5 s excluded | **19.1 min** |
| Drishti (40 / 10) | **2.82** | 411 s ÷ 146 epochs, report §3 | **14.1 min** |
| RIM-ONE-DL (340 / 48) | **~4.6 (est.)** | never run at scale; scaled from REFUGE by sample count | **~23 min** |

### 4.2 Projections

The brief's shape (seeds × held-out domains × arms) does not exist yet, so both:

- **Stage 2 as it stands** — 3 datasets × 3 seeds × 1 arm = 9 runs → **≈2.8 GPU-hours.**
- **The LODO study once built** — 3 arms × 3 held-out domains × 3 seeds = 27 runs; each trains on two pooled source domains at ≈7.5 s/epoch → ≈37 min per run; +15% for SpFiLM's conditioning path → **≈20 GPU-hours.**

**Re-running Stage 2 from scratch under the new config is not remotely infeasible.** Nothing has to give. Even the full 27-run LODO fits comfortably.

### 4.3 The one thing that will break, and it isn't GPU budget

`submit_drishti_s2.sh:9` sets `--time=0-00:30:00`. At 300 epochs Drishti needs ~14 min of training plus discovery, audit, contact sheet, test pass and figures — roughly 20 min against a 30-minute wall, on a partition that can hand you a slower GPU. **Raise it to `0-02:00:00`.** REFUGE (6 h) and RIM-ONE (5 h) have ample headroom.

Second: `--partition=interruptible_gpu` with no `--requeue` and no resume path. `last_model.pt` at least leaves recoverable weights after a preemption, but a preempted run is still a lost run.

### 4.4 GPU memory — not measured

No CUDA in the audit environment, and no `max_memory_allocated` call exists in the codebase to read from the logs. Hard facts available: **1,944,066 parameters** = 7.78 MB fp32 weights + ~15.6 MB Adam state; the jobs ran on one A100 with `--mem=32G` host RAM under AMP at batch 8 / 512 px and completed. **Add `torch.cuda.max_memory_allocated()` to the report block on the first re-run** rather than trusting an estimate — activations at 512² × batch 8 dominate, and guessing at them would be worthless.

### 4.5 Disk

`best_model.pt` measured at **23.43 MB** (verified by loading `artifacts/runs/rim_one_wiring_smoke_final/best_model.pt`: model + optimizer state). Adding `last_model.pt` doubles it to **46.9 MB per run** — 422 MB for 9 Stage-2 runs, 1.27 GB for a 27-run LODO. Negligible.

**I/O cost of `last_model.pt`:** rewritten every epoch = 23.43 MB × 300 = **7.0 GB written per run**, sustained ~6 MB/s; `torch.save` ~30–80 ms ≈ 1–2% of a 3.8 s epoch. Fine on cephfs. To avoid it, drop `optimizer_state_dict` from `last_model.pt` (7.78 MB) — at the cost of any future resume option.

---

## 5. Risk review

### 5.1 Leakage check — highest priority

**No cross-domain leakage is possible at Stage 2, because Stage 2 has no held-out target domain.** Each run draws from exactly one dataset, and each dataset stamps exactly one `domain` literal:

- REFUGE → `domain="refuge_zeiss"` at [data.py:151](src/spfilm/data.py#L151)
- Drishti → `domain="drishti_gs"` at [data.py:203](src/spfilm/data.py#L203)
- RIM-ONE-DL → `domain="rim_one_dl"` at [data.py:428](src/spfilm/data.py#L428)

Split construction, all within that single domain:

| dataset | function | file:line | policy |
| --- | --- | --- | --- |
| REFUGE | `stratified_partition` | [data.py:987-1019](src/spfilm/data.py#L987-L1019) | seeded stratified 80/20 then 80/20 within Training400 only → 256/64/80 |
| Drishti | `provider_partition` | [data.py:1022-1049](src/spfilm/data.py#L1022-L1049) | provider's 51-image test set **locked** via `split_hint` ([data.py:208](src/spfilm/data.py#L208)); val drawn only from `provider_train` → 40/10/51 |
| RIM-ONE-DL | `load_rim_one_dl_split_manifest` | [data.py:492](src/spfilm/data.py#L492) | committed manifest `splits/rim_one_dl.json`, 340/48/97 |

Every path terminates in [`validate_splits`](src/spfilm/data.py#L1052-L1064), which raises unless train/val/test IDs are **pairwise disjoint**, **cover every record exactly once**, and are all non-empty. The run also prints the domain set at [run_stage_s2.py:373](run_stage_s2.py#L373) — a multi-domain pool would be visible in every log.

**Conclusion: the validation split used for checkpoint selection is drawn only from the run's single domain and cannot touch a held-out target domain, because none exists.**

**The invariant to write into Stage 3.** The leakage question becomes live at LODO, which is unimplemented ([lodo.py:3-4](src/spfilm/lodo.py#L3-L4), [data.py:1202-1205](src/spfilm/data.py#L1202-L1205), both `pass`, no callers). When `compose_lodo_fold` is written, the rule to enforce is: **the val split is drawn from the source-domain pool only, and the held-out domain contributes to `test` alone.** Put that assertion inside `validate_splits`, where it cannot be bypassed.

### 5.2 One genuine leakage risk that exists today

RIM-ONE-DL's manifest carries its own caveat in `provenance.fellow_eye_caveat` (`splits/rim_one_dl.json`):

> r3 filenames encode eye but not patient, so fellow-eye correlation is undetectable from filenames rather than known to be absent

The split policy is a flat random 70/10/20 over 485 images. **The same patient's two eyes can straddle train and test.** That inflates in-domain RIM-ONE numbers and is not detectable from the available data. It is honestly documented; it should appear in the methodology as a stated limitation, not be discovered by a reviewer.

### 5.3 Overfitting — the best-vs-final gap

Measured on REFUGE: best (ep 63) val_loss 0.0954, disc 0.9564, cup 0.8705 → final (ep 83) 0.0966 / 0.9565 / 0.8680. **Gap: +0.0012 loss, +0.0001 disc, −0.0025 cup.** No overfitting; the model is simply flat.

Reporting both is still the right call, and the patch does it: an extra test pass on the in-memory final-epoch model *before* the best checkpoint reloads, written to `test_per_image_metrics_final_epoch.csv` and `report["test_final_epoch"]`, printed as a best / final / gap block. Cost: one test pass, ~1 s. Config-gated by `report_final_epoch_test`.

### 5.4 Validation set size and noise — the real statistical problem

| dataset | val n | epoch-to-epoch val Dice noise | verdict |
| --- | ---: | --- | --- |
| REFUGE | 64 | disc sd 0.00084, cup sd 0.00261 (epochs 50–83); mean \|Δ/epoch\| cup 0.00248 | selection is **within noise** but harmless |
| **Drishti** | **10** | not logged per-epoch in the report | **selection is close to arbitrary** |
| RIM-ONE-DL | 48 | no full run | untested |

**The REFUGE arithmetic.** Test-set per-image cup Dice sd is 0.0628 over n=80 → standard error **0.0070**. The best-vs-final val difference the selector discriminates on is **0.0025** — about **3× smaller than the test-set standard error**. Choosing epoch 63 over epoch 83 is not a statistically meaningful act; it is fitting validation noise. It does not hurt, but it should not be described as having found a better model.

**Drishti at n=10 is the one to worry about.** Its test-set cup Dice sd is 0.1375; a 10-image validation set has a standard error around **0.043** — 4.3 Dice points. Best-checkpoint selection on that is close to a coin flip, and the 40/10/51 split is fixed by the provider's 50-image development set, so it cannot simply be enlarged.

Mitigations, in order of preference:

1. Report the **final-epoch metric as the headline for Drishti** and best-checkpoint as secondary — the patch makes both available.
2. Select on a **smoothed monitored metric** (e.g. 5-epoch moving mean) rather than the raw per-epoch value.
3. **k-fold** the 50 development images.

None of these is in the patch. Each changes the selection rule, which is a study-design decision rather than a bug fix.

---

## 6. The patch

`run_reports/s2_epoch_budget_audit.patch` — **not applied.** `git apply --check` passes against `a25e223` with a clean tree. Built and executed in an isolated copy; the repository itself was never modified during the audit.

```
 src/spfilm/engine.py               | 238 +++++++++++++++++++++++++++++++++---
 run_stage_s2.py                    |  76 ++++++++++-
 configs/stage2_refuge.json         |  13 ++
 configs/stage2_refuge_create.json  |  13 ++
 configs/stage2_drishti_create.json |  13 ++
 configs/stage2_rimone_create.json  |  13 ++
 6 files changed, 323 insertions(+), 43 deletions(-)
```

Apply with:

```bash
git apply run_reports/s2_epoch_budget_audit.patch
```

The hunk carrying all of the semantics:

```diff
@@ engine.py — the epoch loop @@
         scheduler.step(val_loss)
+        monitored = monitored_value(stopping.metric, val_loss, val_metrics)
+        is_best = stopping.is_improvement(monitored, best_monitored)
+        epochs_without_improvement = 0 if is_best else epochs_without_improvement + 1
+        # The terminating rule is still evaluated in full every epoch. Under
+        # mode="monitor" its only effect is to record the epoch it first fired,
+        # so the early-stopped model stays reportable after the fact.
+        stop_rule_met = (
+            epoch >= stopping.min_epochs
+            and epochs_without_improvement >= stopping.patience
+        )
+        if stop_rule_met and would_have_stopped_at_epoch is None:
+            would_have_stopped_at_epoch = epoch
         row = { ... ,
+            "monitored_metric": monitored,
+            "epochs_without_improvement": float(epochs_without_improvement),
+            "would_have_stopped_at_epoch": float(would_have_stopped_at_epoch or -1),
         }
-        is_best = val_loss < best_val_loss - 1e-5

         if is_best:
+            best_monitored = monitored
             best_epoch = epoch
-            epochs_without_improvement = 0
-            torch.save({...}, checkpoint_path)
+            _save_checkpoint(checkpoint_path, model, optimizer, epoch, val_metrics, config)
-        else:
-            epochs_without_improvement += 1
+        # last_model.pt is rewritten every epoch so the final-epoch weights are
+        # recoverable without re-running, whatever the monitor decided.
+        _save_checkpoint(last_checkpoint_path, model, optimizer, epoch, val_metrics, config)
+
-        if epoch >= config.min_epochs and epochs_without_improvement >= config.patience:
+        if stopping.mode == "terminate" and stop_rule_met:
             print(f"early_stopping best_epoch={best_epoch}", flush=True)
             break
```

**No hyperparameter values change:** `patience` stays 20, `min_epochs` stays 30, and `min_delta` is the same `1e-5` that was hard-coded at [engine.py:563](src/spfilm/engine.py#L563).

---

## 7. Verification plan

### 7.1 Production smoke

```bash
python -u run_stage_s2.py --config configs/stage2_refuge.json \
  --epochs 3 --out-dir artifacts/runs/verify_monitor --device cuda
```

Expected: all 3 epochs run; the epoch line carries two new columns; the summary reads

```
epoch budget      ran 3 of 3 configured epochs (early_stopping.mode=monitor)
early stopping    metric=val_loss (min) patience=3 min_delta=1e-05 min_epochs=0 -> would_have_stopped_at_epoch=None
```

### 7.2 Forcing the monitor to fire

A 3-epoch run is too short to trip patience naturally. Force it with `min_delta=10.0, patience=2, min_epochs=0` over 6 epochs. **Actual output from the audit run:**

```
  epoch |        lr | train_loss |   val_loss | val_dice_disc | val_dice_cup |  time | patience | wh_stop | best
  1/6   | 1.000e-03 |     1.8029 |     1.7582 |        0.0551 |       0.0134 |  0.4s |     0/2  |       - | *
  2/6   | 1.000e-03 |     1.7697 |     1.7367 |        0.0934 |       0.0144 |  0.4s |     1/2  |       - |
  3/6   | 1.000e-03 |     1.7398 |     1.7193 |        0.1168 |       0.0153 |  0.4s |     2/2  |       3 |
  4/6   | 1.000e-03 |     1.7192 |     1.7004 |        0.1510 |       0.0162 |  0.4s |     3/2  |       3 |
  5/6   | 1.000e-03 |     1.6995 |     1.6836 |        0.1807 |       0.0167 |  0.4s |     4/2  |       3 |
  6/6   | 1.000e-03 |     1.6837 |     1.6685 |        0.2273 |       0.0166 |  0.4s |     5/2  |       3 |
```

### 7.3 Checks executed, all passing

- Rule fired at epoch 3; loop ran to 6; counter kept climbing past patience.
- `history.csv` gained `monitored_metric`, `epochs_without_improvement`, `would_have_stopped_at_epoch`.
- `best_model.pt` epoch = 1, `last_model.pt` epoch = 6, both 23.43 MB.
- `test_metrics.json` → `"would_have_stopped_at_epoch": 3, "terminated_training": false`.
- `resolved_config.json` round-trips through `Stage2Config.from_json`.
- Legacy flat `patience`/`min_epochs` still load; both-forms and bad `mode` raise.
- `pytest tests -q` → **22 passed** against the patched tree.

---

## 8. Rollback

Config alone, no code change:

```json
"early_stopping": { "mode": "terminate", ... }
```

Restores the pre-patch behaviour exactly: same `min_delta` (1e-5), same `patience` (20), same `min_epochs` (30), same `break`, same `early_stopping best_epoch=N` line.

**Verified:** the terminate run stopped at epoch 3 with `early_stopping best_epoch=1` and produced **test metrics bit-identical to the monitor run**, confirming the change is selection-preserving.

A per-invocation lever also exists without touching the config: `--early-stopping-mode terminate`.

---

## 9. Deliberately left alone

- The `ReduceLROnPlateau` horizon — §3.5 option (b), a study-design decision.
- `submit_drishti_s2.sh`'s 30-minute wall — a one-line change better made knowingly (§4.3).
- [STAGE2.md:88-89](STAGE2.md#L88-L89), which still documents the 40-epoch / patience-8 regime.
