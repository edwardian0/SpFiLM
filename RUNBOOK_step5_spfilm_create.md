# Runbook: SpFiLM (K=8) on CREATE

Prepared 2026-09-28 for §5.G of `HANDOFF_step5_spatial_film.md`. Everything
before this step is done and verified locally (tests, `check` of both configs,
CPU smokes, simulated submit scripts, aggregators on synthetic runs). Nothing
has been run on CREATE. The commands below are in order; each stage says what
to read before going on.

Grid: 5 train-on-all jobs (`allsf_s4`, seeds 42–46) and 15 LODO jobs
(`sfilm_s5`, 3 held-out domains × seeds 42–46), on the same folds, seeds and
test images as the completed plain and Global FiLM arms.

## 0. Mac: push the commit

```bash
git push origin main
```

## 1. CREATE: pull

If `ssh create` hangs for ~40 s and dies, refresh MFA at
<https://portal.er.kcl.ac.uk/mfa/> first. Work on `erc-hpc-login1` or `-login2`.

```bash
cd ~/edward/spfilm && git status && git pull
```

`git status` should show only untracked run outputs. If `git pull` refuses
because untracked files would be overwritten (for example `run_reports/`
files the aggregators wrote there that the commit now tracks), move exactly
those files aside and pull again.

## 2. CREATE: environment and `check`

```bash
module load anaconda3/2022.10-gcc-13.2.0 && eval "$(conda shell.bash hook)" && conda activate spfilm
```

```bash
python run_stage4_all_domains.py --config configs/stage4_all_domains_spatial_film_k8_3dom_create.json check --skip-mask-audit
```

```bash
python run_stage5_lodo.py --config configs/stage5_lodo_spatial_film_k8_3dom_create.json check --skip-mask-audit
```

Expect `"status": "OK"`, manifest `af4fa639…`, train-on-all 120 / 30 / 3 × 50,
and LODO 80 / 20 / 50 per fold with `conditioning_reference` 40.

Before the grid, find the node that killed `gfilm_s5_37503695` with an
uncorrectable ECC error, and add it to the `#SBATCH --exclude` list of both
SpFiLM scripts (and the Global FiLM ones) if it is not already there:

```bash
sacct -j 37503695 -X -o JobID,NodeList%20,State
```

## 3. Smokes (20 min limit)

```bash
sbatch --time=0-00:20:00 submit_stage4_all_domains_spatial_film.sh 42 --smoke
```

```bash
sbatch --time=0-00:20:00 submit_stage5_lodo_spatial_film.sh drishti_gs 42 --smoke
```

In each `.out` (`~/edward/logs/{allsf_s4,sfilm_s5}_<job>.out`): `arm=spatial_film`,
`rank=8`, and the per-domain lines (train-on-all) or `code used=` line (LODO).
Smoke Dice and the smoke's held-out signal (decided from 3 images at 128 px)
mean nothing; smoke timing says nothing about the real run.

## 4. Seed 42 for real (1 + 3 jobs)

```bash
sbatch submit_stage4_all_domains_spatial_film.sh 42
```

```bash
for d in refuge_zeiss refuge_canon_val drishti_gs; do sbatch submit_stage5_lodo_spatial_film.sh "$d" 42; done
```

When they finish, read `Elapsed` against the limits (train-on-all 3 h, LODO 2 h;
Global FiLM train-on-all took ~20 min). If a job came close, raise `--time` in
the script before seeds 43–46.

```bash
sacct -u "$USER" -S today -o JobID,JobName%10,State,Elapsed,NodeList%20 | grep -E "allsf_s4|sfilm_s5"
```

First look, SpFiLM against Global FiLM, one seed ("(1 seed)" cells, no spread):

```bash
python aggregate_stage4_all_domains.py --arm-a stage4_all_domains_fixed_budget_global_film_3dom --arm-b stage4_all_domains_fixed_budget_spatial_film_k8_3dom --expected-seeds 42
```

```bash
python aggregate_stage5_lodo.py --arm-a stage5_lodo_fixed_budget_global_film_3dom --arm-b stage5_lodo_fixed_budget_spatial_film_k8_3dom --expected-seeds 42
```

Read the wrong-signal penalty first (train-on-all): if SpFiLM's is near zero
its conditioning is inert. For LODO, read each arm's "used − best signal" and
sweep spread next to the Dice difference: the nearest-colour rule handed Global
FiLM the worse signal in every disc cell, and SpFiLM gets signals by the same
rule.

## 5. Seeds 43–46 (4 + 12 jobs)

```bash
for s in 43 44 45 46; do sbatch submit_stage4_all_domains_spatial_film.sh "$s"; done
```

```bash
for s in 43 44 45 46; do for d in refuge_zeiss refuge_canon_val drishti_gs; do sbatch submit_stage5_lodo_spatial_film.sh "$d" "$s"; done; done
```

A preempted job dies (it does not requeue): resubmit that cell; it restarts
from epoch 1 in a new directory. An "uncorrectable ECC error" is a broken GPU:
resubmit with `--exclude=` repeating the script's whole list plus the node
(a command-line `--exclude` replaces the list).

## 6. The four reports

Each is Holm-adjusted within its own table (6 tests).

```bash
python aggregate_stage4_all_domains.py --arm-a stage4_all_domains_fixed_budget_plain_unet_3dom --arm-b stage4_all_domains_fixed_budget_spatial_film_k8_3dom --report-out run_reports/s5_all_domains_spfilm_vs_plain.md --csv-out run_reports/s5_all_domains_spfilm_vs_plain_cells.csv
```

```bash
python aggregate_stage4_all_domains.py --arm-a stage4_all_domains_fixed_budget_global_film_3dom --arm-b stage4_all_domains_fixed_budget_spatial_film_k8_3dom --report-out run_reports/s5_all_domains_spfilm_vs_global.md --csv-out run_reports/s5_all_domains_spfilm_vs_global_cells.csv
```

```bash
python aggregate_stage5_lodo.py --arm-a stage5_lodo_fixed_budget_plain_unet_3dom --arm-b stage5_lodo_fixed_budget_spatial_film_k8_3dom --report-out run_reports/stage5_lodo_spfilm_vs_plain.md --csv-out run_reports/stage5_lodo_spfilm_vs_plain_cells.csv
```

```bash
python aggregate_stage5_lodo.py --arm-a stage5_lodo_fixed_budget_global_film_3dom --arm-b stage5_lodo_fixed_budget_spatial_film_k8_3dom --report-out run_reports/stage5_lodo_spfilm_vs_global.md --csv-out run_reports/stage5_lodo_spfilm_vs_global_cells.csv
```

These write new files; the committed Step 4 report and the hand-written
findings are not touched.

## 7. Mac: pull the reports and run folders (without weights)

Keep the quotes, or zsh expands the globs locally.

```bash
rsync -avz 'create:edward/spfilm/run_reports/s5_all_domains_spfilm_*' ~/Desktop/Projects/Research/SpFilm/code/spfilm/run_reports/
```

```bash
rsync -avz 'create:edward/spfilm/run_reports/stage5_lodo_spfilm_*' ~/Desktop/Projects/Research/SpFilm/code/spfilm/run_reports/
```

```bash
rsync -avz --exclude='*.pt' 'create:edward/spfilm/artifacts/runs/allsf_s4_*' ~/Desktop/Projects/Research/SpFilm/code/spfilm/artifacts/runs/
```

```bash
rsync -avz --exclude='*.pt' 'create:edward/spfilm/artifacts/runs/sfilm_s5_*' ~/Desktop/Projects/Research/SpFilm/code/spfilm/artifacts/runs/
```

Then write each report's findings section by hand, and the three-line message
for Pushpendra: result, conclusion, next step.
