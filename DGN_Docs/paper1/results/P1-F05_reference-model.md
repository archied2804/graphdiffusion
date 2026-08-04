---
id: P1-F05
spec: "../specs/P1-F05_reference-model.md"
title: "Reference model under the clean protocol"
date_run: 2026-08-03
date_reported: 2026-08-04
status: complete
verdict: supported (training half)
git_sha: 1030982
---

# P1-F05 — results

> **Spec:** [[../specs/P1-F05_reference-model]] · **Verdict:** ✅ **training half
> supported.** The held-out grading half is deferred to
> [[../specs/P1-G01_holdout-inverse-design]] by design.

## 1. Verdict against the pre-registered decision rule

Pass criterion was *"val loss within the seed CI of EXP-022's, validity ≥ 95%"*.
Five seeds trained the full 1000 epochs on the frozen partition and converged
consistently. **P1-REF is adopted as the reference model.**

The spec's second prediction — held-out ρ ∈ [0.88, 0.94] — is **not tested here**.
Grading is G01's job, deliberately, so the reference experiment cannot quietly
become the headline result before the baselines report.

## 2. Headline numbers

| Seed | Epochs | best_val | at epoch | final_train |
|:--:|--:|--:|--:|--:|
| 0 | 1000 | 1.833e-4 | 921 | 4.257e-4 |
| 1 | 1000 | 1.557e-4 | 986 | 3.794e-4 |
| 2 | 1000 | 1.500e-4 | 990 | 3.773e-4 |
| 3 | 1000 | 9.592e-5 | 952 | 1.687e-4 |
| 4 | 1000 | 1.570e-4 | 989 | 3.386e-4 |

**best_val = 1.484e-4, 95% CI [1.086e-4, 1.882e-4], CV = 21.6%, n = 5.**

| | |
|---|---|
| Partition | 5131 train / 570 val, `paper1_partition.json` |
| Score-network parameters | **147,585** (165,513 including the pressure head) |
| Wall clock | 70–78 min per seed; 7.5 h total |

**This CV is the model-side noise scale.** Every A/B/C/D/S comparison must clear
it as well as F04's instrument-side floor. A 21.6% spread in val loss across seeds
is large enough that single-seed model comparisons — which is what the entire
Year-1 series ran — cannot distinguish recipes.

## 3. Protocol deviation — attempt 1 discarded

**Dated 2026-08-03.** The first attempt used `early_stopping_patience: 100`,
inherited from EXP-022. Seeds received **unequal training budgets**: seed 0 stopped
at epoch 281 (best at 181) while seeds 1–3 ran the full 1000 and never triggered
it. Seed 0's best_val was **1.44e-3**, an order of magnitude worse, and the spread
across seeds was **14.6×**.

Reported as-is that would have been recorded as seed variance. It was a budget
artefact.

Fixed by raising patience above the epoch budget so it can never fire —
deliberately not `null`, because in `train.py` the best-checkpoint save sits inside
the early-stopping branch and would be disabled with it. Applied uniformly, and
not chosen for its effect on results: unequal budgets are a defect whichever way
they fall.

**Effect:** seed spread fell from 14.6× to **1.91×**, and seed 0 moved from
1.44e-3 to 1.83e-4, in line with the others. Attempt 1 is preserved at
`outputs/paper1/_archive_P1-F05_attempt1_earlystop/`.

## 4. Observation — the epoch budget may be under-specified

Every seed reached its best validation loss at **epoch 921–990 of 1000**. None had
plateaued. The 1000-epoch budget, inherited from EXP-022, is very likely cutting
training short.

**Logged, not acted on.** Changing the reference recipe to chase this would break
the "reproduce EXP-022" premise and invalidate every comparison built on P1-REF.
It is [[../specs/P1-S03_diffusion-schedule]]'s question. If S03 shows a longer
budget helps materially, that is a finding to report — not a reason to
retrospectively re-run the suite.

## 5. Metadata correction — the `git_dirty` flag

All five runs recorded `git_dirty: true` and were flagged unusable by the status
board. **This was a defect in the metadata, not in the runs.**

Verified: seven commits landed during the run window, and **none touched anything
`train.py` imports** — they were `postprocessing/`, `tests/`, `scripts/` and docs.
`train.py` does not import `postprocessing` at all. The training code path was
unchanged from launch to completion.

Two real defects in the `run_meta.json` implementation, both now fixed:

1. Git state was captured at the **end** of the run, so a run was attributed to
   whatever had been committed by then rather than the commit it executed.
2. The dirty check was **repo-wide**, so an Obsidian vault autosave during a
   7-hour run marked it unusable.

`git_dirty` is now evaluated at process start over code paths only (`train.py`,
`src/`, `configs/`, `data/splits/`), with the full repo-wide list retained as
`git_dirty_files`. See [[../PROTOCOL]] §7, amended 2026-08-04.

**These five runs are retained**, on the verifiable grounds that no imported code
changed during the window. The next reference-class run will carry correct
metadata natively.

## 6. Reproduction

| | |
|---|---|
| Config | `configs/paper1/P1-F05_reference.yaml` (commit `1030982`) |
| Outputs | `outputs/paper1/P1-F05_reference/seed{0..4}/` |
| Partition | `data/splits/paper1_partition.json` |
| Command | `scripts/paper1/run_p1_f05.sh` |

## 7. What this changes

**In the paper**
- §3 states the reference recipe, 5 seeds, and the 21.6% model-side CV.
- §4.1 reports the seed spread as the scale against which every ablation is read.

**In the plan**
- **P1-REF is frozen.** A01–A04, B02, B04, C01–C04, D02, G01–G04, S01–S03 all
  compare against these checkpoints and must reuse seeds 0–4.
- The 21.6% CV means **single-seed comparisons are not interpretable** — this
  retrospectively explains why the Year-1 EXP-02x recipe sweep produced no clear
  winner. Worth one sentence in §5.
- G01 can now run as soon as F03/F04 qualify the grading pipeline.
