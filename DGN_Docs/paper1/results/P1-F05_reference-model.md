---
id: P1-F05
spec: "../specs/P1-F05_reference-model.md"
title: "Reference model under the clean protocol — attempt 3"
date_run: 2026-08-06
date_reported: 2026-08-06
status: complete
verdict: supported (training half)
supersedes: P1-F05_reference-model.md
git_sha: 5ec4fe4
---

# P1-F05 — results, attempt 3

> **Spec:** [[../specs/P1-F05_reference-model]] · **Verdict:** ✅ **training half
> supported.** Five seeds, full 1000 epochs, on the frozen partition and the
> **corrected conditioning descriptor** from
> [[P1-D01_conditioning-invariance]] §6.1.
>
> Supersedes [[superseded/P1-F05_reference-model_attempt1-2]] (attempt 2), which is retained as the
> record of the legacy-conditioning run and archived at
> `outputs/paper1/_archive_P1-F05_attempt2_orthonorm/`.

## 1. Why there is a third attempt

| Attempt | Discarded because | Record |
|:--:|---|---|
| 1 | `early_stopping_patience: 100` gave seeds unequal training budgets | archived |
| 2 | Conditioned on the `sqrt(N)`-scaling, spacing-blind descriptor that [[P1-D01_conditioning-invariance]] falsified | [[superseded/P1-F05_reference-model_attempt1-2]] |
| **3** | — | **this file; P1-REF** |

Attempt 3 is not a re-seeding of attempt 2. The conditioning vector itself
changed, so the two are **different methods**, not different runs (see §4).

## 2. Headline numbers

| Seed | best_val | at epoch | final_train | final_val |
|:--:|--:|--:|--:|--:|
| 0 | 1.558e−5 | 921 | 5.851e−5 | 5.921e−5 |
| 1 | 1.817e−5 | 986 | 6.403e−5 | 4.615e−5 |
| 2 | 1.152e−5 | 991 | 2.605e−5 | 2.242e−5 |
| 3 | **3.706e−5** | 919 | 7.101e−5 | 7.630e−5 |
| 4 | 1.786e−5 | 973 | 6.018e−5 | 8.563e−5 |

**best_val = 2.004e−5, CV = 49.3%, spread 3.22×, n = 5.**

| | |
|---|---|
| Partition | 5131 train / 570 val, `paper1_partition.json` |
| Conditioning | `cond_norm: forward`, K = 8 |
| Wall clock | ~105 min per seed; 7.0 h total |

## 3. Finding — seed variance roughly doubled, driven by one outlier

CV rose from **21.6%** (attempt 2) to **49.3%**. That is the opposite of what a
cleaner conditioning signal might be expected to give, and it needs stating
carefully because two different things could produce it.

**It is one seed.** Excluding seed 3:

| | mean | CV |
|---|--:|--:|
| all five | 2.004e−5 | **49.3%** |
| excluding seed 3 | 1.578e−5 | **19.4%** |

Seed 3 is **2.35×** the mean of the other four. Excluding it recovers a CV of
19.4%, statistically indistinguishable from attempt 2's 21.6%.

**There is no pre-registered basis for excluding it**, and it is not excluded
here — the headline CV is 49.3%. But the honest description is "four seeds
agree to 19% and one is 2.4× worse", not "variance doubled uniformly". That
distinction matters for [[../specs/P1-S01_capacity]] and every downstream
comparison, which should treat 49.3% as the operative noise scale while
recognising it is outlier-driven.

Seed 3 is not otherwise anomalous: it reached its best at epoch 919, in line
with the others, and its final train loss (7.101e−5) is only ~20% above the
group. It is a worse optimum, not a diverged run.

## 4. ⚠️ The loss values are not comparable across attempts

Attempt 2's mean was 1.484e−4; attempt 3's is 2.004e−5, **7.4× lower**. This is
**not** evidence that the corrected conditioning trains better.

The training objective is a sum of the diffusion noise-prediction MSE and an
auxiliary pressure term weighted by `lambda_pressure: 0.1`. That auxiliary term
regresses the conditioning vector, whose magnitude fell by roughly 7×
([[P1-D01_conditioning-invariance]] §6.1: `|cond|` median 2.50–5.10 → 0.376–0.594).
Its squared contribution therefore fell by roughly 50×, and the total loss with
it.

**Only the total loss is logged** (`loss_log.json` carries `train_loss` and
`val_loss` and nothing else), so the two terms cannot be separated after the
fact. Consequences:

- **No cross-attempt loss comparison is valid**, in either direction.
- CV is scale-invariant under a pure rescaling, but *not* under a change in the
  mixture of two differently-scaled terms. So even the CV comparison in §3 is
  weaker than it looks, and the 21.6% → 49.3% change cannot be attributed to
  the conditioning with confidence.
- **Recommended change:** log the two loss components separately. It costs
  nothing and would have made this section unnecessary. Logged as a protocol
  improvement, not applied retroactively.

The only defensible cross-attempt comparison is on **graded** quantities — ρ,
`a`, validity — which is [[../specs/P1-G01_holdout-inverse-design]]'s job, not
this experiment's.

## 5. Observation — the epoch budget is still under-specified

Best validation loss occurred at epoch **919–991 of 1000** for all five seeds,
exactly as in attempt 2. None had plateaued.

Unchanged from attempt 2's finding, and still not acted on: changing the budget
would break the "reproduce EXP-022" premise. It remains
[[../specs/P1-S03_diffusion-schedule]]'s question. That it reproduced across a
conditioning change makes it more clearly a property of the recipe than of the
run.

## 6. What this changes

**In the paper**
- §3 states the reference recipe, 5 seeds, and the corrected descriptor.
- §4.1 reports 49.3% CV as the model-side noise scale, with the outlier
  structure stated.

**In the plan**
- **P1-REF is these checkpoints.** A01–A04, B02, B04, C01–C04, D02, G01–G04,
  S01–S03 compare against them and must reuse seeds 0–4.
- **B01, D02 and G03 are unblocked** — they were waiting on this.
- A 49.3% CV means single-seed comparisons remain uninterpretable, and
  three-seed comparisons are marginal. S01/S02/S03 should budget 5.
- Add per-component loss logging before the next training experiment.

## 7. Reproduction

| | |
|---|---|
| Config | `configs/paper1/P1-F05_reference.yaml` (`cond_norm: forward`) |
| Outputs | `outputs/paper1/P1-F05_reference/seed{0..4}/` |
| Command | `scripts/paper1/run_p1_f05.sh` |
| Partition | `data/splits/paper1_partition.json` |

**Protocol deviation, dated 2026-08-06.** The first launch of attempt 3 crashed
immediately with `edge_attr = None`. Cause: a dataset cache built during a
verification step *without* the `pre_transform`, which PyG then reused for
training — it warned (`The pre_transform argument differs...`) and proceeded.
The cache was deleted, rebuilt through `train.py`'s own code path, and verified
(`edge_attr (720, 2)` present) before relaunching. No results were affected;
the cost was one overnight slot.
