---
id: P1-F01
group: F — Foundation & protocol
title: "Data partition and leakage audit"
tier: gate
status: planned
depends_on: []
blocks: [P1-F05, P1-B01, P1-B03, P1-D01, P1-G01]
paper_section: "4.1"
contribution: protocol
cost: {train_runs: 0, solver_runs: 0, effort: "~1 day"}
---

# P1-F01 — Data partition and leakage audit

## Question

Is there a clean, disjoint three-way partition of the pOnEllipse data — and
exactly what did the Year-1 evaluation measure on?

## Hypothesis *(pre-registered)*

`pOnEllipseTest.h5` is a genuinely disjoint held-out set drawn from the same
generator as the training file, so a clean three-way partition can be defined
**without sacrificing training data**. Separately, the Year-1 evaluation targets
(#4849, #4066) will be confirmed as training samples.

## Falsifier

The test file overlaps the training file — any test sample within
`|ΔRe| < 1` **and** `|Δb| < 1e-3` of a training sample. If so, the upstream test
file is unusable as a held-out set and a test partition must be carved out of
`pOnEllipseTrain.h5` instead, at a cost of ~10% of the training data.

## Method

1. Download `DatasetUrl.TEST_H5` (`src/graph_diffusion/data/pOnEllipse.py:80`) —
   it has never been fetched. Confirm `split="test"` loads end to end.
2. Extract design coordinates `(Re, H, a, b)` and node counts for all three files:
   train, test, AoA10.
3. **Cross-file nearest-neighbour audit.** For every test sample, find its nearest
   training sample in normalised design space and in K=8 DCT-Cp space. Compare the
   cross-file distance distribution against the within-train nearest-neighbour
   distribution — if test samples are no further from train than train samples are
   from each other, the "held-out" set is effectively interpolation.
4. Define the partition: test = `pOnEllipseTest.h5`; val = 10% of the training
   file under `torch.Generator().manual_seed(42)` to stay consistent with
   `GraphDataLoader`; train = remainder.
5. Freeze to `data/splits/paper1_partition.json` — explicit index lists, RNG seed,
   SHA-256 of every `.h5`, audit results. Commit it.
6. **Confirm the Year-1 leak.** Verify #4849 and #4066 land in train under this
   partition, and record it for the paper's honest-reporting note.

## Measurements

| Quantity | Purpose |
|---|---|
| `n_train`, `n_val`, `n_test`, `n_ood` | Reported in §3.5 and every `run_meta.json` |
| Duplicate/leak count at the stated tolerance | The pass/fail number |
| Cross-file vs within-train NN-distance distributions | Whether "held out" means anything here |
| Node-count histogram per file | Confirms 96 for train/test, 80 for AoA10 |
| Design-space coverage plot (Re × b), three partitions overlaid | Figure for §4.1 |

## Decision rule

- **Pass:** leak count = 0 → adopt `pOnEllipseTest.h5` as the test partition.
- **Fail:** leak count > 0 → carve a test partition from the training file by
  **stratified design-space split**, not random split, so the test set spans the
  (Re, b) plane rather than clustering. Record the change as a protocol amendment.
- **Either way**, if the cross-file NN distance is not materially larger than the
  within-train NN distance, state in §5 that "held-out" here means held-out
  *samples*, not held-out *regions* — and that G02/G03 carry the extrapolation
  claims.

## Consequence

- **Supported →** the top submission blocker is closed. §4.1 reports a clean
  partition; G01 replaces the Year-1 headline with a defensible number.
- **Falsified →** training set shrinks ~10%; S02's data-fraction curve tells you
  whether that costs anything measurable.

## Notes

The 15-case `eval_grid` solver gate is **unaffected** by the Year-1 leak — those
runs push dataset shapes through the solver with the model uninvolved. Do not
over-correct and re-run what was already sound; F03 re-runs it for protocol
consistency, not because it was wrong.

## Results

_Not yet run._
