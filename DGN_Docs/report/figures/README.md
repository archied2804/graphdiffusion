# Report figures (collected copy)

A convenience copy of **every figure embedded in the first-year report**
(`docs/report/year1_report.md` / `docs/report/latex/`). The canonical source for
each figure is `docs/assets/`; this folder is a flat, self-contained snapshot for
sharing or dropping into slides. PNG for all; PDF (vector) for the
matplotlib-generated plots.

> Regenerated copies: run the generators below, then re-copy from `docs/assets/`.
> This folder is **not** auto-synced — it is a manual snapshot.

## How the figures are produced

All report figures share one house style (`scripts/_report_style.py`): DejaVu
Serif to match the report body, consistent sizes/palette, and no redundant
in-plot titles (the caption is the title).

| Generator | Figures |
|-----------|---------|
| `scripts/make_report_results_figures.py` | `EXP-043_cp_validation`, `EXP-043_cp_stability`, `EXP-043_eval_grid_coverage`, `EXP-041_cp_inverse_design`, `EXP-043_ood_results_summary`, `EXP-043_ood_cp_manifold`, `EXP-042_creep_ladder`, `EXP-041_dps_sweep` |
| `scripts/make_report_dataset_figures.py` | `report_dataset_example`, `report_design_space`, `report_cp_envelope`, `report_generated_shapes` |
| `scripts/make_report_methodology_figures.py` | `report_conditioning_pipeline`, `report_diffusion_process`, `report_dct_encoding` |
| `scripts/make_report_architecture_figure.py` | `report_architecture` |
| `scripts/make_exp04x_doc_figures.py` | `EXP-043_domain_schematic` |
| external (ParaView) | `report_paraview_velocity_wake`, `report_paraview_pressure` |

## Regenerate everything

```bash
python scripts/make_report_methodology_figures.py
python scripts/make_report_architecture_figure.py
python scripts/make_exp04x_doc_figures.py
python scripts/make_report_dataset_figures.py
python scripts/make_report_results_figures.py
# then refresh this snapshot:
#   cp docs/assets/<name>.{png,pdf} docs/report/figures/
```

The two ParaView fields are rendered manually from the OpenLB VTK output
(`cfd/ellipse2d/runs/<case>/tmp/vtkData/`) and are not script-generated.
