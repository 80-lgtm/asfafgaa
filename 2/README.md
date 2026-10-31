# SADD Region-Disjoint Anomaly-Detection Prototype

This workspace contains a CPU-only, simplified coreset nearest-neighbour detector and
region-disjoint evaluation on SADD v1.1.0. It is not a full PatchCore reproduction.

## Status

- The exploratory stripe-suppression module is not an active contribution: the main
  configuration selects alpha=0. Frequency-domain results are supplementary only.
- Current main report: `outputs/region/region_report.json`; test AUROC=0.956,
  faulty recall=0.909, and bbox hit rate=0.812 under the documented protocol.
- The 115 test state-specific images correspond to 105 unique recorded regions. Metrics
  are not yet accompanied by unique-region bootstrap confidence intervals.
- Chinese working draft: `manuscript_draft_cn.md`; it is not a submission-ready paper.
- Reproduction notes: `REPRODUCIBILITY.md`; figure provenance: `FIGURE_TRACEABILITY_CN.md`.

## Tests

Use the existing `perovskite` Conda environment:

```powershell
conda run -n perovskite python -m unittest discover -s tests -v
```

## Main evaluation

The already downloaded local archive and existing feature cache are required. Do not run
dependency installation or download data without reviewing the reproduction notes.

```powershell
conda run -n perovskite python -m scripts.build_region_split
conda run -n perovskite python -m scripts.run_region_evaluation
conda run -n perovskite python -m scripts.make_region_figures
```

Historical pilot/L2 outputs use different splits and must not be mixed with the main report.
