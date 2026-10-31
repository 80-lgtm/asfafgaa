# SAGA-AD / SADD region-level evaluation — reproducibility notes

This file records the environment and commands that produced the historical L2
evaluation, the coreset ablation, and the corrected region-disjoint
numbers. No result in this document should be treated as stronger than the
actual files under `outputs/`.

## 1. Data

- Dataset: SADD v1.1.0 (Shearographic Anomaly Detection Dataset)
- DOI: 10.5281/zenodo.18301911
- Archive: `data/raw/SADD_v1.1.zip`
- Size: 4,964,010,557 bytes
- MD5: `3b694539f8fa12bc12af2ee4f939f064`
- License: CC BY 4.0

## 2. Environment

- OS: Windows 10 (platform string recorded in `outputs/l2/l2_report.json`)
- Python: 3.11.17 (Conda env `perovskite`)
- PyTorch: 2.14.1+cpu
- torchvision: 0.29.2+cpu
- NumPy: 2.4.6
- pandas: 3.0.6
- scikit-learn: 1.9.1
- Pillow: 13.0.0
- SciPy: 1.17.1
- matplotlib: 3.11.2
- PyYAML: installed (used by `src/config.py`)

## 3. Region-level manifest

Run:

```powershell
conda run -n perovskite python -m scripts.build_full_manifest
```

The manifest is written to `data/processed_full/manifest.csv`. It selects one
median-time frame per `(region, type, split)`. Known issue: 1,131 rows come from
1,103 unique `region` values because 19 regions appear under more than one
`type`. Six regions appear in both `train` and `test` under different types.
The corrected region-disjoint split is built separately as described below.

## 4. Main L2 evaluation

```powershell
conda run -n perovskite python -m scripts.run_l2_evaluation
```

Hyperparameters:

- backbone: frozen ResNet-18, ImageNet-1k v1 weights
- feature layer: `layer3` output, reshaped to a 14x14 grid of 256-d features
- image size: 224x224
- memory bank: train normals (Good_clean + Good_stripes)
- coreset: greedy k-center, k=2000, seed=42
- patch anomaly: Euclidean nearest-neighbor distance
- image anomaly: 95th percentile of patch scores
- alpha candidates: 0, 0.5, 1, 2, 4, 8 (main) / 0, 1, 2, 4 (ablation)
- threshold: Youden index on the validation split
- outputs: `outputs/l2/l2_report.json`, `outputs/l2/scores.csv`,
  `outputs/l2/figure_roc_ablation.png/pdf`

## 5. Coreset and context ablation

```powershell
conda run -n perovskite python -m scripts.run_stripe_ablation
```

This compares:

- memory bank with Good_stripes vs. without Good_stripes
- gradient-domain vs. FFT-domain stripe consistency
- alpha in {0, 1, 2, 4}

The current run uses coreset k=2000 and is saved to
`outputs/ablation/ablation_report.json`. The FFT context is cached at
`outputs/ablation/fft_context.npz`.

## 6. Corrected region-disjoint main evaluation

```powershell
conda run -n perovskite python -m scripts.build_region_split
conda run -n perovskite python -m scripts.run_region_evaluation
conda run -n perovskite python -m scripts.make_region_figures
```

- Input: `data/processed_full/manifest.csv`; output:
  `data/processed_region/manifest.csv` and `region_split_summary.json`.
- Faulty entries retain their official split. Normal regions spanning splits are moved
  atomically using priority test > val > train. All 18 changed entries are normal;
  no Faulty split changes.
- All configurations use coreset k=2000, seed=42, the 95th percentile image score, and
  validation Youden thresholds. Gradient alpha candidates are 0, 1, 2, 4, 8; frequency
  candidates are 0, 1, 2, 4. Alpha maximizes validation AUROC with the smaller value
  preferred for ties.
- k is capped by the available candidate patches: the clean-only bank has 1,960 patches
  and therefore 1,960 coreset members; the clean+stripes bank has 3,724 candidates and
  2,000 coreset members. Both banks cover the same 10 training normal regions.
- Main numerical output: `outputs/region/region_report.json`; figure source and QA:
  `FIGURE_TRACEABILITY_CN.md`.
- Existing `outputs/l2/cache/<image>.npz` features are reused by image identity. The
  historical directory name does not mean that the old official split is used.
- No per-file cache checksum manifest is currently available.

## 7. Tests

```powershell
conda run -n perovskite python -m unittest discover -s tests -v
```

The last verified test count is recorded in the M7 review/state documents; rerun the
command rather than relying on this prose after any code change.

## 7.5 Region-level uncertainty and matched control

```powershell
conda run -n perovskite python -m scripts.run_region_uncertainty
```

- Reuses the existing feature cache; does not re-extract features.
- Region-level metrics: max image score per unique region; any-positive rule for FPR
  and recall; cluster bootstrap (10,000 replicates, seed 20261031) over the 105 test
  regions. Undefined replicates (single-class samples) are skipped and counted.
- Matched control: each bank has 10 state frames and effective coreset k=1,960;
  clean-only versus 5 clean plus 5 stripe frames. Both banks are sampled with
  replacement over the available state frames, with 10 fixed seeds.
- To update only this control without recomputing the region bootstrap:
  `conda run -n perovskite python -m scripts.run_region_uncertainty --matched-only`.
- Output: `outputs/region/uncertainty_report.json`.
- Region-level point estimates (main configuration): AUROC=0.894, faulty region
  recall=0.909, normal region any-FPR=0.333 (2/6), stripe region any-FPR=0.333.
  These aggregate differently from the state-level headline metrics and must not be
  interchanged with them.

## 8. Evidence boundaries

- The corrected test set has 8 Good_stripes state-specific images (FPR=2/8 in the main
  configuration), 16 normal state-specific images, and 6 unique normal regions.
- The full test set has 115 state-specific images and 105 unique regions. Nineteen normal
  regions have both Good_clean and Good_stripes states; image/state metrics are not yet
  unique-region statistical inference.
- Bounding boxes are weak labels, not pixel masks.
- All reported AUROC values are computed at region level after median-time
  de-duplication per `(region,type,split)`, with regions kept within one split. They
  are not comparable to image-level random splits in other papers.
- The with/without-stripes comparison changes the number of normal state frames
  (19 versus 10) and effective coreset size (2,000 versus 1,960), not just a controlled
  diversity variable; it does not add new training normal region identities.
- The current main report has no confidence intervals or repeated-split uncertainty.
