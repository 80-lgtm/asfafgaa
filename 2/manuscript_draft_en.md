# Defect-Free Samples with Global Deformation in Shearographic Anomaly Detection: A Region-Disjoint Evaluation of a Lightweight Memory Bank

> English working draft for NDT&E International; not a submission-ready manuscript.
> The values are traceable to `outputs/region/region_report.json` and
> `outputs/region/uncertainty_report.json`. Author metadata, declarations, repository
> details, journal-specific formatting, and final author review remain unresolved.

## Abstract

Global deformation fringes in shearographic images can coexist with local defect responses. A nominal reference bank constructed only from defect-free images without global deformation may therefore flag a defect-free deformation state as anomalous. In addition, repeated acquisitions of the same measurement region can occur in different official data splits, complicating the interpretation of image-level metrics as performance on unseen physical regions. Using SADD v1.1.0, we evaluated a lightweight pipeline comprising a frozen ResNet-18, greedy k-center coreset selection, and nearest-neighbour distances, while keeping each recorded region within a single train, validation, or test split. One median-time frame was retained per `(region, type, split)` combination; 19 normal regions contributed additional clean or stripe state frames. With a 2,000-feature coreset, the state-level test AUROC was 0.956, faulty recall was 0.909 at the validation-derived threshold, and the normal false-positive rate was 0.125. Excluding Good_stripes yielded an AUROC of 0.843 and a Good_stripes false-positive rate of 0.875. This primary comparison also changed the number of state frames and the effective coreset size. A matched sensitivity analysis used 10 state frames and 1,960 features per bank, with five clean and five stripe frames in the mixed bank. Across 10 seeds, the mixed bank consistently yielded higher AUROC (0.915–0.968 versus 0.830–0.864) and lower Good_stripes false-positive rates (0.125–0.625 versus 0.875–1.000), with variable recall. A region-level cluster bootstrap produced wide intervals because only six normal test regions were available. We report a region-disjoint re-evaluation, nominal-bank comparisons, and weak bounding-box analysis, without claiming cross-component generalisation or deployment readiness.

**Keywords:** shearography; anomaly detection; nominal reference bank; coreset; region-disjoint evaluation; weakly annotated localisation

## 1. Introduction

Shearography supports non-destructive testing through interferometric fringe responses. SADD contains measurements of a polymer–rubber adhesive joint under controlled conditions, with Faulty, Good_clean, and Good_stripes samples. Good_stripes images contain no adhesive discontinuity but include global deformation used to generate phase jumps [1,2]. Whether the nominal reference bank covers this state is therefore a practical design question for normal-reference anomaly detection.

Previous shearographic research has already compared a training subset containing only undeformed defect-free samples with one that additionally includes globally deformed defect-free samples [2]. We therefore do not claim that including Good_stripes is a novel idea. Instead, we examine the detection and false-positive behaviour associated with this choice under a split that separates repeated measurement regions, and assess a simplified pipeline that requires no GPU training.

This work provides: (1) a region-disjoint manifest with an explicit account of the statistical hierarchy created by multiple normal states per region; (2) nominal-bank comparisons within a common lightweight nearest-neighbour pipeline; (3) a negative result for gradient-based post-hoc stripe suppression and an exploratory frequency-domain comparison; and (4) bounding-box hit analysis that is not equated with pixel-level segmentation accuracy.

## 2. Related Work

PatchCore uses a memory bank of nominal patch features and coreset compression for industrial anomaly detection [3]. PaDiM represents the normal class using patch embeddings from a pretrained CNN and multivariate Gaussian distributions [4]. EfficientAD uses a lightweight student–teacher approach for efficient anomaly detection [5]. These methods provide methodological context; we did not implement and run PaDiM or EfficientAD under the same SADD split and therefore make no numerical comparison with them.

Our implementation is PatchCore-inspired but is not a full reproduction of PatchCore. It uses only the ResNet-18 `layer3` grid, Euclidean nearest-neighbour distances, and a 95th-percentile image score, without reproducing the full backbone configuration, neighbourhood aggregation, spatial smoothing, or localisation processing of the original method.

The SADD article notes that repeated acquisitions from the same physical defect region may occur across official splits, which are not defect-instance-disjoint [1]. We therefore use a region-atomic re-evaluation. Since the addition of globally deformed nominal samples has been examined previously [2], our contribution is restricted to the present evaluation protocol, its comparisons, and weak-label analysis rather than a claim of general superiority.

## 3. Materials and Methods

### 3.1 Dataset, licence, and citations

We used SADD v1.1.0, available at DOI `10.5281/zenodo.18301911`. The local archive contains 10,173 image records in the three classes. Expert annotations are supplied as bounding boxes, CSV, and YOLO labels, not pixel masks [1]. The archive MD5 is `3b694539f8fa12bc12af2ee4f939f064`, matching the Zenodo record. The dataset article and the specific dataset version are cited separately.

### 3.2 State-frame selection and region-atomic splitting

The median-time frame was selected for each `(region, type, split)` combination, yielding 1,131 state-specific images from 1,103 unique recorded regions. The additional 28 entries arise from 19 normal regions with both Good_clean and Good_stripes states, rather than from copied faulty samples. All state entries of a region were retained within one split. Faulty entries retained their official split; normal regions spanning splits were assigned atomically using the predetermined priority `test > val > train`. The resulting train/validation/test region intersections were all zero.

This operation ensures that a recorded region does not cross splits. It does not imply that every scored image is an independent region. The primary metrics are calculated over state-specific images; region-level aggregation and cluster bootstrap are reported separately.

### 3.3 Features and lightweight reference bank

Grayscale images were replicated across three channels and resized directly to 224×224 pixels. A frozen ImageNet-1K v1 ResNet-18 [6] supplied the `layer3` output as a 14×14 grid of 256-dimensional features. All grid features from the training nominal images were compressed using greedy k-center selection with a random starting point and seed 42.

For a test grid feature \(f_i\), the local anomaly score is

\[
d_i = \min_{m \in \mathcal{M}}\lVert f_i-m\rVert_2,
\]

where \(\mathcal{M}\) is the coreset. The image score is the 95th percentile of local scores. The pipeline requires no GPU training; this does not establish industrial real-time performance or deployment readiness.

### 3.4 Nominal-bank comparisons and context variants

The full nominal bank includes training Good_clean and Good_stripes frames; the restricted bank contains only Good_clean. Gradient and frequency-domain contexts were exploratory adjustments. The value of alpha maximising validation AUROC was selected, with the smaller value preferred in a tie. The primary configuration selected alpha=0, disabling post-hoc context suppression. The operating threshold was determined using the Youden index on the validation set and then fixed for the test set.

### 3.5 Weak bounding-box analysis

For each faulty test image, the 14×14 anomaly map was mapped linearly to the original image coordinates. The map threshold was the 95th percentile of its 196 grid scores. A bounding box was counted as hit if any thresholded grid cell fell inside it. This measure is sensitive to the relative threshold, coarse grid, and single-cell rule. It is not IoU, Dice, pixel AUROC, or proof of localisation accuracy.

### 3.6 Region-level uncertainty and matched sensitivity analysis

For region-level analysis, each region was assigned its maximum state-image score and an any-positive decision rule. We resampled the 105 unique test regions using 10,000 cluster-bootstrap replicates (seed 20261031). Replicates containing only one class were skipped and counted.

For the matched control, both banks contained 10 state frames and an effective coreset of 1,960 features. The clean-only bank was sampled with replacement from clean frames; the mixed bank contained five clean and five stripe frames sampled with replacement. Ten fixed seeds were used. This is a sensitivity analysis over frame composition and coreset initialisation, not an independent causal experiment.

## 4. Results

### 4.1 Dataset sizes

| Split | Faulty | Good_clean | Good_stripes | State images |
|---|---:|---:|---:|---:|
| train | 886 | 10 | 9 | 905 |
| validation | 98 | 6 | 7 | 111 |
| test | 99 | 8 | 8 | 115 |

The test set contains 105 unique regions, including only six normal regions. Faulty samples dominate the test set. The validation set contains only 13 normal state-specific images, so alpha and threshold selection may be unstable.

### 4.2 Primary configuration and nominal-bank comparison

| Nominal bank/context | Validation AUROC | Test AUROC | Faulty recall | Normal FPR | Good_stripes FPR |
|---|---:|---:|---:|---:|---:|
| C+S / G (primary, alpha=0) | 0.980 | 0.956 | 0.909 | 0.125 | 0.250 |
| C+S / F (alpha=1) | 0.981 | 0.970 | 0.838 | 0.063 | 0.125 |
| C / G (alpha=0) | 0.782 | 0.843 | 0.949 | 0.438 | 0.875 |
| C / F (alpha=0) | 0.782 | 0.843 | 0.949 | 0.438 | 0.875 |

C denotes Good_clean, S Good_stripes, G gradient context, and F frequency-domain context. The requested maximum coreset size was 2,000 for all configurations. C/G and C/F are equivalent because both select alpha=0.

Under this deterministic split and implementation, including Good_stripes was accompanied by higher test AUROC and fewer Good_stripes false positives, but recall decreased from 0.949 to 0.909. Both banks cover the same 10 training normal regions. The clean-only bank has 10 state frames and 1,960 candidate patches, so its effective coreset is 1,960; the full bank has 19 state frames, 3,724 candidates, and an effective coreset of 2,000. This primary comparison therefore changes state composition, state-frame count, and effective coreset size together.

**Figure 1.** ROC curve of the primary configuration under the region-disjoint split. AUROC=0.956 over 115 state-specific test images (99 faulty, 16 normal; 105 unique regions). This is not a region-bootstrap confidence interval.

**Figure 2.** Validation and test AUROC for the nominal-bank and context configurations. The y-axis starts at 0.7; abbreviations are defined above.

**Figure 3.** Normal-sample false-positive rates at the validation-derived threshold. Good_stripes uses eight state-specific images and all normal samples use 16 state-specific images, from six unique normal regions. No significance claim is implied.

### 4.3 Context variants

The gradient context selected alpha=0 and provided no demonstrated gain. The frequency-domain context selected alpha=1, changing test AUROC from 0.956 to 0.970 and Good_stripes FPR from 0.250 to 0.125, while faulty recall decreased from 0.909 to 0.838. With few normal test samples and no dedicated uncertainty analysis for this exploratory configuration, these values do not establish superiority or stripe robustness.

### 4.4 Region-level uncertainty and matched control

The region-level point estimates were AUROC=0.894, faulty-region recall=0.909, normal-region any-FPR=0.333 (2/6), and Good_stripes-region any-FPR=0.333. With only six normal test regions, the 95% cluster-bootstrap intervals were [0.764, 0.983] for AUROC and [0, 0.80] for both normal and stripe any-FPR. Nineteen single-class AUROC replicates were skipped. The wide intervals preclude stable normal-sample performance claims.

Across the 10 matched-control seeds, the mixed bank consistently produced higher test AUROC (0.915–0.968 versus 0.830–0.864) and lower Good_stripes FPR (0.125–0.625 versus 0.875–1.000) than the clean-only bank. Recall varied with frame sampling and threshold selection. The direction was consistent with the primary comparison, but the magnitude was seed-dependent and does not establish causality.

### 4.5 Weak bounding-box analysis

The primary configuration hit 134 of 165 expert bounding boxes, giving a hit rate of 0.812. Figure 4 provides one qualitative example for coordinate inspection; it does not represent the full distribution or pixel-level localisation accuracy.

**Figure 4.** SADD test image `000161.png` (region `112503`): input image, enlarged 14×14 anomaly map, and expert bounding box, using the primary alpha=0 configuration. This qualitative panel does not demonstrate sample representativeness or segmentation accuracy.

## 5. Discussion

The application relevance of this study is not a universally optimal detector. Rather, it shows how nominal-bank construction and the treatment of repeated measurement regions affect observed metrics and their interpretation. A clean-only nominal bank may not cover an expected defect-free global-deformation state. Nevertheless, the current experiments do not isolate all effects of state coverage, frame count, coreset size, and split selection, and provide no cross-device or cross-component evidence.

Region-atomic splitting reduces same-region cross-split dependence but does not establish specimen-level independence. More normal regions are needed for useful uncertainty estimates. SADD's controlled materials, loading protocol, and artificial interfacial discontinuities cannot represent every naturally occurring kissing bond or industrial component [1].

## 6. Limitations

- One controlled dataset was used, without cross-dataset, material, device, or industrial validation.
- Only 16 normal state images and six unique normal regions were available for testing; the Good_stripes FPR of 0.25 represents two errors among eight state images.
- State-level and region-level metrics answer different questions and must not be interchanged. The region-level normal any-FPR interval is very wide ([0, 0.80]).
- The primary comparison did not match state-frame count or effective coreset size; the matched sensitivity analysis addressed these factors only through resampling.
- No PaDiM, EfficientAD, or complete PatchCore baseline was run under the same protocol.
- Bounding-box hit rate is not pixel-level localisation or defect-size measurement.
- CPU-only execution does not establish end-to-end latency, peak-memory, or online-deployment performance.

## 7. Conclusions

In this SADD region-disjoint re-evaluation, including Good_stripes in a lightweight nominal bank was associated with higher test AUROC and fewer Good_stripes false positives, with a trade-off in recall. Matched resampling gave the same direction across 10 seeds, but normal-region uncertainty remained large. These findings motivate further evaluation of nominal-state coverage and split design rather than a claim of a new architecture or general performance advantage. The negative gradient-suppression result and the frequency-domain trade-off are retained. Future work should expand normal-region coverage and add baselines under the same protocol.

## Data and Code Availability

SADD v1.1.0 is available at https://doi.org/10.5281/zenodo.18301911. The derived region manifest, analysis code, reports, and figure-generation scripts are planned for deposition in `AUTHOR_INPUT_NEEDED: repository name, stable link, version, and DOI after actual deposition`. No project repository or DOI is claimed at this draft stage.

## Author Declarations

**Corresponding author:** Yongkang Zhang, Nankai University, 2413614@mail.nankai.edu.cn, ORCID 0009-0008-6188-1469

**Author contributions (CRediT):** Yongkang Zhang — Conceptualization, Data curation, Formal analysis, Investigation, Methodology, Project administration, Software, Validation, Visualization, Writing – original draft, Writing – review & editing.

**Competing interests:** The author declares no competing interests.

**Funding:** This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

**Acknowledgements:** None.

This study analysed a public dataset and did not involve new human participants, animal experiments, or new optical data acquisition.

## References

[1] Krause, J., Schuler, N., Schuth, M., von Freymann, G. Shearographic anomaly detection dataset (SADD). *Machine Vision and Applications* 38, 12 (2027; online 26 October 2026). https://doi.org/10.1007/s00138-026-01948-0

[2] Plassmann, J., Schuler, N., von Freymann, G., Schuth, M. Unsupervised Learning for Industrial Defect Detection: A Case Study on Shearographic Data. *Artificial Intelligence XLII*, LNAI 16302, 316–329 (2025). https://doi.org/10.1007/978-3-032-11442-6_22

[3] Roth, K., Pemula, L., Zepeda, J., Schölkopf, B., Brox, T., Gehler, P. Towards Total Recall in Industrial Anomaly Detection. CVPR 2022. https://doi.org/10.1109/CVPR52688.2022.01392

[4] Defard, T., Setkov, A., Loesch, A., Audigier, R. PaDiM: A Patch Distribution Modeling Framework for Anomaly Detection and Localization. *Pattern Recognition. ICPR International Workshops and Challenges*, LNCS 12664, 475–489 (2021). https://doi.org/10.1007/978-3-030-68799-1_35

[5] Batzner, K., Heckler, L., König, R. EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies. WACV 2024. https://doi.org/10.1109/WACV57701.2024.00020

[6] He, K., Zhang, X., Ren, S., Sun, J. Deep Residual Learning for Image Recognition. CVPR 2016, 770–778. https://doi.org/10.1109/CVPR.2016.90

[7] Plaßmann, J., Schuler, N., von Freymann, G., Schuth, M. Shearographic Anomaly Detection Dataset (SADD), v1.1.0. Zenodo (2026). https://doi.org/10.5281/zenodo.18301911
