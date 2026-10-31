# M7 Finalization Repair Plan

**Goal:** Produce a traceable Chinese working package and an honest M7 readiness assessment, without treating it as a submission-ready manuscript.

**Architecture:** Keep the verified region-level report as the numerical source of truth. Render results through the existing Python workflow, record image-to-coordinate provenance, and keep journal requirements and publication status explicitly unresolved.

**Constraints:** No downloads, dependency installation, changes outside the workspace, invented citations, or unapproved method-diagram rendering. Do not re-run or alter the main experimental results.

## Tasks

- [x] Reproduce the four result figures from `outputs/region/region_report.json`, the region manifest, cached features, and the SADD archive; inspect image dimensions, bbox coordinates, heatmap alignment, labels, and exports.
- [x] Add a reproducible figure traceability record and a manuscript-level figure map with legends and explicit user-approval status.
- [x] Verify core bibliographic records and SADD metadata from primary/official web pages only. Record unresolved citation and licence details rather than guessing.
- [x] Add the Chinese-English terminology ledger and translation-risk guidance; revise overstrong causal, deployment, and generalization wording in the Chinese draft.
- [ ] Update the submission checklist and project-state file; run tests and an evidence audit, then apply M7 acceptance criteria and provide the appropriate repair card.

## Review Focus

- The 1,131 selected images represent 1,103 regions; multi-type regions must not be misreported as accidental duplicates or cross-split leakage.
- The bbox example is a visual coordinate check, not a pixel-localization benchmark.
- The 8 stripe state images (6 unique normal test regions) make FPR uncertainty substantial; no superiority or equivalence claim is justified without uncertainty analysis.
- All rendered figures remain blocked from final manuscript insertion until the user reviews them.
- A target journal has not been selected, so formatting and submission requirements cannot pass final verification.
