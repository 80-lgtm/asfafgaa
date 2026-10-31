# SAGA-AD MVP Implementation Plan

**Goal:** Build a CPU-only, testable skeleton for a 200-image SADD pilot.

**Architecture:** The prototype separates dataset manifests, feature extraction, normal-memory construction, stripe-aware scoring, and evaluation. SADD's unknown archive layout is isolated behind a manifest contract, so only the manifest builder should change after the archive is inspected.

**Tech Stack:** Python 3.11, NumPy, Pillow, pandas, PyTorch/torchvision, scikit-learn; standard-library `unittest` for dependency-free tests.

## Global Constraints

- Do not download SADD or model weights during skeleton implementation.
- Do not claim experimental performance before a verified pilot run.
- Keep CPU-only execution as the default path.
- Use fixed random seeds and deterministic pilot splits.
- Never treat SADD bounding boxes as pixel masks.

## Review Focus

- Empty or malformed manifests must fail explicitly rather than silently produce zero scores.
- Deterministic splits must survive reordering and repeated runs.
- Feature backends must remain optional until dependencies are installed.
- Stripe consistency must reduce stripe-like local scores without changing zero anomaly scores.
- Evaluation must reject arrays with inconsistent lengths or missing labels.

## Task 1: Core Manifest and Scoring Logic

- [x] Create core modules.
- [x] Verify all 8 tests pass.

## Task 2: ResNet18 Feature Backend and CLI
- Create `src/config.py`, `src/dataset.py`, `src/memory_bank.py`, `src/stripe_context.py`, `src/scoring.py`, and `src/evaluation.py`.
- Test manifest validation, deterministic splitting, nearest-memory scoring, stripe consistency, and metric computation with `unittest`.
- Do not include file discovery from the unknown SADD layout in this task; provide a manifest-builder interface only.

## Task 2: ResNet18 Feature Backend and CLI

- [x] Add lazy ResNet18 backend.
- [x] Add CLI and pilot configuration.
- [x] Verify all 12 tests pass.

## Task 3: Environment and Data Acquisition

- Add `src/features.py` with a lazy PyTorch/torchvision backend.
- Add `main.py` with a manifest-driven smoke/pilot entry point.
- Tests should cover backend selection and CLI argument parsing without downloading weights.

## Task 3: Environment and Data Acquisition

- Install only missing verified dependencies in the existing `perovskite` environment.
- Download `SADD_v1.1.zip` from Zenodo, verify MD5, and inspect the archive layout.
- Adapt the SADD manifest builder to the actual directories and annotations.

## Task 4: Pilot Execution and Review

- Build the fixed 200-image split.
- Run a 10-image smoke test before the full pilot.
- Preserve logs, anomaly maps, configuration, raw metrics, and environment details.
