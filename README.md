# Knee MRI Abnormality Detection with Multilingual Weak Supervision

An ongoing machine-learning project exploring how multilingual radiology reports can provide training supervision for knee MRI abnormality detection when official labels are scarce.

The pipeline combines **Qwen-based report labeling**, **fold-specific soft targets**, and **frozen ResNet-18 image features**. It is built around the [RSNA Knee Abnormality Detection competition](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection).

**Status:** Report extraction and MRI feature extraction are complete. A classifier forward/backward training test has passed; full classifier training, MRI validation results, and competition submission remain in progress.

## The problem

The training data contains **4,407 MRI studies and 24,371 imaging series**, but only **58 studies have complete official labels**. The remaining 4,349 studies include radiology reports that can supply noisy supervision.

Each study may contain several acquisitions across sagittal, coronal, and axial planes. The task is to predict 12 study-level targets:

`ACL`, `MCL`, `Medial Meniscus`, `Lateral Meniscus`, `Medial OA`, `Lateral OA`, `PF OA`, `Effusion`, `Synovitis`, `Baker's`, `Contusion`, and `Fracture`.

Reports are used to develop training labels. The planned inference pipeline will produce predictions from MRI images without requiring report text.

## Approach

### 1. Extract supervision from reports

[01_report_labeling.ipynb](01_report_labeling.ipynb) audits label availability, report text, and imaging metadata. Language detection identifies nine language groups: English, Spanish, Turkish, Croatian, Greek, German, Bulgarian, Dutch, and French.

Qwen2.5-3B-Instruct extracts a `0`, `1`, or `null` state for each target. The prompt addresses negation, uncertainty, clinical history, and anatomical compartments. Unknown findings remain distinct from explicit negatives.

The notebook evaluates report states against official labels, estimates preliminary state-to-probability mappings, and runs batched extraction with checkpoints and error logging.

### 2. Build training targets without using held-out labels

[02_mri_feature_extraction.ipynb](02_mri_feature_extraction.ipynb) joins the completed report states to official labels and creates three development-validation folds. Normalized identical report text stays within one group; the prompt-development study is excluded from validation.

For the active fold, soft targets are estimated using only officially labeled training studies, with shrinkage toward that training subset's prevalence. Official labels retain full weight. Unknown weak states receive zero weight in this baseline.

The earlier all-study calibration table is exploratory; fold-specific calibration is rebuilt before MRI training to avoid leaking held-out labels.

### 3. Convert multi-plane MRI into study features

The image pipeline:

- Selects one acquisition per plane, preferring fluid-sensitive and then fat-suppressed series.
- Orders slices using spatial metadata, with an explicit instance-number fallback.
- Samples 16 slices per plane, applies intensity normalization, and resizes/pads to 224 × 224 pixels.
- Encodes slices with a frozen ImageNet-pretrained ResNet-18.
- Averages slice features within each plane and concatenates the three plane vectors into a **1,536-dimensional study representation**.

Feature checkpoints include study IDs, completion flags, configuration, and errors. A small classifier test separately normalizes official and weak losses.

## Recorded results

These figures come from the saved notebook runs, not a leaderboard submission.

| Measurement | Recorded result |
| --- | --- |
| Reports processed | 4,407 / 4,407 |
| Report extraction parsing failures | 0 |
| Report-extractor evaluation set | 57 officially labeled studies |
| Non-null label coverage on that set | 93.71% |
| Agreement with official labels among covered labels | 80.34% |
| Studies with complete MRI features | 4,405 / 4,407 |
| Feature dimensions per study | 1,536 |

**The 80.34% figure measures report-label extraction agreement, not MRI diagnostic accuracy.** The evaluation set is small and has been inspected during development. Later comparison of the full batched extraction against the saved sequential evaluation found four differing states among 684 label entries; the second notebook uses the completed extraction consistently.

Two unlabelled studies each contain one unreadable DICOM slice. Their feature rows remain flagged incomplete; the downstream alignment cell excludes them rather than training on placeholder vectors. All 58 officially labeled studies have complete features.

## Running the notebooks

The notebooks target Kaggle's Python environment and require access to the competition data.

1. Attach the competition data, enable a GPU and Internet access for model downloads, and run `01_report_labeling.ipynb` from the top.
2. Preserve its completed outputs: `qwen_all_report_states.csv`, `qwen_state_calibration.csv`, and `qwen_verified_evaluation.csv`.
3. Attach those outputs and the competition data to the second notebook, then run it from the top with a GPU enabled for feature extraction.
4. Preserve `resnet18_mri_features_v1.npz`. Later sessions can attach that output and reuse completed features instead of repeating extraction.

The code expects the competition at `/kaggle/input/competitions/rsna-knee-abnormality-detection`; adjust `DATA_DIR` if your mount differs. The second notebook expects all three report CSVs. Compressed DICOM decoding may require additional pydicom decoder packages.

Use a completed **Save & Run All** version to preserve long-running outputs. Draft-session checkpoints are not a durable backup after a session reset.

**Stack:** Python, PyTorch, torchvision, Hugging Face Transformers, pydicom, pandas, NumPy, scikit-learn, Matplotlib, langdetect, and tqdm. Dependencies are not yet pinned to a reproducible environment.

## Running the tests

The repo's test command is `pytest`:

```bash
python -m pip install pytest
python -m pytest
```

The suite in `tests/` loads the `.ipynb` JSON, extracts the notebooks' own code cells, and verifies the fixes that protect re-execution — no GPU, dataset, or network needed. It covers:

- **Restart and prerequisite guards** — running the guard cells in a namespace missing their prerequisites raises a clear `RuntimeError` naming the cell to run first (B1, B2).
- **JSON response parsing** — clean JSON, JSON inside Markdown fences, JSON followed by brace-bearing prose (the old greedy-regex failure), and two-object responses (first object wins); unbalanced or missing braces raise the documented fallback error (B6).
- **Atomic checkpoints** — the save helpers write a complete, parseable CSV, leave no temporary file behind, and touch the destination only through `os.replace` (B4).
- **Structural checks** — the evaluation resume set retries rows with a recorded `ParsingError` (B3), the full-extraction loop saves in a `finally` block (B5), both figure cells end with `plt.close(fig)` (B7), and the parser copies in cells 9 and 12 stay behaviorally identical.
- **Whole-repo compilation** — every code cell of both notebooks byte-compiles after stripping IPython magics.

Requirements: `pytest` plus `pandas` and `numpy`, which the notebooks already depend on; heavy ML packages (`torch`, `transformers`, `pydicom`, `langdetect`) are stubbed automatically when not installed. The suite is hermetic and runs in under a second.

## Limitations and next steps

- Train the classifier on cached features and report per-label ROC-AUC across development folds.
- Compare official-only supervision with the weak-supervision baseline.
- Improve acquisition selection, unreadable-slice handling, and physical-spacing/orientation standardization.
- Investigate stronger MRI representations and alternatives to mean slice pooling.
- Build and test the image-only submission pipeline.

The official subset is small and may differ in prevalence from the broader dataset. Soft targets and weights are experimental estimates, not established confidence scores. Report grouping cannot rule out repeated patients with different reports. French has no officially labeled example in this subset, and language detection is approximate.

This is a research prototype, not a clinically validated diagnostic system.

## Data and publication

This repository contains notebook source with saved outputs cleared. Competition images, report datasets, generated per-study files, and model weights are not bundled. Obtain the data through the competition and follow its applicable terms. Running the notebooks can display report text and images again; clear those outputs before publishing updated copies.

## Author

Rahim Askarov
