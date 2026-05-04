# CIS527-RL — FEVER Claim Verification with RLHF

CIS527 Final Project

Given a Wikipedia passage and a factual claim, fine-tune GPT-4.1-nano to predict one of three labels (SUPPORTED, CONTRADICTED, NOT MENTIONED) with a grounded one-sentence justification.

- **Base model:** GPT-4.1-nano (student)
- **Teacher model:** GPT-4.1-mini (for data generation)
- **Dataset:** FEVER benchmark

## Approach

Three fine-tuning strategies are compared:

1. **SFT** — supervised fine-tuning on teacher-generated (label, justification) pairs across three justification styles: Standard, Evidence-First (EF), and Contrastive (CT).
2. **DPO** — direct preference optimization using chosen/rejected pairs; negatives include wrong-label responses, hallucinated justifications, hedging, degenerate outputs, and reasoning–label mismatches. Positive (chosen) responses were also varied using EF and CT justification styles, mirroring the SFT ablations.
3. **Base model evaluation** — six prompt/inference configurations on untuned GPT-4.1-nano as baseline.

## Notebooks

| Notebook | Description |
|---|---|
| `FEVERPreprocessing.ipynb` | Download and preprocess FEVER splits |
| `DataGeneration.ipynb` | Generate SFT training data (Standard style) |
| `BaseModel.ipynb` | Base model evaluation across prompt configs |
| `SFT/DataGeneration_SFT_v2.ipynb` | Generate EF and CT SFT training data |
| `SFT/SFT.ipynb` | Train and evaluate SFT c1–c4 |
| `SFT/SFT_EF_CT.ipynb` | Train and evaluate SFT EF and CT configs |
| `DPO/DPO_c1-c4.ipynb` | DPO Round 1: train and evaluate c1–c4 |
| `DPO/DPO_EF.ipynb` | DPO with evidence-first chosen responses |
| `DPO/CT_DPO_from_SFT.ipynb` | DPO with contrastive chosen responses |
| `DataGeneration_DPO_v2.ipynb` | Generate Round 2 DPO data (richer negatives) |
| `DPO/DPO_v2_c5.ipynb` | DPO Round 2: train and evaluate v2\_c5 |

## Data

- `data/raw/` — raw FEVER files
- `data/joined/` — preprocessed passage–claim–label triples
- `data/generated/` — training files uploaded to Azure
- `data/results/` — dev/test eval outputs and loss curves
