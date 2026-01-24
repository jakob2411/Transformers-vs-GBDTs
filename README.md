Paying Too Much Attention? A Systematic Comparison of Transformer Models and GBDT Ensembles for Tabular Data - a Master’s Thesis
================================

Overview
--------
This repository contains scripts for benchmarking various tabular machine learning models (classification and regression). The main scripts are:

- `classification.py` — classification benchmark with cross-validation
- `regression.py` — regression benchmark with cross-validation
- `pipeline.py` — shared helpers, model factories, CV runners, preprocessing
- `requirements.txt` — Python dependencies
- `LICENSE` — MIT license

Repository Summary
------------------

### Models (6 total for classification, 5 for regression)
1. **TabPFN** — Pre-trained Transformer for tabular data (classification & regression)
2. **TabFlex** — Advanced TabPFN variant with adaptive model selection (classification only)
3. **LightGBM** — Gradient boosting framework (classification & regression)
4. **XGBoost** — Extreme gradient boosting (classification & regression)
5. **CatBoost** — Gradient boosting with categorical feature support (classification & regression)
6. **FT-Transformer** — Feature Tokenizer Transformer via pytorch-tabular (classification & regression)

### Tasks
- **Classification**: Binary and multi-class classification on tabular datasets
- **Regression**: Continuous target prediction on tabular datasets
- **Cross-Validation**: Stratified k-fold CV for classification, standard k-fold for regression
- **Metrics**: 
  - Classification: Accuracy, Balanced Accuracy, Precision, Recall, F1-Score, ROC-AUC, Log Loss
  - Regression: MSE, RMSE, MAE, R²

### Datasets

**Classification (default: Credit-G)**

| Dataset | OpenML ID | Size / Features | Notes |
| --- | --- | --- | --- |
| Credit-G | 31 | 1,000 / 21 | Binary credit risk |
| Electricity | 151 | 45,312 / 9 | Binary price up/down |
| Covertype | 150 | 581,012 / 55 | Multiclass forest cover |
| Nursery | 26 | 12,960 / 9 | Multiclass categorical |
| Adult | 1590 | 48,842 / 15 | Binary income >50K |
| HiggsBoson | 23512 | 98,050 / 29 | Binary HEP |
| BankMarketing | 1461 | 45,211 / 17 | Binary marketing response |
| Gesture_Wiimote | 4538 | 9,873 / 33 | Multiclass sensor data |
| PhishingWebsites | 4534 | 11,055 / 31 | Binary phishing |
| TelcoCustomerChurn | 42178 | 7,043 / 20 | Binary churn |

**Regression (default: BrazilianHouses)**

| Dataset | OpenML ID | Size / Features | Notes |
| --- | --- | --- | --- |
| Diamonds | 42225 | 53,940 / 10 | Price prediction |
| BikeSharingDemand | 42712 | 17,379 / 13 | Bike demand |
| BrazilianHouses | 42688 | 10,692 / 13 | House prices |
| AirlinesDelay | 42721 | 1,000,000 / 10 | Flight delays |
| MedicalCharges | 42720 | 163,065 / 12 | Insurance charges |
| CaliforniaHousing | 43939 | 20,640 / 10 | Housing values |
| YachtHydrodynamics | 42370 | 308 / 7 | Small hydrodynamics |
| EnergyEfficiency | 1472 | 768 / 10 | Building energy |
| Superconductivity | 44964 | 21,263 / 82 | High-dimensional |
| ProteinStructure_CASP | 44963 | 45,730 / 10 | Protein structure |

## Command-Line Options

Both scripts accept the same options:

```bash
# Common options for both classification.py and regression.py
--dataset NAME              # Dataset name from the built-in list
--openml-id ID              # OpenML dataset ID (overrides --dataset)
--task-id ID                # OpenML task ID (overrides dataset/openml-id)
--models MODELS             # Comma-separated list or 'all'
--cv-folds N                # Number of CV folds (default: 10)
--preserve-categorical      # Keep categorical features for CatBoost/LightGBM
--smoke                     # Quick test mode (2 folds, small models)
--n-estimators N            # Estimators for boosting models (default: 100)
--max-epochs N              # Max epochs for FT-Transformer (default: 20)
--n-jobs N                  # Parallel jobs for tree models (default: -1)
```

**Examples:**

```bash
# Quick smoke test
python3 classification.py --smoke

# Specific models on Adult dataset
python3 classification.py --dataset Adult --models "TabPFN,LightGBM,XGBoost" --cv-folds 5

# Regression with custom parameters
python3 regression.py --dataset Diamonds --n-estimators 200 --max-epochs 30

# Use OpenML ID directly
python3 regression.py --openml-id 42712 --models "TabPFN,CatBoost"
```

## Model Configuration

**Global Settings:**
- Seed: 2411 (for reproducibility)
- CV folds: 10 (configurable)

**Tree Models (LightGBM, XGBoost, CatBoost):**
- Estimators: 100 (default)
- Learning rate: 0.1
- Parallel jobs: -1 for LightGBM, 1 for XGBoost

**Transformer Models (TabPFN, TabFlex, FT-Transformer):**
- TabPFN: 4 ensemble configurations, CPU only
- TabFlex: Adaptive model selection, CPU only (classification only)
- FT-Transformer: 20 epochs with early stopping, auto accelerator (MPS/CPU)

## Evaluation Metrics

**Classification:**
- Accuracy, Balanced Accuracy
- Precision, Recall, F1-Score
- ROC-AUC, Log Loss
- Training and prediction time

**Regression:**
- RMSE, MAE, R²
- Training and prediction time

## Output Files

Results are saved to `outputs/` directory:
- Filename: `{task}_{dataset}_openml{id}_results.xlsx`
- Two sheets: "summary" (averages) and "folds" (per-fold metrics)

## Notes

- Large datasets may require significant RAM (especially Covertype, AirlinesDelay)
- First run downloads and caches datasets in `data_cache/`
- Use `--smoke` for quick validation
- Virtual environment recommended

Use of AI-Assisted Development Tools
--------
The software implementation for this project was supported by GitHub Copilot as an assistive development tool. The system was used in this project as an assistive development tool for code generation and real-time syntax completion. While the system contributed to drafting implementations and reducing manual coding time, every line of code was reviewed and tested. The author maintains full responsibility for the architectural design, logic verification, and final integrity of the software.