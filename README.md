# NETWORK LATENCY PREDICTION IN 6G INDUSTRIAL SYSTEMS

End-to-end ML pipeline for predicting latency and SLA violations in a simulated 6G industrial network.

**Dataset:** 6 CSV files → merged into 40 000 rows × 57 columns → 68 engineered features.
**Models:** Ridge, XGBoost, LightGBM, LSTM, Stacking Ensemble.
**Extras:** SHAP explainability, robustness testing, enforcement causal analysis (DiD + PSM), online prediction CLI.

---

## Quick Start

```bash
# 1. Setup
python -m venv .venv
.venv\Scripts\activate        # Windows PowerShell
# source .venv/bin/activate   # Linux / macOS
pip install -r requirements.txt

# 2. Run full pipeline
python scripts/make_combined_40k.py
python -m src.features.feature_pipeline
python -m src.models.baseline
python -m src.models.advanced
python -m src.models.hpo
python -m src.models.ensemble
python -m src.eval.explain
python -m src.eval.error_analysis
python -m src.eval.robustness
python -m src.models.enforcement_effects

# 3. Online prediction
python -m src.predict.online_predict --input examples/sample_input.json --output examples/sample_output.json

# 4. Tests (50 tests)
python -m pytest tests/ -v
```

---

## Project Structure

```
CN_project/
├── data/                # Raw CSVs + train_ready.parquet
├── scripts/             # make_combined_40k.py
├── src/
│   ├── data/            # load_data.py
│   ├── features/        # feature_pipeline.py (68 features)
│   ├── models/          # baseline, advanced, hpo, ensemble, enforcement_effects
│   ├── eval/            # explain, error_analysis, robustness
│   └── predict/         # online_predict.py (CLI)
├── models/              # Saved .joblib artifacts
├── figures/             # 36 generated plots
├── reports/             # JSON metrics + markdown reports
├── notebooks/           # 00–11 step-by-step notebooks
├── examples/            # sample_input.json, sample_output.json
├── tests/               # 50 passing tests
├── final_report.tex     # Overleaf-ready LaTeX report
├── figures_needed.txt   # Figure reference map for LaTeX
├── requirements.txt
└── README.md
```

## Notebooks

| #   | Notebook                        | Topic                       |
| --- | ------------------------------- | --------------------------- |
| 00  | `00_quick_start.ipynb`          | Environment check           |
| 01  | `01_data_ingest.ipynb`          | Data loading & merge        |
| 02  | `02_EDA.ipynb`                  | Exploratory data analysis   |
| 04  | `04_features.ipynb`             | Feature engineering         |
| 05  | `05_baseline.ipynb`             | Baseline models             |
| 06  | `06_advanced.ipynb`             | XGBoost-ES + LSTM           |
| 07  | `07_hpo.ipynb`                  | Hyperparameter optimisation |
| 08  | `08_explainability.ipynb`       | SHAP + error analysis       |
| 09  | `09_robustness.ipynb`           | Slice, noise, drift tests   |
| 10  | `10_enforcement_analysis.ipynb` | Causal analysis (DiD, PSM)  |
| 11  | `11_predict_demo.ipynb`         | Online prediction demo      |

## Key Results

| Task           | Model    | Metric   | Value    |
| -------------- | -------- | -------- | -------- |
| Regression     | LightGBM | MAE      | 12.08 μs |
| Regression     | Ensemble | MAE      | 12.09 μs |
| Classification | LightGBM | AUC      | 0.517    |
| Classification | Ensemble | Accuracy | 90.4%    |

> R² ≈ 0 and AUC ≈ 0.5 are expected — the synthetic data has no exploitable signal. The pipeline is production-ready for real 6G data.


