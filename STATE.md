# STATE

## Done

- [x] Deps in `pyproject.toml` (pandas, scikit-learn, xgboost, shap, matplotlib, jupyter, openpyxl, xlrd, requests, numpy)
- [x] `src/m3_loan_default/data.py` — UCI download + clean + CSV cache
- [x] `src/m3_loan_default/model.py` — XGBoost train + 5-fold CV
- [x] `src/m3_loan_default/explain.py` — SHAP TreeExplainer + summary/waterfall plots
- [x] `notebooks/01_demo.ipynb` — full pipeline walkthrough
- [x] 10 pytest tests across `tests/test_{data,model,explain,smoke}.py`, all passing locally in ~4s
- [x] `scripts/run_pipeline.py` — refresh PNGs + metrics.json in docs/
- [x] `docs/confusion_matrix.png` + `docs/shap_summary.png` + `docs/shap_waterfall.png`
- [x] README rewritten with problem, dataset, approach, results table, screenshots, how-to-run

## Results

- **Holdout AUC: 0.773** (target > 0.75 — achieved)
- 5-fold CV AUC: 0.780
- Precision 0.472 / Recall 0.604 / F1 0.530 / Accuracy 0.763
- Top features: `PAY_0`, `LIMIT_BAL`, `BILL_AMT1`, `PAY_AMT2`, `PAY_AMT1`

## Next steps

- Tune threshold for desired precision/recall trade-off (currently 0.5).
- Try a logistic-regression baseline + LightGBM comparison.
- Add fairness slice metrics across `SEX`, `EDUCATION`, `AGE` buckets.
- Calibration plot — XGBoost probabilities tend to be over-confident.
- Optional: package a CLI entry point.

## Repo

- GitHub: https://github.com/ypatel39-commits/m3-loan-default
- Last commit hash: see `git log -1 --format=%H` (updated by commit step)
