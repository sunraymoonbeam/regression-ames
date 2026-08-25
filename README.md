# Ames Housing — Advanced Regression

Predicting residential house sale prices on the [Ames Housing dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
(Kaggle's *House Prices: Advanced Regression Techniques*).

The work is an end-to-end regression study: data inspection, exploratory analysis,
preprocessing, feature engineering, feature selection, and a comparison of seven
regularised / boosted / ensembled regressors, ending in a weighted blend used for
the Kaggle submission.

**Best single model:** Gradient Boosting Regressor — RMSE **20,749**, R² **0.911** on the hold-out split.
**Best overall:** weighted blend of all seven models — RMSE **19,873**, R² **0.918**.

---

## Results

Hold-out performance, target transformed before fitting and inverse-transformed for scoring:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Bayesian Ridge | 14,331 | 20,256 | 0.915 |
| Ridge | 14,332 | 20,305 | 0.915 |
| Voting Regressor | 14,454 | 20,502 | 0.913 |
| CatBoost | 14,770 | 20,649 | 0.912 |
| Gradient Boosting | 14,623 | 20,749 | 0.911 |
| LightGBM | 15,002 | 21,330 | 0.906 |
| Stacking (meta-learner) | 14,576 | 21,351 | 0.906 |
| **Weighted blend (all 7)** | **13,941** | **19,873** | **0.918** |

Blend weights are inverse to each model's RMSE, so stronger models contribute more.

Three target transformations were compared — log, Box-Cox, and Yeo-Johnson — with a
submission file produced for each.

---

## Repository layout

```
├── reports/
│   └── notebooks/
│       └── regression_ames_house_prices.ipynb   # ← the actual work (118 cells)
├── house_prices_advanced_regression_techniques/
│   ├── submission/                              # Kaggle submission CSVs per model / transformation
│   └── reference/                               # public Kaggle notebooks collected for study (not mine)
├── conf/                                        # Hydra config + model params (see caveat below)
├── src/                                         # modular pipeline scaffolding (see caveat below)
├── main.py                                      # CLI entrypoint (see caveat below)
├── main_sklearn_pipeline.py                     # sklearn-Pipeline variant of the same
├── requirements.txt
└── run.sh
```

### A note on `src/`, `conf/` and `main.py`

These are carried over from an earlier classification project and are **not wired to the
Ames dataset**. `conf/config.yaml` still points at a lung-cancer SQLite source and
`conf/model_config.py` still registers classifiers, so `python main.py` will not reproduce
anything in this README. They are kept here as the reusable pipeline skeleton they were
intended to be — porting them to regression is outstanding work.

**Everything reported above comes from the notebook**, which is self-contained.

---

## The notebook

`reports/notebooks/regression_ames_house_prices.ipynb`

**Data inspection** — duplicates, missing values, zero-variance features, mixed dtypes,
inconsistent categorical values, case-inconsistent formatting, dtype correction.

**EDA** — univariate analysis of `SalePrice` and of the continuous, ordinal and categorical
predictors; bivariate analysis against the target; multicollinearity screening.

**Preprocessing** — drops highly correlated and near-constant dominating features, then
imputes missing values and handles outliers.

**Feature engineering** — new derived features, target transformation (log / Box-Cox /
Yeo-Johnson compared), and feature selection via `SelectKBest` with `f_regression` and
mutual information.

**Modelling** — PyCaret for a fast first pass at model scoping, then Bayesian Ridge, Ridge,
CatBoost, Gradient Boosting, LightGBM, a stacking meta-learner, and a voting regressor;
finally the weighted blend.

---

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# the notebook also uses: seaborn, plotly, lightgbm, catboost, pycaret, scipy
jupyter lab reports/notebooks/regression_ames_house_prices.ipynb
```

The dataset is not committed (`data/` is gitignored). Download it from the
[competition page](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)
and place it so the notebook's relative paths resolve from the repo root:

```
data/raw/train.csv
data/raw/test.csv
data/raw/data_description.txt
```

Then run the notebook top to bottom.

---

## Attribution

`house_prices_advanced_regression_techniques/reference/` contains public notebooks by other
Kaggle authors, kept for reference while working through this problem. They are their
authors' work, not mine, and are included only as study material.
