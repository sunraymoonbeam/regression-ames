# Ames Housing — Advanced Regression

Predicting residential house sale prices on the [Ames Housing dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
(Kaggle's *House Prices: Advanced Regression Techniques*).

An end-to-end regression study: data inspection, exploratory analysis, preprocessing,
feature engineering, feature selection, and a comparison of seven regularised / boosted /
ensembled regressors, blended by non-negative least squares on out-of-fold predictions.

Scored on **RMSLE**, the competition metric. Best out-of-fold result: **0.1057** (voting
regressor), with the stacked model statistically tied at 0.1057.

---

## Results

Five-fold out-of-fold scores on the training split, plus a single held-out validation
split that was never used to fit, tune, select or weight anything:

| Model | OOF RMSLE | fold σ | Val RMSLE | Val RMSE ($) |
|---|---:|---:|---:|---:|
| Voting regressor | **0.10570** | 0.0096 | 0.12297 | 22,456 |
| Stacking (out-of-fold meta-features) | 0.10575 | 0.0107 | **0.12252** | 21,817 |
| Gradient Boosting | 0.10832 | 0.0087 | 0.12843 | 23,991 |
| Bayesian Ridge | 0.11014 | 0.0118 | 0.12485 | 21,536 |
| CatBoost | 0.11018 | 0.0071 | 0.12706 | 26,206 |
| Ridge | 0.11126 | 0.0119 | 0.12359 | 21,619 |
| LightGBM | 0.11183 | 0.0074 | 0.12814 | 25,643 |
| NNLS blend | 0.10542¹ | — | 0.12347 | 22,102 |

¹ The blend's OOF figure is measured on the same out-of-fold predictions its weights were
fitted to, so it is mildly optimistic and is shown for reference only. Its honest number
is the validation column.

**Read the fold σ column before reading the ranking.** The spread across folds (±0.007 to
±0.012) is larger than the gap between first and last place (0.0061). The ordering above is
not statistically meaningful — the top two are tied, and the honest conclusion is that a
regularised linear model, a boosted ensemble and a stack all land in the same place on this
dataset.

The NNLS blend puts its weight on stacking (0.44), gradient boosting (0.31), Bayesian Ridge
(0.17) and LightGBM (0.08), and zeroes out ridge, CatBoost and the voting regressor as
redundant. It does **not** beat the best single model on the untouched validation split.

---

## Validation protocol

Every number above comes from this protocol, which is worth stating explicitly because it
is what makes the numbers comparable:

- **Preprocessing lives inside the pipeline.** Imputation, scaling and encoding are steps
  in a `ColumnTransformer`, so they are refit on every CV fold rather than fit once on the
  full frame.
- **Target transform via `TransformedTargetRegressor`**, using `log1p`. This makes the
  in-CV objective (neg-MSE on the transformed target) numerically identical to MSLE, so
  tuning, selection and reporting all optimise the competition metric.
- **Outliers are removed from the training split only.** They are flagged before the split
  and dropped after it — you cannot delete inconvenient rows at inference time, so the
  validation set keeps its hard cases.
- **Blend weights are fitted on out-of-fold predictions** (NNLS in log space), never on the
  data the blend is then scored against.
- **Stacking uses out-of-fold meta-features** via sklearn's `StackingRegressor(cv=...)`.
- **An assertion guards the `ColumnTransformer`.** `remainder='drop'` means any column
  missing from the feature lists is silently discarded; the notebook now fails loudly
  instead of training on a quietly truncated frame.

---

## Repository layout

```
├── reports/
│   └── notebooks/
│       └── regression_ames_house_prices.ipynb   # ← the actual work (118 cells)
├── house_prices_advanced_regression_techniques/
│   ├── submission/                              # Kaggle submission CSVs
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
predictors; bivariate analysis against the target; multicollinearity screening on absolute
correlation.

**Preprocessing** — drops true near-duplicate columns and near-constant features. Note that
aggressive collinearity pruning is deliberately *not* done: every model here is either
L2-regularised or a tree ensemble, and neither is harmed by correlated inputs.

**Feature engineering** — `TotalSF`, `TotalBath`, `TotalBsmtFin`, `TotalPorch`, built from
their component columns and registered into the continuous feature group.

**Feature selection** — `SelectKBest` with `f_regression` and mutual information. These
scores are **diagnostic only**; no features are dropped on the basis of them.

**Modelling** — PyCaret for a fast first pass at model scoping, then Bayesian Ridge, Ridge,
CatBoost, Gradient Boosting and LightGBM, each tuned by `RandomizedSearchCV` over 5 folds;
then a stacking meta-learner, a voting regressor, and the NNLS blend.

---

## Running it

Verified on Python 3.11. PyCaret pins `scikit-learn` and `scipy` fairly tightly, which is
why `requirements.txt` carries bounds rather than bare names.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab reports/notebooks/regression_ames_house_prices.ipynb
```

The dataset is not committed (`data/` is gitignored). Download it from the
[competition page](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)
and place it at the repository root:

```
data/raw/train.csv
data/raw/test.csv
data/raw/data_description.txt
```

The notebook locates the project root itself, so it runs whether the kernel starts in the
repo root or in `reports/notebooks/`. Run it top to bottom; a full pass including all
hyperparameter searches takes roughly 25 minutes on a laptop.

---

## Attribution

`house_prices_advanced_regression_techniques/reference/` contains public notebooks by other
Kaggle authors, kept for reference while working through this problem. They are their
authors' work, not mine, and are included only as study material.
