# Ames Housing — Advanced Regression

Predicting residential house sale prices on the [Ames Housing dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
(Kaggle's *House Prices: Advanced Regression Techniques*).

An end-to-end regression study: data inspection, exploratory analysis, preprocessing,
feature engineering, feature selection, and a comparison of seven regularised / boosted /
ensembled regressors, blended by non-negative least squares on out-of-fold predictions.

Scored on **RMSLE**, the competition metric. Best out-of-fold result: **0.1059** (stacking),
with the voting regressor statistically tied at 0.1062. An earlier revision of this
pipeline scored **0.124 on the Kaggle leaderboard**, against a predicted 0.12–0.13 and a
local validation figure of 0.1235 — so the validation protocol below is calibrated.

---

## Results

Five-fold out-of-fold scores on the training split, plus a single held-out validation
split that was never used to fit, tune, select or weight anything:

| Model | OOF RMSLE | fold σ | Val RMSLE | Val RMSE ($) |
|---|---:|---:|---:|---:|
| Stacking (out-of-fold meta-features) | **0.10595** | 0.0100 | 0.12126 | 22,569 |
| Voting regressor | 0.10615 | 0.0096 | 0.12164 | 23,371 |
| Gradient Boosting | 0.10869 | 0.0077 | 0.12721 | 23,243 |
| ElasticNet | 0.10977 | 0.0105 | 0.12488 | 23,857 |
| Lasso | 0.10993 | 0.0109 | **0.12108** | 23,563 |
| Bayesian Ridge | 0.11015 | 0.0109 | 0.12387 | 23,612 |
| CatBoost | 0.11076 | 0.0078 | 0.12638 | 25,373 |
| Ridge | 0.11137 | 0.0114 | 0.12229 | 23,523 |
| LightGBM | 0.11257 | 0.0076 | 0.12845 | 25,057 |
| NNLS blend | 0.10540¹ | — | 0.12162 | 22,732 |

¹ The blend's OOF figure is measured on the same out-of-fold predictions its weights were
fitted to, so it is mildly optimistic and is shown for reference only. Its honest number
is the validation column.

**Read the fold σ column before reading the ranking.** The spread across folds (±0.008 to
±0.011) is larger than the gap between first and last place (0.0066). The ordering above is
not statistically meaningful — the top two are tied, and the honest conclusion is that a
regularised linear model, a boosted ensemble and a stack all land in the same place on this
dataset. Note that Lasso ranks 5th out-of-fold but 1st on validation: exactly the kind of
reshuffling the fold σ predicts, and a reason not to read either column too closely.

The NNLS blend puts its weight on gradient boosting (0.42), Lasso (0.29) and stacking
(0.28), zeroing out everything else as redundant. Lasso earning 29% of the blend is the
clearest evidence that adding L1 was worthwhile — it contributes diversity the L2 models
and the trees do not, even though its own OOF rank is mid-table.

**On the size of these gains:** relative to the revision that scored 0.124, the changes
below moved out-of-fold from 0.1057 to 0.1059 (i.e. not at all) and validation from 0.1225
to 0.1211. Both differences sit well inside the fold σ. The honest statement is that the
feature work is defensible on its merits and produced a better-behaved model — predictions
no longer extrapolate 20% past the training maximum — but it did not produce a measurable
accuracy gain on a dataset this size.

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
- **Learned imputations live in the pipeline too**, including the per-`Neighborhood`
  median used for `LotFrontage`, which is a fitted statistic and so must be refit per fold.
- **An assertion guards the `ColumnTransformer`.** `remainder='drop'` means any column
  missing from the feature lists is silently discarded; the notebook now fails loudly
  instead of training on a quietly truncated frame.

---

## Repository layout

```
├── reports/
│   ├── notebooks/
│   │   └── regression_ames_house_prices.ipynb   # the analysis, end to end
│   └── figures/                                 # written by main.py
├── house_prices_advanced_regression_techniques/
│   ├── submission/                              # Kaggle submission CSVs
│   └── reference/                               # public Kaggle notebooks collected for study (not mine)
├── conf/                                        # Hydra config, column knowledge, model grids
├── src/                                         # the CLI pipeline (see below)
├── main.py                                      # CLI entrypoint
├── requirements.txt
└── run.sh
```

### The CLI pipeline

`src/`, `conf/` and `main.py` implement the same approach as a configurable
command-line pipeline, driven by Hydra:

```bash
python main.py                                              # full run, all 7 models
python main.py training.models=[ridge,lasso] training.tune=false   # fast
python main.py training.n_iter=20 training.ensemble=false
./run.sh preprocessing.numeric_scaling=standard
```

It shares the notebook's evaluation protocol — preprocessing refit per fold, outliers
removed from the training split only, models ranked out-of-fold with fold σ reported,
blend weights from out-of-fold predictions — and writes Kaggle submissions to
`house_prices_advanced_regression_techniques/submission/`.

| Module | Responsibility |
|---|---|
| `src/dataloader/dataloader.py` | Read the competition CSVs onto a shared index |
| `src/features/cleaning.py` | Deterministic recoding, ordinal maps, structural column drops |
| `src/features/transform.py` | Aggregate + age features, and the feature registry |
| `src/pipe/sklearn_pipeline_manager.py` | `ColumnTransformer`, neighbourhood imputer, pipeline factory |
| `src/train/model_factory.py` | Regressor construction from config |
| `src/train/model_trainer.py` | Tuning, out-of-fold scoring, ensembles, NNLS blend |
| `src/train/inference.py` | Submission writing, model persistence |
| `conf/config.yaml` | Everything tunable |
| `conf/feature_config.py` | Column knowledge (ordinal orders, aggregate definitions) |
| `conf/model_config.py` | Model registry, defaults, search grids |

Two invariants are enforced by assertion rather than convention, because both failed
silently in earlier revisions: the feature registry must exactly match the columns in
the frame (`FeatureTransformer.verify_registry`), and every column must be claimed by a
transformer (`TransformerPipelineManager.verify_coverage`). `ColumnTransformer` defaults
to `remainder='drop'`, so a column missing from the registry is discarded with no error
and no warning — the model simply trains on less data than you think.

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
regularised or a tree ensemble, and neither is harmed by correlated inputs. Columns are
also not discarded for missingness alone: where `NA` means "absent" (`GarageType`,
`MasVnrType`) it becomes an explicit `None` category, and `LotFrontage` is imputed with its
neighbourhood median rather than dropped.

**Feature engineering** — `TotalSF`, `TotalBath`, `TotalBsmtFin`, `TotalPorch`, plus
`HouseAge` and `RemodAge` (age at sale is a more direct parametrisation than the raw
calendar years), all registered into the continuous feature group. Continuous predictors
are skew-corrected with Yeo-Johnson inside the pipeline, not just the target.

**Feature selection** — `SelectKBest` with `f_regression` and mutual information. These
scores are **diagnostic only**; no features are dropped on the basis of them.

**Modelling** — PyCaret for a fast first pass at model scoping, then Bayesian Ridge, Ridge,
Lasso, ElasticNet, CatBoost, Gradient Boosting and LightGBM, each tuned by
`RandomizedSearchCV` over 5 folds; then a stacking meta-learner, a voting regressor, and
the NNLS blend.

---

## Running it

Verified on Python 3.11. PyCaret pins `scikit-learn` and `scipy` fairly tightly, which is
why `requirements.txt` carries bounds rather than bare names.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# either the notebook:
jupyter lab reports/notebooks/regression_ames_house_prices.ipynb
# or the CLI:
python main.py
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
hyperparameter searches takes roughly 40 minutes on a laptop.

---

## Attribution

`house_prices_advanced_regression_techniques/reference/` contains public notebooks by other
Kaggle authors, kept for reference while working through this problem. They are their
authors' work, not mine, and are included only as study material.
