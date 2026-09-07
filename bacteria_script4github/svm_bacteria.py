#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train Support Vector Regression (SVR) models for environmental variables using
bacterial OTU features and climate variables.

Input format
------------
- OTU table: rows = OTUs, columns = samples
- Environment table: rows = samples, columns = regression targets
- Climate table: rows = samples, columns = climate predictors

Workflow
--------
1. Load and align samples shared by all input tables.
2. For each environmental target, remove samples with missing target values.
3. Split samples into 70% training and 30% testing sets.
4. Standardize predictors inside a scikit-learn Pipeline.
5. Tune RBF-SVR hyperparameters with 5-fold cross-validation.
6. Evaluate the best model on training and test sets.
7. Estimate bootstrap confidence intervals for R2, RMSE, and MAE.
8. Calculate test-set permutation importance.
9. Save metrics, predictions, CV results, feature importance, split information,
   and the trained Pipeline.

Dependencies
------------
numpy, pandas, scikit-learn, joblib
"""

import json
import os
import re
import warnings

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")


# =============================================================================
# 1. Configuration
# =============================================================================
OTU_PATH = r"/mnt/data/home/tycloud/lijiaying/Bacteria/otu/20251216_core_bacteria_0.25freq_unmerged.txt"
ENV_PATH = r"/mnt/data/home/tycloud/lijiaying/Bacteria/env/bacteria_TP_env_unmerged.txt"
CLIMATE_PATH = r"/mnt/data/home/tycloud/lijiaying/Bacteria/climate/20251214_bacteria_BIO1_12_unmerged.txt"
OUTDIR = r"/mnt/data/home/tycloud/lijiaying/Bacteria/machine_learning/SVM_Bacteria_BIO1_12_unmerged"

SEP_OTU = "\t"
SEP_ENV = "\t"
SEP_CLIM = "\t"

TEST_SIZE = 0.30
CV_FOLDS = 5
RANDOM_SEED = 42
N_JOBS = 2
N_REPEATS_PERM = 10
N_BOOTSTRAPS = 1000
CI_LEVEL = 0.95
STRICT_DROP_NA_FOR_PREDICTORS = True

SVR_PARAM_GRID = {
    "model__kernel": ["rbf"],
    "model__C": [1, 10, 100, 1000],
    "model__epsilon": [0.01, 0.1, 0.2],
    "model__gamma": ["scale", 0.01, 0.1, 1],
}


# =============================================================================
# 2. Helper functions
# =============================================================================
def make_stratification_bins(y, n_bins=10):
    """Create quantile bins for regression stratification when possible."""
    try:
        bins = pd.qcut(y, q=n_bins, duplicates="drop")
        if bins.nunique() < 2 or bins.value_counts().min() < 2:
            return None
        return bins
    except Exception:
        return None


def rmse(y_true, y_pred):
    """Return root mean squared error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def bootstrap_regression_metrics(
    y_true,
    y_pred,
    n_bootstraps=1000,
    ci_level=0.95,
    seed=42,
):
    """Bootstrap R2, RMSE, and MAE with percentile confidence intervals."""
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    alpha = 1.0 - ci_level

    rows = []
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        yt = y_true[idx]
        yp = y_pred[idx]

        r2 = r2_score(yt, yp)
        if not np.isfinite(r2):
            continue

        rows.append(
            [
                r2,
                rmse(yt, yp),
                float(mean_absolute_error(yt, yp)),
            ]
        )

    if not rows:
        return {
            "R2_mean": np.nan,
            "R2_CI_low": np.nan,
            "R2_CI_high": np.nan,
            "RMSE_mean": np.nan,
            "RMSE_CI_low": np.nan,
            "RMSE_CI_high": np.nan,
            "MAE_mean": np.nan,
            "MAE_CI_low": np.nan,
            "MAE_CI_high": np.nan,
        }

    values = np.asarray(rows, dtype=float)
    means = values.mean(axis=0)
    low = np.percentile(values, 100 * alpha / 2, axis=0)
    high = np.percentile(values, 100 * (1 - alpha / 2), axis=0)

    return {
        "R2_mean": float(means[0]),
        "R2_CI_low": float(low[0]),
        "R2_CI_high": float(high[0]),
        "RMSE_mean": float(means[1]),
        "RMSE_CI_low": float(low[1]),
        "RMSE_CI_high": float(high[1]),
        "MAE_mean": float(means[2]),
        "MAE_CI_low": float(low[2]),
        "MAE_CI_high": float(high[2]),
    }


# =============================================================================
# 3. Load data and align samples
# =============================================================================
otu = pd.read_csv(OTU_PATH, sep=SEP_OTU, header=0, index_col=0)
env = pd.read_csv(ENV_PATH, sep=SEP_ENV, header=0, index_col=0)
climate = pd.read_csv(CLIMATE_PATH, sep=SEP_CLIM, header=0, index_col=0)

common_samples = sorted(set(otu.columns) & set(env.index) & set(climate.index))
if len(common_samples) < 5:
    raise SystemExit(
        f"Too few shared samples after alignment: {len(common_samples)}. "
        "Check sample IDs in the input tables."
    )

x_otu = otu[common_samples].T.astype(float)
x_climate = climate.loc[common_samples].apply(pd.to_numeric, errors="coerce")
x_all = pd.concat([x_otu, x_climate], axis=1)
env_all = env.loc[common_samples].copy()

os.makedirs(OUTDIR, exist_ok=True)
summary = {}


# =============================================================================
# 4. Train one SVR model per environmental target
# =============================================================================
for target in env_all.columns:
    print(f"\n===== Target: {target} =====", flush=True)

    # Step 4.1: Keep samples with an observed numeric target value.
    y_full = env_all[target]
    if not pd.api.types.is_numeric_dtype(y_full):
        print(f"Skipped {target}: target is not numeric.", flush=True)
        continue

    valid_target = y_full.notna()
    x_sub = x_all.loc[valid_target].copy()
    y_sub = y_full.loc[valid_target].astype(float).copy()

    if STRICT_DROP_NA_FOR_PREDICTORS:
        valid_predictors = x_sub.notna().all(axis=1)
        x_sub = x_sub.loc[valid_predictors]
        y_sub = y_sub.loc[valid_predictors]

    if len(x_sub) < 10:
        print(f"Skipped {target}: only {len(x_sub)} valid samples.", flush=True)
        continue

    # Step 4.2: Split data into training and test sets.
    stratify_bins = make_stratification_bins(y_sub)
    x_train, x_test, y_train, y_test = train_test_split(
        x_sub,
        y_sub,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=stratify_bins,
    )

    print(
        f"Training samples: {len(x_train)} | Test samples: {len(x_test)} | "
        f"Features: {x_train.shape[1]}",
        flush=True,
    )

    target_dir = os.path.join(OUTDIR, str(target))
    os.makedirs(target_dir, exist_ok=True)
    safe_target = re.sub(r"[^A-Za-z0-9._-]+", "_", str(target))

    split_info = {
        "train_samples": x_train.index.tolist(),
        "test_samples": x_test.index.tolist(),
    }
    with open(
        os.path.join(target_dir, f"{safe_target}_split.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(split_info, handle, ensure_ascii=False, indent=2)

    # Step 4.3: Build a scaling + SVR Pipeline.
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", SVR()),
        ]
    )

    # Step 4.4: Tune SVR hyperparameters with 5-fold cross-validation.
    cv = KFold(
        n_splits=max(2, min(CV_FOLDS, len(x_train))),
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=SVR_PARAM_GRID,
        scoring="r2",
        cv=cv,
        n_jobs=N_JOBS,
        refit=True,
        verbose=0,
    )
    search.fit(x_train, y_train)
    model = search.best_estimator_

    # Step 4.5: Evaluate training and test performance.
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    train_metrics = {
        "R2": float(r2_score(y_train, y_train_pred)),
        "RMSE": rmse(y_train, y_train_pred),
        "MAE": float(mean_absolute_error(y_train, y_train_pred)),
    }

    test_metrics = {
        "R2": float(r2_score(y_test, y_test_pred)),
        "RMSE": rmse(y_test, y_test_pred),
        "MAE": float(mean_absolute_error(y_test, y_test_pred)),
    }

    # Step 4.6: Bootstrap test-set metrics.
    bootstrap_metrics = bootstrap_regression_metrics(
        y_true=y_test,
        y_pred=y_test_pred,
        n_bootstraps=N_BOOTSTRAPS,
        ci_level=CI_LEVEL,
        seed=RANDOM_SEED,
    )

    # Step 4.7: Calculate permutation importance on the test set.
    permutation = permutation_importance(
        model,
        x_test,
        y_test,
        n_repeats=N_REPEATS_PERM,
        random_state=RANDOM_SEED,
        n_jobs=N_JOBS,
        scoring="r2",
    )

    importance_df = pd.DataFrame(
        {
            "feature": x_test.columns,
            "perm_importance_mean": permutation.importances_mean,
            "perm_importance_std": permutation.importances_std,
        }
    ).sort_values("perm_importance_mean", ascending=False)

    # Step 4.8: Save outputs.
    metrics = {
        "task": "regression",
        "n_samples": int(len(x_sub)),
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
        "cv": {
            "cv_folds": int(cv.get_n_splits()),
            "scorer": "r2",
            "best_score": float(search.best_score_),
            "best_params": search.best_params_,
        },
        "train_metrics": train_metrics,
        "test_metrics": {
            **test_metrics,
            "bootstrap": {
                "n_bootstraps": N_BOOTSTRAPS,
                "ci_level": CI_LEVEL,
                **bootstrap_metrics,
            },
        },
    }

    pd.DataFrame(search.cv_results_).to_csv(
        os.path.join(target_dir, f"{safe_target}_cv_results.txt"),
        sep="\t",
        index=False,
    )

    prediction_df = pd.DataFrame(
        {"y_true": y_test.values, "y_pred": y_test_pred},
        index=x_test.index,
    )
    prediction_df.index.name = "sample"
    prediction_df.to_csv(
        os.path.join(target_dir, f"{safe_target}_predictions.txt"),
        sep="\t",
    )

    importance_df.to_csv(
        os.path.join(target_dir, f"{safe_target}_perm_importance_test.txt"),
        sep="\t",
        index=False,
    )

    with open(
        os.path.join(target_dir, f"{safe_target}_metrics.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    dump(model, os.path.join(target_dir, f"{safe_target}_SVMModel.joblib"))

    summary[target] = metrics
    print(
        f"Completed {target} | Test R2 = {test_metrics['R2']:.3f} | "
        f"CV best R2 = {search.best_score_:.3f}",
        flush=True,
    )


# =============================================================================
# 5. Save the combined summary
# =============================================================================
with open(
    os.path.join(OUTDIR, "svm_bacteria_summary.json"),
    "w",
    encoding="utf-8",
) as handle:
    json.dump(summary, handle, ensure_ascii=False, indent=2)

print(f"\nAll targets completed. Results saved to: {OUTDIR}", flush=True)
