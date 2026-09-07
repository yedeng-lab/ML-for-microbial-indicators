#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train Support Vector Regression models for fungal environmental variables.

Workflow
--------
1. Load OTU, environmental, and climate tables and align shared samples.
2. For each numeric target, remove samples missing the target or predictor values.
3. Split data into 70% training and 30% testing sets.
4. Standardize predictors inside a scikit-learn Pipeline.
5. Tune target-specific SVR hyperparameters with 5-fold cross-validation.
6. Evaluate training and test performance.
7. Bootstrap test-set R², RMSE, and MAE with 95% confidence intervals.
8. Calculate permutation importance on the test set.
9. Save model outputs and a project-level summary.
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
OTU_PATH = r"/mnt/data/home/tycloud/lijiaying/Fungi/otu/20260119_core_fungi_0.05freq_unmerged.txt"
ENV_PATH = r"/mnt/data/home/tycloud/lijiaying/Fungi/env/20260119_fungi_env_unmerged.txt"
CLIMATE_PATH = r"/mnt/data/home/tycloud/lijiaying/Fungi/climate/20260119_fungi_BIO1_12_unmerged.txt"
OUTDIR = r"/mnt/data/home/tycloud/lijiaying/Fungi/machine_learning/SVM_Fungi_BIO1_12_unmerged_1"

SEP_OTU = "\t"
SEP_ENV = "\t"
SEP_CLIM = "\t"
STRICT_DROP_NA_FOR_PREDICTORS = True
PREFIX_CLIMATE_COLS = True

TEST_SIZE = 0.30
CV_FOLDS = 5
RANDOM_SEED = 42
N_JOBS = 8
N_REPEATS_PERM = 10
N_BOOTSTRAPS = 1000
CI_LEVEL = 0.95

PARAM_GRIDS = {
    "pH": [
        {
            "model__kernel": ["rbf"],
            "model__C": [3, 10, 30],
            "model__epsilon": [0.005, 0.01, 0.05],
            "model__gamma": ["scale", 0.01, 0.05],
        }
    ],
    "SOC_content": [
        {
            "model__kernel": ["rbf"],
            "model__C": [3, 10, 30, 100],
            "model__epsilon": [0.05, 0.1, 0.2, 0.3],
            "model__gamma": ["scale", 0.01, 0.05],
        }
    ],
    "total_N_content": [
        {
            "model__kernel": ["rbf"],
            "model__C": [0.3, 1, 3, 10],
            "model__epsilon": [0.01, 0.05, 0.1],
            "model__gamma": ["scale", 0.005, 0.01],
        }
    ],
    "total_Ca": [
        {
            "model__kernel": ["linear"],
            "model__C": [0.05, 0.1, 0.3, 1, 3],
            "model__epsilon": [500, 1000, 2000],
        }
    ],
    "total_P": [
        {
            "model__kernel": ["rbf"],
            "model__C": [0.1, 1, 10],
            "model__epsilon": [50, 100, 200],
            "model__gamma": ["scale", 0.001, 0.005],
        },
        {
            "model__kernel": ["linear"],
            "model__C": [0.05, 0.1, 0.3, 1, 3],
            "model__epsilon": [50, 100, 200],
        },
    ],
    "total_K": [
        {
            "model__kernel": ["linear"],
            "model__C": [0.05, 0.1, 0.3, 1, 3],
            "model__epsilon": [200, 500, 1000],
        }
    ],
}


# =============================================================================
# 2. Helper functions
# =============================================================================
def rmse(y_true, y_pred):
    """Return root mean squared error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def bootstrap_regression_metrics(y_true, y_pred, n_boot, alpha, seed):
    """Bootstrap R², RMSE, and MAE and return their means and confidence intervals."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    bootstrap_values = np.empty((n_boot, 3), dtype=float)

    for i in range(n_boot):
        idx = rng.randint(0, n, n)
        y_true_i = y_true[idx]
        y_pred_i = y_pred[idx]
        bootstrap_values[i, 0] = r2_score(y_true_i, y_pred_i)
        bootstrap_values[i, 1] = rmse(y_true_i, y_pred_i)
        bootstrap_values[i, 2] = mean_absolute_error(y_true_i, y_pred_i)

    means = bootstrap_values.mean(axis=0)
    ci_low = np.percentile(bootstrap_values, 100 * alpha / 2, axis=0)
    ci_high = np.percentile(bootstrap_values, 100 * (1 - alpha / 2), axis=0)

    return {
        "R2_mean": float(means[0]),
        "R2_CI_low": float(ci_low[0]),
        "R2_CI_high": float(ci_high[0]),
        "RMSE_mean": float(means[1]),
        "RMSE_CI_low": float(ci_low[1]),
        "RMSE_CI_high": float(ci_high[1]),
        "MAE_mean": float(means[2]),
        "MAE_CI_low": float(ci_low[2]),
        "MAE_CI_high": float(ci_high[2]),
    }


def make_stratification_bins(y):
    """Create quantile bins for regression stratification, or return None if invalid."""
    try:
        bins = pd.qcut(y, q=10, duplicates="drop")
        if bins.nunique() < 2 or bins.value_counts().min() < 2:
            return None
        return bins
    except Exception:
        return None


# =============================================================================
# 3. Load data and align samples
# =============================================================================
otu = pd.read_csv(OTU_PATH, sep=SEP_OTU, header=0, index_col=0)
env = pd.read_csv(ENV_PATH, sep=SEP_ENV, header=0, index_col=0)
clim = pd.read_csv(CLIMATE_PATH, sep=SEP_CLIM, header=0, index_col=0)

common_samples = sorted(set(otu.columns) & set(env.index) & set(clim.index))
if len(common_samples) < 5:
    raise SystemExit(f"Too few aligned samples: {len(common_samples)}")

X_otu_all = otu[common_samples].T.astype(float)
X_clim_all = clim.loc[common_samples].apply(pd.to_numeric, errors="coerce")
if PREFIX_CLIMATE_COLS:
    X_clim_all.columns = [f"CLIM_{column}" for column in X_clim_all.columns]

X_all = pd.concat([X_otu_all, X_clim_all], axis=1)
env_all = env.loc[common_samples].copy()

os.makedirs(OUTDIR, exist_ok=True)
summary = {}


# =============================================================================
# 4. Train one SVR model per environmental target
# =============================================================================
for target in env_all.columns:
    print(f"\n===== Target: {target} =====", flush=True)

    # Step 4.1: Keep numeric targets and samples with complete data.
    y_full = env_all[target]
    if not pd.api.types.is_numeric_dtype(y_full):
        print("Skipped: target is not numeric.")
        continue

    mask_y = y_full.notna()
    X_sub = X_all.loc[mask_y].copy()
    y_sub = y_full.loc[mask_y].astype(float).copy()

    if STRICT_DROP_NA_FOR_PREDICTORS:
        mask_complete = X_sub.notna().all(axis=1)
        X_sub = X_sub.loc[mask_complete]
        y_sub = y_sub.loc[mask_complete]

    if len(X_sub) < 5:
        print(f"Skipped: only {len(X_sub)} valid samples.")
        continue

    if target not in PARAM_GRIDS:
        print(f"Skipped: no hyperparameter grid is defined for '{target}'.")
        continue

    # Step 4.2: Split data into training and test sets.
    stratify_bins = make_stratification_bins(y_sub)
    X_train, X_test, y_train, y_test = train_test_split(
        X_sub,
        y_sub,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=stratify_bins,
    )

    target_dir = os.path.join(OUTDIR, str(target))
    os.makedirs(target_dir, exist_ok=True)
    safe_target = re.sub(r"[^A-Za-z0-9._-]+", "_", str(target))

    with open(os.path.join(target_dir, f"{safe_target}_split.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "train_samples": X_train.index.tolist(),
                "test_samples": X_test.index.tolist(),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    # Step 4.3: Standardize predictors and tune SVR hyperparameters with 5-fold CV.
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", SVR()),
        ]
    )
    cv = KFold(
        n_splits=min(CV_FOLDS, len(X_train)),
        shuffle=True,
        random_state=RANDOM_SEED,
    )
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=PARAM_GRIDS[target],
        scoring="r2",
        cv=cv,
        n_jobs=N_JOBS,
        refit=True,
        verbose=0,
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_

    # Step 4.4: Evaluate training and test performance.
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

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

    # Step 4.5: Bootstrap test-set metrics and calculate 95% confidence intervals.
    bootstrap_metrics = bootstrap_regression_metrics(
        y_true=y_test.to_numpy(),
        y_pred=y_test_pred,
        n_boot=N_BOOTSTRAPS,
        alpha=1 - CI_LEVEL,
        seed=RANDOM_SEED,
    )

    # Step 4.6: Calculate permutation importance on the test set.
    perm = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=N_REPEATS_PERM,
        random_state=RANDOM_SEED,
        n_jobs=N_JOBS,
        scoring="r2",
    )
    importance_df = pd.DataFrame(
        {
            "feature": X_sub.columns,
            "perm_importance_mean": perm.importances_mean,
            "perm_importance_std": perm.importances_std,
        }
    ).sort_values("perm_importance_mean", ascending=False)

    # Step 4.7: Save CV results, metrics, predictions, importance, and fitted pipeline.
    pd.DataFrame(search.cv_results_).to_csv(
        os.path.join(target_dir, f"{safe_target}_cv_results.txt"),
        sep="\t",
        index=False,
    )

    metrics_out = {
        "task": "regression",
        "n_samples": int(len(X_sub)),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "train_metrics": train_metrics,
        "test_metrics": {
            **test_metrics,
            "bootstrap": {
                "n_bootstraps": N_BOOTSTRAPS,
                "ci_level": CI_LEVEL,
                **bootstrap_metrics,
            },
        },
        "cv": {
            "cv_folds": int(cv.n_splits),
            "scorer": "r2",
            "best_score": float(search.best_score_),
            "best_params": search.best_params_,
        },
        "delta_R2_test_minus_train": float(test_metrics["R2"] - train_metrics["R2"]),
    }

    with open(os.path.join(target_dir, f"{safe_target}_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics_out, handle, ensure_ascii=False, indent=2)

    pd.DataFrame(
        {"y_true": y_test.values, "y_pred": y_test_pred},
        index=X_test.index,
    ).rename_axis("sample").to_csv(
        os.path.join(target_dir, f"{safe_target}_predictions.txt"), sep="\t"
    )

    importance_df.to_csv(
        os.path.join(target_dir, f"{safe_target}_importances.txt"),
        sep="\t",
        index=False,
    )
    dump(model, os.path.join(target_dir, f"{safe_target}_SVMModel.joblib"))

    summary[target] = metrics_out
    print(
        f"Completed: {target} | Test R2 = {test_metrics['R2']:.3f} | "
        f"CV R2 = {search.best_score_:.3f}"
    )


# =============================================================================
# 5. Save project-level summary
# =============================================================================
with open(os.path.join(OUTDIR, "svm_fungi_summary.json"), "w", encoding="utf-8") as handle:
    json.dump(summary, handle, ensure_ascii=False, indent=2)

print(f"\nAll targets completed. Output directory: {OUTDIR}")
