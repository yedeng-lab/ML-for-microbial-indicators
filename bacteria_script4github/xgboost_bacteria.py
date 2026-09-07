#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train XGBoost regression models for environmental variables using bacterial
OTU features and climate variables.

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
4. Tune XGBoost hyperparameters with 5-fold cross-validation.
5. Fit the final model with early stopping using an internal validation subset.
6. Evaluate the final model on training and test sets.
7. Estimate a 95% bootstrap confidence interval for test-set R2.
8. Calculate gain importance and test-set permutation importance.
9. Save metrics, predictions, CV results, feature importance, split information,
   and the trained model.

Dependencies
------------
numpy, pandas, scikit-learn, joblib, xgboost
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
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


# =============================================================================
# 1. Configuration
# =============================================================================
OTU_PATH = r"/mnt/data/home/tycloud/lijiaying/Bacteria/otu/20251216_core_bacteria_0.3freq.txt"
ENV_PATH = r"/mnt/data/home/tycloud/lijiaying/Bacteria/env/20251214_bacteria_env.txt"
CLIMATE_PATH = r"/mnt/data/home/tycloud/lijiaying/Bacteria/climate/20251214_bacteria_BIO1_12.txt"
OUTDIR = r"/mnt/data/home/tycloud/lijiaying/Bacteria/machine_learning/XGBoost_bacteria_BIO1_12"

SEP_OTU = "\t"
SEP_ENV = "\t"
SEP_CLIM = "\t"

TEST_SIZE = 0.30
VALIDATION_SIZE = 0.10
CV_FOLDS = 5
RANDOM_SEED = 42
N_JOBS = 16
N_ESTIMATORS_MAX = 2000
EARLY_STOPPING_ROUNDS = 100
N_REPEATS_PERM = 10
N_BOOTSTRAPS = 1000
ALPHA_CI = 0.05
STRICT_DROP_NA_FOR_PREDICTORS = True
TREE_METHOD = "hist"

PARAM_GRID = {
    "learning_rate": [0.03, 0.05, 0.1],
    "max_depth": [4, 6, 8, 10],
    "min_child_weight": [1, 3, 5],
    "subsample": [0.7, 0.9],
    "colsample_bytree": [0.7, 0.9],
    "reg_lambda": [0.5, 1.0, 2.0],
    "reg_alpha": [0.0, 0.1],
    "gamma": [0.0, 0.1, 0.5],
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


def bootstrap_r2(y_true, y_pred, n_bootstraps, alpha, seed):
    """Estimate bootstrap mean and confidence interval for R2."""
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    values = []
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        score = r2_score(y_true[idx], y_pred[idx])
        if np.isfinite(score):
            values.append(score)

    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.nan, np.nan, np.nan

    low, high = np.percentile(
        values,
        [100 * alpha / 2, 100 * (1 - alpha / 2)],
    )
    return float(values.mean()), float(low), float(high)


# =============================================================================
# 3. Load data and align samples
# =============================================================================
otu = pd.read_csv(OTU_PATH, sep=SEP_OTU, header=0, index_col=0)
env = pd.read_csv(ENV_PATH, sep=SEP_ENV, header=0, index_col=0)
climate = pd.read_csv(CLIMATE_PATH, sep=SEP_CLIM, header=0, index_col=0)

env = env.round(2)

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
# 4. Train one XGBoost model per environmental target
# =============================================================================
for target in env_all.columns:
    print(f"\n===== Target: {target} =====", flush=True)

    # Step 4.1: Keep samples with an observed target value.
    y_full = env_all[target]
    valid_target = y_full.notna()
    x_sub = x_all.loc[valid_target].copy()
    y_sub = y_full.loc[valid_target].copy()

    if not pd.api.types.is_numeric_dtype(y_sub):
        print(f"Skipped {target}: target is not numeric.", flush=True)
        continue

    y_sub = y_sub.astype(float)

    if STRICT_DROP_NA_FOR_PREDICTORS:
        valid_predictors = x_sub.notna().all(axis=1)
        x_sub = x_sub.loc[valid_predictors]
        y_sub = y_sub.loc[valid_predictors]

    if len(x_sub) < 5:
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
        f"Training samples: {len(x_train)} | Test samples: {len(x_test)}",
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

    # Step 4.3: Tune hyperparameters with 5-fold cross-validation.
    cv = KFold(
        n_splits=max(2, min(CV_FOLDS, len(x_train))),
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    base_model = XGBRegressor(
        objective="reg:squarederror",
        tree_method=TREE_METHOD,
        n_estimators=N_ESTIMATORS_MAX,
        n_jobs=N_JOBS,
        random_state=RANDOM_SEED,
    )

    search = GridSearchCV(
        estimator=base_model,
        param_grid=PARAM_GRID,
        scoring="r2",
        cv=cv,
        n_jobs=N_JOBS,
        refit=True,
        verbose=0,
    )
    search.fit(x_train, y_train)

    best_params = search.best_params_
    best_cv_score = float(search.best_score_)

    # Step 4.4: Fit the final model with early stopping.
    x_fit, x_valid, y_fit, y_valid = train_test_split(
        x_train,
        y_train,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_SEED,
    )

    final_model = XGBRegressor(
        objective="reg:squarederror",
        tree_method=TREE_METHOD,
        n_estimators=N_ESTIMATORS_MAX,
        n_jobs=N_JOBS,
        random_state=RANDOM_SEED,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        **best_params,
    )

    final_model.fit(
        x_fit,
        y_fit,
        eval_set=[(x_valid, y_valid)],
        verbose=False,
    )

    best_iteration = getattr(final_model, "best_iteration", None)
    effective_trees = None if best_iteration is None else int(best_iteration) + 1

    # Step 4.5: Evaluate training and test performance.
    y_train_pred = final_model.predict(x_train)
    y_test_pred = final_model.predict(x_test)

    train_metrics = {
        "R2": float(r2_score(y_train, y_train_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
        "MAE": float(mean_absolute_error(y_train, y_train_pred)),
    }

    test_metrics = {
        "R2": float(r2_score(y_test, y_test_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
        "MAE": float(mean_absolute_error(y_test, y_test_pred)),
    }

    # Step 4.6: Estimate a 95% bootstrap confidence interval for test R2.
    boot_mean, ci_low, ci_high = bootstrap_r2(
        y_test,
        y_test_pred,
        n_bootstraps=N_BOOTSTRAPS,
        alpha=ALPHA_CI,
        seed=RANDOM_SEED,
    )
    test_metrics.update(
        {
            "R2_bootstrap_mean": boot_mean,
            "R2_CI95_low": ci_low,
            "R2_CI95_high": ci_high,
        }
    )

    # Step 4.7: Calculate gain and permutation feature importance.
    feature_names = list(x_sub.columns)
    gain_importance = np.zeros(len(feature_names), dtype=float)

    try:
        gain_dict = final_model.get_booster().get_score(importance_type="gain")
        for i, feature in enumerate(feature_names):
            if feature in gain_dict:
                gain_importance[i] = gain_dict[feature]

        if np.allclose(gain_importance, 0):
            for i in range(len(feature_names)):
                key = f"f{i}"
                if key in gain_dict:
                    gain_importance[i] = gain_dict[key]
    except Exception:
        pass

    permutation = permutation_importance(
        final_model,
        x_test,
        y_test,
        n_repeats=N_REPEATS_PERM,
        random_state=RANDOM_SEED,
        n_jobs=N_JOBS,
        scoring="r2",
    )

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "gain_importance": gain_importance,
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
        "train_metrics": train_metrics,
        "cv": {
            "cv_folds": int(cv.get_n_splits()),
            "scorer": "r2",
            "best_score": best_cv_score,
            "best_params": best_params,
            "n_estimators_upper_bound": N_ESTIMATORS_MAX,
            "n_estimators_effective": effective_trees,
        },
        "test_metrics": test_metrics,
        "delta_R2_test_minus_train": float(
            test_metrics["R2"] - train_metrics["R2"]
        ),
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
        os.path.join(target_dir, f"{safe_target}_importances.txt"),
        sep="\t",
        index=False,
    )

    with open(
        os.path.join(target_dir, f"{safe_target}_metrics.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    dump(
        final_model,
        os.path.join(target_dir, f"{safe_target}_XGBoostModel.joblib"),
    )

    summary[target] = metrics
    print(
        f"Completed {target} | Test R2 = {test_metrics['R2']:.3f} | "
        f"CV best R2 = {best_cv_score:.3f}",
        flush=True,
    )


# =============================================================================
# 5. Save the combined summary
# =============================================================================
with open(
    os.path.join(OUTDIR, "xgboost_bacteria_summary.json"),
    "w",
    encoding="utf-8",
) as handle:
    json.dump(summary, handle, ensure_ascii=False, indent=2)

print(f"\nAll targets completed. Results saved to: {OUTDIR}", flush=True)
