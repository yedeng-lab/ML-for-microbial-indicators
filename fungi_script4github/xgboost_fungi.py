#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train XGBoost regression models for fungal environmental variables.

Workflow
--------
1. Load OTU, environmental, and climate tables and align shared samples.
2. For each target, remove samples missing the target or predictor values.
3. Split data into 70% training and 30% testing sets.
4. Tune XGBoost hyperparameters with 5-fold cross-validation on the training set.
5. Refit the best parameter set with an internal validation split and early stopping.
6. Evaluate training and test performance.
7. Estimate a 95% bootstrap confidence interval for test-set R².
8. Calculate XGBoost gain importance and permutation importance.
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
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# =============================================================================
# 1. Configuration
# =============================================================================
OTU_PATH = r"/mnt/data/home/tycloud/lijiaying/Fungi/otu/20260119_core_fungi_0.05freq_unmerged.txt"
ENV_PATH = r"/mnt/data/home/tycloud/lijiaying/Fungi/env/20260119_fungi_env_unmerged.txt"
CLIMATE_PATH = r"/mnt/data/home/tycloud/lijiaying/Fungi/climate/20260119_fungi_BIO1_12_unmerged.txt"
OUTDIR = r"/mnt/data/home/tycloud/lijiaying/Fungi/machine_learning/XGBoost_Fungi_BIO1_12_unmerged_1"

SEP_OTU = "\t"
SEP_ENV = "\t"
SEP_CLIM = "\t"
STRICT_DROP_NA_FOR_PREDICTORS = True

TEST_SIZE = 0.30
CV_FOLDS = 5
VALIDATION_SIZE = 0.10
RANDOM_SEED = 42
N_JOBS = 8
N_REPEATS_PERM = 10
N_BOOTSTRAPS = 1000
ALPHA_CI = 0.05
N_ESTIMATORS_MAX = 2000
EARLY_STOPPING_ROUNDS = 100
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
# 2. Helper function
# =============================================================================
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
env = pd.read_csv(ENV_PATH, sep=SEP_ENV, header=0, index_col=0).round(2)
clim = pd.read_csv(CLIMATE_PATH, sep=SEP_CLIM, header=0, index_col=0)

common_samples = sorted(set(otu.columns) & set(env.index) & set(clim.index))
if len(common_samples) < 5:
    raise SystemExit(f"Too few aligned samples: {len(common_samples)}")

X_otu_all = otu[common_samples].T.astype(float)
X_clim_all = clim.loc[common_samples].apply(pd.to_numeric, errors="coerce")
X_all = pd.concat([X_otu_all, X_clim_all], axis=1)
env_all = env.loc[common_samples].copy()

os.makedirs(OUTDIR, exist_ok=True)
summary = {}


# =============================================================================
# 4. Train one XGBoost model per environmental target
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

    # Step 4.3: Tune XGBoost hyperparameters with 5-fold CV on training data.
    base_model = XGBRegressor(
        objective="reg:squarederror",
        tree_method=TREE_METHOD,
        n_estimators=N_ESTIMATORS_MAX,
        n_jobs=N_JOBS,
        random_state=RANDOM_SEED,
    )
    cv = KFold(
        n_splits=min(CV_FOLDS, len(X_train)),
        shuffle=True,
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
    search.fit(X_train, y_train)
    best_params = search.best_params_
    best_cv = float(search.best_score_)

    # Step 4.4: Refit the best parameter set with early stopping.
    X_train_inner, X_val, y_train_inner, y_val = train_test_split(
        X_train,
        y_train,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_SEED,
    )
    model = XGBRegressor(
        objective="reg:squarederror",
        tree_method=TREE_METHOD,
        n_estimators=N_ESTIMATORS_MAX,
        n_jobs=N_JOBS,
        random_state=RANDOM_SEED,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        **best_params,
    )
    model.fit(
        X_train_inner,
        y_train_inner,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    best_iteration = getattr(model, "best_iteration", None)
    effective_trees = int(best_iteration) + 1 if best_iteration is not None else None

    # Step 4.5: Evaluate training and test performance.
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

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

    # Step 4.6: Estimate a 95% bootstrap CI for test-set R².
    rng = np.random.RandomState(RANDOM_SEED)
    bootstrap_r2 = np.empty(N_BOOTSTRAPS, dtype=float)
    y_true = y_test.to_numpy()
    for i in range(N_BOOTSTRAPS):
        idx = rng.randint(0, len(y_true), len(y_true))
        bootstrap_r2[i] = r2_score(y_true[idx], y_test_pred[idx])

    ci_low, ci_high = np.percentile(
        bootstrap_r2,
        [100 * ALPHA_CI / 2, 100 * (1 - ALPHA_CI / 2)],
    )
    test_metrics.update(
        {
            "R2_bootstrap_mean": float(bootstrap_r2.mean()),
            "R2_CI95_low": float(ci_low),
            "R2_CI95_high": float(ci_high),
        }
    )

    # Step 4.7: Calculate XGBoost gain importance and permutation importance.
    booster = model.get_booster()
    gain_dict = booster.get_score(importance_type="gain")
    gain_importance = np.zeros(X_sub.shape[1], dtype=float)
    for i, feature in enumerate(X_sub.columns):
        if feature in gain_dict:
            gain_importance[i] = gain_dict[feature]
        elif f"f{i}" in gain_dict:
            gain_importance[i] = gain_dict[f"f{i}"]

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
            "gain_importance": gain_importance,
            "perm_importance_mean": perm.importances_mean,
            "perm_importance_std": perm.importances_std,
        }
    ).sort_values("perm_importance_mean", ascending=False)

    # Step 4.8: Save CV results, metrics, predictions, importance, and model.
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
        "test_metrics": test_metrics,
        "cv": {
            "cv_folds": int(cv.n_splits),
            "scorer": "r2",
            "best_score": best_cv,
            "best_params": best_params,
            "n_estimators_upper_bound": N_ESTIMATORS_MAX,
            "n_estimators_effective": effective_trees,
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
    dump(model, os.path.join(target_dir, f"{safe_target}_XGBoostModel.joblib"))

    summary[target] = metrics_out
    print(f"Completed: {target} | Test R2 = {test_metrics['R2']:.3f} | CV R2 = {best_cv:.3f}")


# =============================================================================
# 5. Save project-level summary
# =============================================================================
with open(os.path.join(OUTDIR, "xgboost_fungi_summary.json"), "w", encoding="utf-8") as handle:
    json.dump(summary, handle, ensure_ascii=False, indent=2)

print(f"\nAll targets completed. Output directory: {OUTDIR}")
