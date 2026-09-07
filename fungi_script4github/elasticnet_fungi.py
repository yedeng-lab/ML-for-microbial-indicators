#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train Elastic Net regression models for fungal environmental variables.

Input format
------------
- OTU table: rows = OTUs/features, columns = samples
- Environmental table: rows = samples, columns = regression targets
- Climate table: rows = samples, columns = climate predictors

Workflow
--------
1. Load and align samples shared by all three input tables.
2. For each environmental target, keep samples with a measured target value.
3. Split data into 70% training and 30% testing sets.
4. Expand predictors into raw, log-transformed, and squared terms.
5. Tune Elastic Net hyperparameters with 5-fold cross-validation on the training set.
6. Evaluate the best model on the training and test sets.
7. Estimate a 95% bootstrap confidence interval for test-set R².
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
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

warnings.filterwarnings("ignore")

# =============================================================================
# 1. Configuration
# =============================================================================
OTU_PATH = r"/mnt/data/home/tycloud/lijiaying/Fungi/otu/20260119_core_fungi_0.05freq_unmerged.txt"
ENV_PATH = r"/mnt/data/home/tycloud/lijiaying/Fungi/env/20260119_fungi_env_unmerged.txt"
CLIMATE_PATH = r"/mnt/data/home/tycloud/lijiaying/Fungi/climate/20260119_fungi_BIO1_12_unmerged.txt"
OUTDIR = r"/mnt/data/home/tycloud/lijiaying/Fungi/machine_learning/ElasticNet_Fungi_BIO1_12_unmerged_1"

TEST_SIZE = 0.30
CV_FOLDS = 5
RANDOM_SEED = 42
N_JOBS = 8
N_REPEATS_PERM = 10
N_BOOTSTRAPS = 1000
ALPHA_CI = 0.05
STRICT_DROP_NA_FOR_PREDICTORS = True

PARAM_GRIDS = {
    "pH": {
        "enet__alpha": [0.05, 0.10, 0.20, 0.35, 0.46, 0.70],
        "enet__l1_ratio": [0.01, 0.05, 0.10, 0.20],
    },
    "SOC_content": {
        "enet__alpha": [0.3, 0.6, 1.0, 1.3, 2.0, 3.5, 6.0],
        "enet__l1_ratio": [0.70, 0.85, 0.90, 0.95],
    },
    "total_N_content": {
        "enet__alpha": [0.005, 0.01, 0.02, 0.03, 0.05, 0.10],
        "enet__l1_ratio": [0.70, 0.85, 0.90, 0.95],
    },
    "total_Ca": {
        "enet__alpha": [1, 10, 1e2, 1e3, 1e4, 1e5],
        "enet__l1_ratio": [0.01, 0.05, 0.10, 0.20, 0.30],
    },
    "total_P": {
        "enet__alpha": [1, 10, 1e2, 1e3, 1e4, 1e5],
        "enet__l1_ratio": [0.40, 0.60, 0.70, 0.80, 0.90],
    },
    "total_K": {
        "enet__alpha": [10, 1e2, 1e3, 1e4, 1e5, 1e6],
        "enet__l1_ratio": [0.01, 0.05, 0.10, 0.20],
    },
}


# =============================================================================
# 2. Helper functions
# =============================================================================
def log1p_eps(X, eps=1e-6):
    """Apply log(x + eps). This is used only for predictors positive in training data."""
    return np.log(X + eps)


def square(X):
    """Return squared predictor values."""
    return X ** 2


def build_feature_transformer(feature_names, X_train):
    """Create raw, log-transformed, and squared predictor terms.

    Log terms are created only for predictors that are strictly positive in the
    training set. The returned log feature list is reused when naming the fitted
    Elastic Net coefficients.
    """
    log_features = [feature for feature in feature_names if (X_train[feature] > 0).all()]

    transformers = [
        ("raw", "passthrough", feature_names),
        ("square", FunctionTransformer(square, validate=False), feature_names),
    ]
    if log_features:
        transformers.insert(
            1,
            (
                "log",
                FunctionTransformer(log1p_eps, kw_args={"eps": 1e-6}, validate=False),
                log_features,
            ),
        )

    return ColumnTransformer(transformers=transformers, remainder="drop"), log_features


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
otu = pd.read_csv(OTU_PATH, sep="\t", header=0, index_col=0)
env = pd.read_csv(ENV_PATH, sep="\t", header=0, index_col=0).round(2)
clim = pd.read_csv(CLIMATE_PATH, sep="\t", header=0, index_col=0)

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
# 4. Train one Elastic Net model per environmental target
# =============================================================================
for target in env_all.columns:
    print(f"\n===== Target: {target} =====", flush=True)

    # Step 4.1: Keep samples with a measured target value.
    y_full = env_all[target]
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
    print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")

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

    # Step 4.3: Build the feature-expansion + scaling + Elastic Net pipeline.
    feature_transformer, log_features = build_feature_transformer(
        X_sub.columns.tolist(), X_train
    )
    pipeline = Pipeline(
        [
            ("features", feature_transformer),
            ("scaler", StandardScaler()),
            ("enet", ElasticNet(max_iter=10000, random_state=RANDOM_SEED)),
        ]
    )

    # Step 4.4: Tune Elastic Net hyperparameters with 5-fold CV on training data.
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
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_

    # Step 4.5: Evaluate model performance.
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

    # Step 4.7: Calculate permutation importance on the test set.
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
            "feature": X_train.columns,
            "perm_importance_mean": perm.importances_mean,
            "perm_importance_std": perm.importances_std,
        }
    ).sort_values("perm_importance_mean", ascending=False)

    # Step 4.8: Save non-zero Elastic Net coefficients for expanded features.
    raw_features = X_sub.columns.tolist()
    expanded_feature_names = (
        [f"raw__{feature}" for feature in raw_features]
        + [f"log__{feature}" for feature in log_features]
        + [f"square__{feature}" for feature in raw_features]
    )

    coefficients = model.named_steps["enet"].coef_
    if len(expanded_feature_names) != len(coefficients):
        raise RuntimeError(
            "Expanded feature names do not match the fitted Elastic Net coefficients."
        )

    selected_features = pd.DataFrame(
        {"feature": expanded_feature_names, "coef": coefficients}
    )
    selected_features = selected_features.loc[selected_features["coef"] != 0]
    selected_features.to_csv(
        os.path.join(target_dir, f"{safe_target}_selected_features.txt"),
        sep="\t",
        index=False,
    )

    # Step 4.9: Save predictions, metrics, feature importance, and the fitted pipeline.
    train_predictions = pd.DataFrame(
        {"set": "train", "y_true": y_train.values, "y_pred": y_train_pred},
        index=X_train.index,
    )
    test_predictions = pd.DataFrame(
        {"set": "test", "y_true": y_test.values, "y_pred": y_test_pred},
        index=X_test.index,
    )
    predictions = pd.concat([train_predictions, test_predictions])
    predictions.index.name = "sample"
    predictions.to_csv(
        os.path.join(target_dir, f"{safe_target}_predictions.txt"), sep="\t"
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
            "best_score": float(search.best_score_),
            "best_params": search.best_params_,
        },
    }

    with open(os.path.join(target_dir, f"{safe_target}_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics_out, handle, ensure_ascii=False, indent=2)

    importance_df.to_csv(
        os.path.join(target_dir, f"{safe_target}_perm_importances_raw_features.txt"),
        sep="\t",
        index=False,
    )
    dump(model, os.path.join(target_dir, f"{safe_target}_ElasticNetModel.joblib"))

    summary[target] = metrics_out
    print(f"Completed: {target} | Test R2 = {test_metrics['R2']:.3f}")


# =============================================================================
# 5. Save project-level summary
# =============================================================================
with open(os.path.join(OUTDIR, "elasticnet_fungi_summary.json"), "w", encoding="utf-8") as handle:
    json.dump(summary, handle, ensure_ascii=False, indent=2)

print(f"\nAll targets completed. Output directory: {OUTDIR}")
