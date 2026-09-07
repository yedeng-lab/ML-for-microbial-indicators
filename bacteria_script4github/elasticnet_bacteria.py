#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train Elastic Net regression models for environmental variables using bacterial
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
4. Expand predictors into raw, log-transformed, and squared terms.
5. Standardize predictors and tune Elastic Net with 5-fold cross-validation.
6. Evaluate the final model on training and test sets.
7. Estimate a 95% bootstrap confidence interval for test-set R2.
8. Calculate test-set permutation importance.
9. Save metrics, predictions, selected coefficients, split information, and model.

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
OTU_PATH = r"/mnt/data/home/tycloud/lijiaying/Bacteria/otu/20251216_core_bacteria_0.25freq_unmerged.txt"
ENV_PATH = r"/mnt/data/home/tycloud/lijiaying/Bacteria/env/bacteria_TP_env_unmerged.txt"
CLIMATE_PATH = r"/mnt/data/home/tycloud/lijiaying/Bacteria/climate/20251214_bacteria_BIO1_12_unmerged.txt"
OUTDIR = r"/mnt/data/home/tycloud/lijiaying/Bacteria/machine_learning/ElasticNet_Bacteria_BIO1_12_unmerged"

SEP_OTU = "\t"
SEP_ENV = "\t"
SEP_CLIM = "\t"

TEST_SIZE = 0.30
CV_FOLDS = 5
RANDOM_SEED = 42
N_JOBS = 8
N_REPEATS_PERM = 10
N_BOOTSTRAPS = 1000
ALPHA_CI = 0.05
STRICT_DROP_NA_FOR_PREDICTORS = True

PARAM_GRID_DEFAULT = {
    "enet__alpha": np.logspace(-3, 1, 10),
    "enet__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
}

PARAM_GRID_SOC = {
    "enet__alpha": np.logspace(-3, 0.5, 10),
    "enet__l1_ratio": [0.1, 0.2, 0.3, 0.4, 0.5],
}

PARAM_GRID_CN = {
    "enet__alpha": np.concatenate(
        [np.logspace(-3, -1, 5), np.logspace(-1, 0.5, 6)]
    ),
    "enet__l1_ratio": [0.3, 0.5, 0.7, 0.9],
}


# =============================================================================
# 2. Helper functions
# =============================================================================
def log_transform(x, eps=1e-6):
    """Apply log(x + eps)."""
    return np.log(x + eps)


def square_transform(x):
    """Apply x^2."""
    return x**2


def get_log_features(feature_names, x_reference):
    """Return features that are strictly positive in the training data."""
    return [feature for feature in feature_names if (x_reference[feature] > 0).all()]


def build_feature_transformer(feature_names, x_train):
    """Create raw, log-transformed, and squared predictor terms."""
    log_features = get_log_features(feature_names, x_train)

    transformers = [
        ("raw", "passthrough", feature_names),
        (
            "square",
            FunctionTransformer(square_transform, validate=False),
            feature_names,
        ),
    ]

    if log_features:
        transformers.insert(
            1,
            (
                "log",
                FunctionTransformer(
                    log_transform,
                    kw_args={"eps": 1e-6},
                    validate=False,
                ),
                log_features,
            ),
        )

    return ColumnTransformer(transformers=transformers, remainder="drop"), log_features


def choose_param_grid(target):
    """Select the target-specific Elastic Net hyperparameter grid."""
    if target == "SOC_g_kg":
        return PARAM_GRID_SOC
    if target == "C_N_ratio":
        return PARAM_GRID_CN
    return PARAM_GRID_DEFAULT


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
# 4. Train one Elastic Net model per environmental target
# =============================================================================
for target in env_all.columns:
    print(f"\n===== Target: {target} =====", flush=True)

    # Step 4.1: Keep samples with an observed target value.
    y_full = env_all[target]
    valid_target = y_full.notna()
    x_sub = x_all.loc[valid_target].copy()
    y_sub = y_full.loc[valid_target].astype(float).copy()

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

    # Step 4.3: Build nonlinear feature expansion and Elastic Net pipeline.
    raw_features = x_train.columns.tolist()
    feature_transformer, log_features = build_feature_transformer(
        raw_features, x_train
    )

    pipeline = Pipeline(
        [
            ("features", feature_transformer),
            ("scaler", StandardScaler()),
            (
                "enet",
                ElasticNet(max_iter=10000, random_state=RANDOM_SEED),
            ),
        ]
    )

    # Step 4.4: Tune hyperparameters with cross-validation on the training set.
    cv = KFold(
        n_splits=max(2, min(CV_FOLDS, len(x_train))),
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=choose_param_grid(target),
        scoring="r2",
        cv=cv,
        n_jobs=N_JOBS,
        refit=True,
    )
    search.fit(x_train, y_train)
    model = search.best_estimator_

    # Step 4.5: Evaluate training and test performance.
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

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

    # Step 4.7: Calculate permutation importance on the independent test set.
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
            "feature": x_train.columns,
            "perm_importance_mean": permutation.importances_mean,
            "perm_importance_std": permutation.importances_std,
        }
    ).sort_values("perm_importance_mean", ascending=False)

    # Step 4.8: Extract non-zero Elastic Net coefficients.
    enet_model = model.named_steps["enet"]
    expanded_feature_names = (
        [f"raw__{feature}" for feature in raw_features]
        + [f"log__{feature}" for feature in log_features]
        + [f"square__{feature}" for feature in raw_features]
    )

    if len(expanded_feature_names) != len(enet_model.coef_):
        raise RuntimeError(
            "Expanded feature names do not match the number of Elastic Net coefficients."
        )

    coefficient_df = pd.DataFrame(
        {"feature": expanded_feature_names, "coef": enet_model.coef_}
    )
    coefficient_df = coefficient_df.loc[coefficient_df["coef"] != 0]

    # Step 4.9: Save outputs.
    prediction_df = pd.concat(
        [
            pd.DataFrame(
                {
                    "set": "train",
                    "y_true": y_train.values,
                    "y_pred": y_train_pred,
                },
                index=x_train.index,
            ),
            pd.DataFrame(
                {
                    "set": "test",
                    "y_true": y_test.values,
                    "y_pred": y_test_pred,
                },
                index=x_test.index,
            ),
        ]
    )
    prediction_df.index.name = "sample"

    metrics = {
        "task": "regression",
        "n_samples": int(len(x_sub)),
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "cv": {
            "cv_folds": int(cv.get_n_splits()),
            "scorer": "r2",
            "best_score": float(search.best_score_),
            "best_params": search.best_params_,
        },
    }

    coefficient_df.to_csv(
        os.path.join(target_dir, f"{safe_target}_selected_features.txt"),
        sep="\t",
        index=False,
    )
    prediction_df.to_csv(
        os.path.join(target_dir, f"{safe_target}_predictions.txt"),
        sep="\t",
    )
    importance_df.to_csv(
        os.path.join(target_dir, f"{safe_target}_perm_importances.txt"),
        sep="\t",
        index=False,
    )
    pd.DataFrame(search.cv_results_).to_csv(
        os.path.join(target_dir, f"{safe_target}_cv_results.txt"),
        sep="\t",
        index=False,
    )
    with open(
        os.path.join(target_dir, f"{safe_target}_metrics.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    dump(model, os.path.join(target_dir, f"{safe_target}_ElasticNetModel.joblib"))

    summary[target] = metrics
    print(f"Completed {target} | Test R2 = {test_metrics['R2']:.3f}", flush=True)


# =============================================================================
# 5. Save the combined summary
# =============================================================================
with open(
    os.path.join(OUTDIR, "elasticnet_bacteria_summary.json"),
    "w",
    encoding="utf-8",
) as handle:
    json.dump(summary, handle, ensure_ascii=False, indent=2)

print(f"\nAll targets completed. Results saved to: {OUTDIR}", flush=True)
