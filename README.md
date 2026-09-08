# Microbiome-Based Prediction and Spatial Mapping of Soil Chemical Properties

## Overview

This repository contains machine-learning workflows for predicting soil chemical properties from microbial community composition and associated climatic variables.

Separate model-training pipelines are provided for **bacterial** and **fungal** datasets. Five regression algorithms are implemented for each microbial group:

- Elastic Net
- Random Forest
- Support Vector Regression (SVR)
- XGBoost
- LightGBM

The main prediction targets used for the spatial products are:

- **pH** — soil pH
- **SOC** — soil organic carbon
- **TN** — total nitrogen
- **TP** — total phosphorus

The overall objective is to determine how well microbial community information can be used to estimate soil chemical properties and to use trained models to fill missing soil-property observations. After gap filling, the combined observed and machine-learning-predicted values are spatially interpolated by **Kriging** to produce continuous raster maps across China.

---

## Project Workflow

The analysis follows the general workflow below:

```text
Microbial community data
        +
Selected climate variables
        +
Measured soil chemical properties
        │
        ▼
Sample alignment and data filtering
        │
        ▼
70% training set / 30% independent test set
        │
        ▼
Hyperparameter optimization
using 5-fold cross-validation
        │
        ▼
Machine-learning model training
        │
        ▼
Model evaluation
R² / RMSE / MAE
        │
        ▼
Bootstrap uncertainty estimation
and feature-importance analysis
        │
        ▼
Prediction of missing soil-property values
        │
        ▼
Observed + predicted point-level data
        │
        ▼
Kriging interpolation
        │
        ▼
GeoTIFF maps of pH, SOC, TN and TP
```

---

## Repository Contents

### Bacterial models

| File | Model |
|---|---|
| `elasticnet_bacteria.py` | Elastic Net regression |
| `randomforest_bacteria.py` | Random Forest regression |
| `svm_bacteria.py` | Support Vector Regression |
| `xgboost_bacteria.py` | XGBoost regression |
| `lightgbm_bacteria.py` | LightGBM regression |

### Fungal models

| File | Model |
|---|---|
| `elasticnet_fungi.py` | Elastic Net regression |
| `randomforest_fungi.py` | Random Forest regression |
| `svm_fungi.py` | Support Vector Regression |
| `xgboost_fungi.py` | XGBoost regression |
| `lightgbm_fungi.py` | LightGBM regression |

### Spatial raster products

| File | Description |
|---|---|
| `output/bacteria/China_pH_kriging_prediction.tif` | Kriging-interpolated soil pH |
| `output/bacteria/China_SOC_kriging_prediction.tif` | Kriging-interpolated soil organic carbon |
| `output/bacteria/China_TN_kriging_prediction.tif` | Kriging-interpolated soil total nitrogen |
| `output/bacteria/China_TP_kriging_prediction.tif` | Kriging-interpolated soil total phosphorus |
| `output/fungi/China_pH_kriging_prediction.tif` | Kriging-interpolated soil pH |
| `output/fungi/China_SOC_kriging_prediction.tif` | Kriging-interpolated soil organic carbon |
| `output/fungi/China_TN_kriging_prediction.tif` | Kriging-interpolated soil total nitrogen |
| `output/fungi/China_TP_kriging_prediction.tif` | Kriging-interpolated soil total phosphorus |


---

## Input Data

Each training script uses three main input tables.

### 1. Microbial community table

The microbial community table contains bacterial or fungal features.

Expected orientation:

```text
rows    = microbial features
columns = samples
```

The scripts transpose this table before model fitting so that samples become rows and microbial features become predictors.

### 2. Environmental table

The environmental table contains measured soil chemical properties.

Expected orientation:

```text
rows    = samples
columns = soil chemical variables
```

Only samples with an observed value for the current target variable are used for training and evaluation.

### 3. Climate table

The climate table contains climatic predictors associated with each sample.

Expected orientation:

```text
rows    = samples
columns = climate variables
```

The current scripts use selected bioclimatic predictors, including BIO1 and BIO12 in the supplied workflows.

Microbial and climate predictors are combined before model fitting.

---

## Sample Alignment and Data Filtering

For each script:

1. Sample IDs shared by the microbial, environmental and climate tables are identified.
2. The three input datasets are aligned to the same samples.
3. Samples without a measured value for the current response variable are excluded from model training.
4. Samples containing missing predictor values are removed when strict predictor filtering is enabled.
5. The remaining samples are divided into training and test sets.

Where possible, the scripts create quantile-based bins of the response variable and use them during the train/test split to maintain a broadly similar response distribution between the two subsets.

The default split is:

- **70% training**
- **30% independent testing**
- **random seed = 42**

---

## Machine-Learning Models

### Elastic Net

Elastic Net combines L1 and L2 regularization and is implemented using a scikit-learn pipeline.

The supplied Elastic Net workflows additionally expand the predictors into:

- raw features
- log-transformed features, when values are strictly positive
- squared features

The expanded predictors are standardized before model fitting.

Hyperparameters are optimized by cross-validation, mainly including:

- `alpha`
- `l1_ratio`

Non-zero fitted coefficients are saved as selected features.

---

### Random Forest

Random Forest regression is implemented with `RandomForestRegressor`.

The hyperparameter search includes combinations of parameters such as:

- number of trees
- maximum tree depth
- number or fraction of features considered per split
- minimum samples per leaf
- minimum samples required to split a node
- bootstrap sampling settings

Random Forest outputs include both:

- impurity-based feature importance
- permutation importance calculated on the independent test set

Out-of-bag performance is also recorded when enabled.

---

### Support Vector Regression

Support Vector Regression is implemented with `SVR`.

Predictors are standardized within a scikit-learn `Pipeline` before fitting, preventing information leakage from the test set during scaling.

The main tuned parameters include:

- kernel
- `C`
- `epsilon`
- `gamma`

Most workflows use the radial basis function (RBF) kernel, while some target-specific fungal models also allow linear kernels.

Permutation importance is calculated on the test set.

---

### XGBoost

XGBoost regression is implemented with `XGBRegressor`.

The hyperparameter search includes parameters such as:

- learning rate
- maximum tree depth
- minimum child weight
- row subsampling
- column subsampling
- L1 regularization
- L2 regularization
- minimum loss reduction (`gamma`)

After hyperparameter selection, the model is refitted using an internal validation subset and **early stopping**.

The scripts save both:

- gain-based feature importance
- test-set permutation importance

---

### LightGBM

LightGBM regression is implemented with `LGBMRegressor`.

The tuned parameters include:

- learning rate
- number of leaves
- maximum tree depth
- minimum child samples
- row subsampling
- column subsampling
- L1 regularization
- L2 regularization

The final model is refitted with an internal validation subset and **early stopping**.

The scripts save:

- gain-based feature importance
- test-set permutation importance

> **GPU note:** `lightgbm_bacteria.py` is currently configured with `DEVICE_TYPE = "cuda"`. If CUDA-enabled LightGBM is not available, change this setting to `DEVICE_TYPE = "cpu"` before running the script.

---

## Hyperparameter Optimization

Model hyperparameters are optimized using cross-validation on the training data.

The common configuration is:

- **5-fold cross-validation**
- **R² as the optimization score**
- fixed random seed for reproducibility

Depending on the model, either `GridSearchCV` or `RandomizedSearchCV` is used.

The independent test set is not used for hyperparameter selection.

---

## Model Evaluation

The trained models are evaluated using:

### Coefficient of determination

```text
R²
```

This describes the proportion of variation in the observed soil property explained by model predictions.

### Root mean squared error

```text
RMSE
```

RMSE gives greater weight to large prediction errors.

### Mean absolute error

```text
MAE
```

MAE summarizes the average absolute difference between observed and predicted values.

---

## Bootstrap Confidence Intervals

The scripts use bootstrap resampling of the independent test-set predictions to quantify uncertainty in model performance.

The standard configuration is:

```text
1,000 bootstrap replicates
95% confidence interval
```

All model workflows estimate uncertainty for test-set R², while the SVR workflows additionally bootstrap RMSE and MAE.

---

## Feature Importance

Feature importance is evaluated differently depending on the model.

| Model | Feature-importance outputs |
|---|---|
| Elastic Net | Non-zero coefficients + permutation importance |
| Random Forest | Impurity importance + permutation importance |
| SVR | Permutation importance |
| XGBoost | Gain importance + permutation importance |
| LightGBM | Gain importance + permutation importance |

Permutation importance is calculated on the independent test set and therefore provides a model-agnostic measure of how strongly model performance depends on individual predictors.

---

## Model Outputs

Each model creates a separate output directory for each environmental target.

Typical output files include:

```text
<target>_split.json
<target>_cv_results.txt
<target>_metrics.json
<target>_predictions.txt
<target>_importances.txt
<target>_<ModelName>Model.joblib
```

Depending on the algorithm, additional files may also be produced, for example:

```text
<target>_selected_features.txt
<target>_perm_importance_test.txt
```

### Output descriptions

- `*_split.json`  
  Sample IDs assigned to the training and test sets.

- `*_cv_results.txt`  
  Complete cross-validation results for the tested hyperparameter combinations.

- `*_metrics.json`  
  Training, cross-validation and independent test-set performance statistics.

- `*_predictions.txt`  
  Observed and predicted values.

- `*_importances.txt`  
  Model-specific and/or permutation feature importance.

- `*.joblib`  
  Serialized trained model for subsequent prediction.

A model-level summary JSON file is also generated after all available targets have been processed.

---

## Prediction of Missing Soil Properties

After model evaluation, trained models can be used to estimate soil chemical properties for samples where the corresponding measured value is missing but the required microbial and climatic predictors are available.

Conceptually, the completed dataset contains:

```text
measured value, if available
        +
machine-learning prediction, if the measurement is missing
```

These completed point-level soil-property datasets are then used for spatial interpolation.

The prediction step should use exactly the same predictor structure, feature names, preprocessing steps and model object used during training.

---

## Kriging and GeoTIFF Products

After machine-learning prediction of missing values, the combined observed and predicted point data are interpolated using **Kriging** to generate continuous spatial surfaces.

The supplied GeoTIFF files therefore represent the final spatial products derived from:

```text
observed soil measurements
        +
machine-learning-predicted missing values
        +
Kriging interpolation
```

The four mapped properties are:

- pH
- SOC
- TN
- TP

For the supplied raster files, the grids are:

- single-band GeoTIFF rasters
- `float32`
- approximately **5 km × 5 km** grid cells
- **967 × 807** pixels
- projected using a WGS84-based **Albers Equal Area** coordinate system
- central meridian: **105°E**
- standard parallels: **25°N** and **47°N**

The rasters can be opened in GIS software such as QGIS or ArcGIS, or analyzed programmatically with packages such as `rasterio`.

---

## Software Requirements

Core Python dependencies are:

```text
Python 3
numpy
pandas
scikit-learn
joblib
xgboost
lightgbm
```

A minimal installation can be created with:

```bash
pip install numpy pandas scikit-learn joblib xgboost lightgbm
```

For raster inspection or downstream spatial analysis in Python:

```bash
pip install rasterio
```

GPU-enabled LightGBM requires a compatible CUDA-enabled LightGBM installation. CPU execution is sufficient for reproducing the workflow if the device setting is changed accordingly.

---

## Running the Scripts

Before running a model, edit the configuration section near the top of the corresponding Python script.

At minimum, update:

```python
OTU_PATH = "path/to/microbial_table.txt"
ENV_PATH = "path/to/environment_table.txt"
CLIMATE_PATH = "path/to/climate_table.txt"
OUTDIR = "path/to/output_directory"
```

Then run, for example:

```bash
python randomforest_bacteria.py
```

or:

```bash
python xgboost_fungi.py
```

The script will automatically:

1. load the input tables,
2. align shared samples,
3. train a model for each available target,
4. optimize hyperparameters,
5. evaluate the fitted model,
6. calculate feature importance,
7. save predictions and statistics, and
8. serialize the trained model.

---

## Reproducibility Notes

- Random processes use a fixed seed (`42`) throughout the supplied scripts.
- Hyperparameter optimization is performed only on the training set.
- The 30% test set is retained for independent performance evaluation.
- Predictor scaling for SVR and Elastic Net is performed within scikit-learn pipelines.
- XGBoost and LightGBM use internal validation subsets for early stopping.
- Saved train/test sample IDs allow the original split to be reconstructed.
- Exact numerical results can depend on package versions, hardware, thread scheduling and GPU configuration.
- File paths in the scripts are project-specific and must be changed before the code is run on another system.
- The GeoTIFF products are downstream spatial products; the current repository scripts shown here focus on machine-learning model training rather than the Kriging implementation itself.

---

## Suggested Repository Structure

A clear organization for the repository is:

```text
.
├── models/
│   ├── bacteria/
│   │   ├── elasticnet_bacteria.py
│   │   ├── lightgbm_bacteria.py
│   │   ├── randomforest_bacteria.py
│   │   ├── svm_bacteria.py
│   │   └── xgboost_bacteria.py
│   │
│   └── fungi/
│       ├── elasticnet_fungi.py
│       ├── lightgbm_fungi.py
│       ├── randomforest_fungi.py
│       ├── svm_fungi.py
│       └── xgboost_fungi.py
│
├── maps/
│   ├── bacteria/
│   │   ├── China_pH_kriging_prediction.tif
│   │   ├── China_SOC_kriging_prediction.tif
│   │   ├── China_TN_kriging_prediction.tif
│   │   └── China_TP_kriging_prediction.tif
│   │
│   └── fungi/
│       ├── China_pH_kriging_prediction.tif
│       ├── China_SOC_kriging_prediction.tif
│       ├── China_TN_kriging_prediction.tif
│       └── China_TP_kriging_prediction.tif
│
└── README.md
```

This structure avoids filename conflicts when bacterial and fungal raster products have the same names.

---

## Scope

This repository is intended to provide a reproducible framework for:

1. quantifying relationships between soil microbial communities and soil chemical properties,
2. comparing multiple machine-learning algorithms,
3. predicting missing soil-property observations, and
4. generating spatially continuous soil-property products after Kriging interpolation.

The workflow can also be adapted to other microbial datasets or environmental variables by modifying the input tables, target variables and model configuration.

---

## Citation

If you use this repository, please cite the associated study once the publication information becomes available.

