"""Train the model and persist the resulting artifact."""

import pandas as pd
import numpy as np
import mlflow
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
import optuna
import matplotlib.pyplot as plt

NUMERIC = ["GrLivArea", "TotalBsmtSF", "GarageArea", "YearBuilt", "OverallQual"]
CATEGORICAL = ["MSZoning"]
TARGET = "SalePrice"

df = pd.read_csv("data/ames_train.csv")
X = df[NUMERIC + CATEGORICAL]
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("house-price")

with mlflow.start_run(run_name="baseline-ridge"):
    num_prep = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_prep = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    prep_ridge = ColumnTransformer([
        ("num", num_prep, NUMERIC),
        ("cat", cat_prep, CATEGORICAL)
    ])
    baseline_model = Pipeline([
        ("prep", prep_ridge),
        ("model", Ridge(alpha=1.0))
    ])
    
    baseline_model.fit(X_train, y_train)
    preds = baseline_model.predict(X_test)
    
    mlflow.log_metrics({
        "test_r2": r2_score(y_test, preds),
        "test_mae": mean_absolute_error(y_test, preds),
        "test_rmse": mean_squared_error(y_test, preds, squared=False),
        "test_mape": mean_absolute_percentage_error(y_test, preds)
    })

def build_model(params: dict) -> Pipeline:
    prep = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([
        ("prep", prep),
        ("model", TransformedTargetRegressor(
            regressor=XGBRegressor(**params, random_state=42),
            func=np.log1p, 
            inverse_func=np.expm1,
        )),
    ])

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0)
    }
    
    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
        mlflow.log_params(params)
        model = build_model(params)
        
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")
        cv_r2 = cv_scores.mean()
        
        mlflow.log_metric("cv_r2", cv_r2)
        return cv_r2

with mlflow.start_run(run_name="xgboost-tuning") as parent_run:
    sampler = optuna.samplers.TPESampler(seed=42) 
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=20) 

with mlflow.start_run(run_name="xgboost-final") as run:
    final_model = build_model(study.best_params)
    final_model.fit(X_train, y_train)
    preds = final_model.predict(X_test)
    
    mlflow.log_params(study.best_params)
    mlflow.log_metrics({
        "test_r2": r2_score(y_test, preds),
        "test_mae": mean_absolute_error(y_test, preds),
        "test_rmse": mean_squared_error(y_test, preds, squared=False),
        "test_mape": mean_absolute_percentage_error(y_test, preds)
    })
    
    dataset = mlflow.data.from_pandas(df, targets=TARGET)
    mlflow.log_input(dataset, context="train")
    
    row_count = len(df)
    skew_before = y_train.skew()
    skew_after = np.log1p(y_train).skew()
    
    mlflow.log_metrics({
        "row_count": row_count,
        "skew_before": skew_before,
        "skew_after": skew_after
    })
    
    stats_dict = df.describe().to_dict()
    mlflow.log_dict(stats_dict, "data_stats.json")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.hist(y_train, bins=50)
    ax1.set_title("Target (original)")
    ax2.hist(np.log1p(y_train), bins=50)
    ax2.set_title("Target (log1p)")
    mlflow.log_figure(fig, "target_distributions.png")
    
    X_train.to_parquet("reference_data.parquet")
    mlflow.log_artifact("reference_data.parquet")
    
    signature = infer_signature(X_train, final_model.predict(X_train))
    mlflow.sklearn.log_model(final_model, name="model", signature=signature)
    
    version = mlflow.register_model(f"runs:/{run.info.run_id}/model", "house-price")
    MlflowClient().set_registered_model_alias("house-price", "prod", version.version)
    print(f"Successfully registered model 'house-price' version {version.version} with alias 'prod'")