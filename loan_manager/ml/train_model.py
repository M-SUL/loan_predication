"""Training script for loan approval classifier.

Run this module as a script:

```bash
python -m loan_manager.ml.train_model --data-file path/to/train.csv
```
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif

from .features import add_features

# Paths
_PACKAGE_ROOT = Path(__file__).resolve().parent
_MODEL_PATH = _PACKAGE_ROOT / "model.joblib"
_METRICS_PATH = _PACKAGE_ROOT / "metrics.json"


def load_data(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    # Assume target column is named 'Loan_Status' (Y/N)
    y = df["Loan_Status"].map({"Y": 1, "N": 0})
    X = df.drop(columns=["Loan_Status", "Col_num", "Loan_ID"])
    return X, y


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    # Define parameter grid for GridSearchCV
    param_grid = {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__learning_rate": [0.01, 0.05, 0.1],
        "classifier__max_depth": [3, 4, 5],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__subsample": [0.8, 0.9, 1.0],
    }

    clf = GradientBoostingClassifier(random_state=42)

    pipeline = Pipeline(
        steps=[
            ("feature_engineering", FunctionTransformer(add_features)),
            ("preprocessor", preprocessor),
            ("feature_selection", SelectKBest(f_classif, k="all")),
            ("classifier", clf),
        ]
    )

    return pipeline, param_grid


def train(csv_path: Path):
    X, y = load_data(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline, param_grid = build_pipeline(X)

    # Perform GridSearchCV
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=1
    )

    print("Training model with GridSearchCV...")
    grid_search.fit(X_train, y_train)

    # Get best model
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")

    # Make predictions
    y_pred = best_model.predict(X_test)

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred),
        "best_parameters": grid_search.best_params_,
        "feature_importance": dict(
            zip(X.columns, best_model.named_steps["classifier"].feature_importances_)
        ),
    }

    # Save artifacts
    joblib.dump(best_model, _MODEL_PATH)
    _METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print("\nModel saved to", _MODEL_PATH)
    print("\nMetrics:", json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Train loan approval model")
    parser.add_argument(
        "--data-file",
        type=Path,
        required=True,
        help="CSV dataset path",
    )
    args = parser.parse_args()
    train(args.data_file)


if __name__ == "__main__":
    main()
