"""Model service for loading the trained credit approval classifier and
performing predictions.

This module expects a file `model.joblib` (or other name) located in the same
`ml` package directory. The model should follow the scikit-learn API and accept
pandas DataFrames in its `predict` / `predict_proba` methods.

Because we have not run training yet, the service is resilient to missing model
files – it will lazy-load and raise a friendly error if the model is not
available.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import pandas as pd

# Directory where this file lives
_PACKAGE_ROOT = Path(__file__).resolve().parent
_MODEL_PATH = _PACKAGE_ROOT / "model.joblib"
_METRICS_PATH = _PACKAGE_ROOT / "metrics.json"

# Singleton cache
_MODEL = None


def _ensure_model_loaded():
    global _MODEL  # noqa: PLW0603
    if _MODEL is None:
        if not _MODEL_PATH.exists():
            raise FileNotFoundError(
                "Trained model file not found. Please run the training script "
                "to generate 'model.joblib' first."
            )
        _MODEL = joblib.load(_MODEL_PATH)


def predict(features: Dict[str, Any]) -> Tuple[str, float]:
    """Predict approval (Yes/No) and probability.

    Parameters
    ----------
    features : dict
        Dictionary of model input features matching the training pipeline.

    Returns
    -------
    tuple
        (label, probability) where label is "Approved" or "Rejected".
    """
    _ensure_model_loaded()

    # Convert to DataFrame with single row
    df = pd.DataFrame([features])
    prob = float(_MODEL.predict_proba(df)[:, 1][0])
    label = "Approved" if prob >= 0.5 else "Rejected"
    return label, prob


def load_metrics() -> Dict[str, Any]:
    """Load evaluation metrics saved during training, if present."""
    if _METRICS_PATH.exists():
        return json.loads(_METRICS_PATH.read_text())
    return {}
