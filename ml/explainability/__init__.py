"""SHAP-based explainability module for FlixChurn.

Provides:
- Global explanations: top features influencing population predictions
- Local explanations: top features for a specific customer's prediction

IMPORTANT: SHAP values indicate feature contribution to the model's
prediction, NOT causal relationships. The UI must clearly communicate
this distinction (R-016).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from ml.data_contract import ExplanationFeature, PredictionExplanation
from ml.training import get_feature_names_from_pipeline

logger = logging.getLogger(__name__)


class ShapExplainer:
    """SHAP explainer for FlixChurn models.

    Wraps SHAP to provide both global and local explanations
    consistent with the model version used for predictions.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        X_background: pd.DataFrame,
        model_version: str = "1.0.0",
        max_background_samples: int = 100,
    ):
        """Initialize the SHAP explainer.

        Args:
            pipeline: Fitted model pipeline (preprocessor + classifier).
            X_background: Background data for SHAP (typically training data).
            model_version: Version string for tracking.
            max_background_samples: Maximum background samples for efficiency.
        """
        self.pipeline = pipeline
        self.model_version = model_version

        # Get feature names from the pipeline
        self.feature_names = get_feature_names_from_pipeline(pipeline)

        # Subsample background data for efficiency
        if len(X_background) > max_background_samples:
            bg_sample = X_background.sample(
                n=max_background_samples, random_state=42
            )
        else:
            bg_sample = X_background.copy()

        # Preprocess background data
        preprocessor = pipeline.named_steps["preprocessor"]
        self.bg_processed = preprocessor.transform(bg_sample)

        # Create SHAP explainer based on model type
        classifier = pipeline.named_steps["classifier"]
        classifier_type = type(classifier).__name__

        if classifier_type in ("XGBClassifier", "LGBMClassifier"):
            self.explainer = shap.TreeExplainer(classifier)
        elif classifier_type == "RandomForestClassifier":
            self.explainer = shap.TreeExplainer(classifier)
        else:
            # KernelExplainer for models without native SHAP support
            self.explainer = shap.LinearExplainer(
                classifier, self.bg_processed
            )

        logger.info(
            f"SHAP explainer initialized for {classifier_type} "
            f"with {len(self.feature_names)} features"
        )

    def explain_local(
        self,
        X: pd.DataFrame,
        customer_id: str,
        top_n: int = 10,
    ) -> PredictionExplanation:
        """Generate local SHAP explanation for a single customer.

        Args:
            X: Feature DataFrame for the customer (single row).
            customer_id: Customer identifier.
            top_n: Number of top features to return.

        Returns:
            PredictionExplanation with top contributing features.
        """
        preprocessor = self.pipeline.named_steps["preprocessor"]
        X_processed = preprocessor.transform(X)

        shap_values = self.explainer.shap_values(X_processed)

        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            # Binary classification: use positive class
            sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            sv = shap_values[:, :, 1]
        else:
            sv = shap_values

        if sv.ndim == 2:
            sv = sv[0]

        # Get base value
        if hasattr(self.explainer, "expected_value"):
            base_value = self.explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                base_value = base_value[1] if len(base_value) > 1 else base_value[0]
        else:
            base_value = 0.5

        # Get predicted probability
        predicted_value = float(self.pipeline.predict_proba(X)[:, 1][0])

        # Map SHAP values to feature names
        feature_contributions = []
        X_vals = X_processed[0] if hasattr(X_processed, '__getitem__') else X_processed

        for i, (fname, shap_val) in enumerate(zip(self.feature_names, sv)):
            try:
                fval = float(X_vals[i])
            except (IndexError, TypeError):
                fval = 0.0

            feature_contributions.append({
                "feature": fname,
                "feature_value": fval,
                "contribution": float(shap_val),
                "abs_contribution": abs(float(shap_val)),
                "direction": "increases_risk" if shap_val > 0 else "decreases_risk",
            })

        # Sort by absolute contribution and take top N
        feature_contributions.sort(key=lambda x: x["abs_contribution"], reverse=True)
        top_features = feature_contributions[:top_n]

        contributors = [
            ExplanationFeature(
                feature=f["feature"],
                feature_value=f["feature_value"],
                contribution=f["contribution"],
                direction=f["direction"],
            )
            for f in top_features
        ]

        return PredictionExplanation(
            customer_id=customer_id,
            model_version=self.model_version,
            base_value=float(base_value),
            predicted_value=predicted_value,
            top_contributors=contributors,
        )

    def explain_global(
        self,
        X: pd.DataFrame,
        top_n: int = 15,
    ) -> dict[str, Any]:
        """Generate global SHAP explanation across all predictions.

        Args:
            X: Feature DataFrame for the population.
            top_n: Number of top features to return.

        Returns:
            Dictionary with global feature importance.
        """
        preprocessor = self.pipeline.named_steps["preprocessor"]
        X_processed = preprocessor.transform(X)

        shap_values = self.explainer.shap_values(X_processed)

        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            sv = shap_values[:, :, 1]
        else:
            sv = shap_values

        # Mean absolute SHAP value per feature
        mean_abs_shap = np.abs(sv).mean(axis=0)

        # Create ranking
        feature_importance = []
        for i, fname in enumerate(self.feature_names):
            feature_importance.append({
                "feature": fname,
                "mean_abs_shap": float(mean_abs_shap[i]),
                "mean_shap": float(sv[:, i].mean()),
                "direction": "increases_risk" if sv[:, i].mean() > 0 else "decreases_risk",
            })

        feature_importance.sort(key=lambda x: x["mean_abs_shap"], reverse=True)

        return {
            "model_version": self.model_version,
            "n_samples": len(X),
            "top_features": feature_importance[:top_n],
            "all_features": feature_importance,
        }
