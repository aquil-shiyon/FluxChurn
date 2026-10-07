"""Model service - loads and manages ML models and data for the API layer.

Singleton pattern: models and data are loaded once and cached.
This service provides the bridge between API routes and ML modules.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from ml.config import PROJECT_ROOT, get_feature_config, get_prediction_config
from ml.data_contract import RiskBand
from ml.explainability import ShapExplainer
from ml.features import generate_features
from ml.inference import (
    get_risk_distribution,
    predict_single,
    simulate_scenario,
)
from ml.ingestion import load_raw_data
from ml.monitoring import compute_feature_drift
from ml.validation import get_data_quality_summary

logger = logging.getLogger(__name__)


class ModelService:
    """Manages model lifecycle and prediction services."""

    def __init__(self):
        self._production_model: Optional[Pipeline] = None
        self._all_models: dict[str, Pipeline] = {}
        self._predictions: Optional[pd.DataFrame] = None
        self._features: Optional[pd.DataFrame] = None
        self._raw_data: Optional[pd.DataFrame] = None
        self._training_features: Optional[pd.DataFrame] = None
        self._explainer: Optional[ShapExplainer] = None
        self._evaluation_results: Optional[dict] = None
        self._model_comparison: Optional[dict] = None
        self._feature_stats: Optional[dict] = None

    def load(self) -> None:
        """Load all models and data. Called on startup."""
        logger.info("Loading models and data...")
        models_dir = PROJECT_ROOT / "data" / "models"
        processed_dir = PROJECT_ROOT / "data" / "processed"

        # Load production model
        prod_path = models_dir / "production_model.joblib"
        if prod_path.exists():
            self._production_model = joblib.load(prod_path)
            logger.info("Production model loaded")
        else:
            logger.warning(f"Production model not found at {prod_path}")

        # Load all benchmark models
        for model_file in models_dir.glob("*_model.joblib"):
            if model_file.stem == "production_model":
                continue
            key = model_file.stem.replace("_model", "")
            self._all_models[key] = joblib.load(model_file)
            logger.info(f"Loaded model: {key}")

        # Load raw data and generate features
        try:
            self._raw_data = load_raw_data()
            self._features = generate_features(self._raw_data)
            logger.info(f"Loaded {len(self._features)} customer records")
        except Exception as e:
            logger.warning(f"Could not load raw data: {e}")

        # Load predictions
        pred_path = processed_dir / "latest_predictions.parquet"
        if pred_path.exists():
            self._predictions = pd.read_parquet(pred_path)
            logger.info(f"Loaded {len(self._predictions)} predictions")

        # Load training features for monitoring
        train_feat_path = processed_dir / "training_features.parquet"
        if train_feat_path.exists():
            self._training_features = pd.read_parquet(train_feat_path)

        # Load evaluation results
        eval_path = processed_dir / "evaluation_results.json"
        if eval_path.exists():
            with open(eval_path) as f:
                self._evaluation_results = json.load(f)

        # Load model comparison
        comp_path = processed_dir / "model_comparison.json"
        if comp_path.exists():
            with open(comp_path) as f:
                self._model_comparison = json.load(f)

        # Load feature statistics
        stats_path = processed_dir / "feature_statistics.json"
        if stats_path.exists():
            with open(stats_path) as f:
                self._feature_stats = json.load(f)

        # Initialize SHAP explainer
        if self._production_model and self._features is not None:
            try:
                config = get_feature_config()
                feature_cols = config.numerical_features + config.categorical_features
                X = self._features[feature_cols]
                self._explainer = ShapExplainer(
                    self._production_model, X, model_version="1.0.0"
                )
                logger.info("SHAP explainer initialized")
            except Exception as e:
                logger.warning(f"Could not initialize SHAP explainer: {e}")

        logger.info("Model service ready")

    @property
    def is_ready(self) -> bool:
        return self._production_model is not None

    def get_overview(self) -> dict[str, Any]:
        """Get risk overview data for the dashboard."""
        if self._predictions is None:
            return {"error": "No predictions available. Run the training pipeline first."}

        risk_dist = get_risk_distribution(self._predictions)
        config = get_prediction_config()

        return {
            "prediction_horizon_days": config.prediction_horizon_days,
            "model_version": "1.0.0",
            "last_scored": str(self._predictions["prediction_timestamp"].iloc[0]) if ("prediction_timestamp" in self._predictions.columns and len(self._predictions) > 0) else None,
            "total_customers": risk_dist.get("total_eligible", len(self._predictions)),
            "risk_distribution": risk_dist.get("bands", {}),
            **risk_dist,
        }

    def get_risk_distribution(self) -> dict[str, Any]:
        """Get risk band distribution."""
        if self._predictions is None:
            return {"error": "No predictions available"}
        return get_risk_distribution(self._predictions)

    def get_drivers(self) -> dict[str, Any]:
        """Get global predictive drivers (SHAP-based)."""
        if self._explainer is None or self._features is None:
            return {"error": "Explainer or features not available"}

        config = get_feature_config()
        feature_cols = config.numerical_features + config.categorical_features

        try:
            X = self._features[feature_cols].sample(
                n=min(500, len(self._features)), random_state=42
            )
            return self._explainer.explain_global(X)
        except Exception as e:
            logger.error(f"Failed to compute drivers: {e}")
            return {"error": str(e)}

    def get_customer_risk(self, customer_id: str) -> dict[str, Any]:
        """Get risk details for a specific customer."""
        if self._predictions is None or self._features is None:
            return {"error": "No data available"}

        # Find customer in predictions
        pred_row = self._predictions[self._predictions["customer_id"] == customer_id]
        if pred_row.empty:
            return {"error": f"Customer {customer_id} not found"}

        # Get customer features
        feat_row = self._features[self._features["customer_id"] == customer_id]
        if feat_row.empty:
            return {"error": f"Features not found for customer {customer_id}"}

        pred = pred_row.iloc[0].to_dict()
        feat = feat_row.iloc[0].to_dict()

        return {
            "customer_id": customer_id,
            "churn_probability": float(pred.get("churn_probability", 0.0)),
            "risk_band": pred.get("risk_band", "low"),
            "prediction_horizon_days": int(pred.get("prediction_horizon_days", 30)),
            "model_version": str(pred.get("model_version", "1.0.0")),
            "prediction_timestamp": str(pred.get("prediction_timestamp", "")),
            "monthly_fee": float(pred.get("monthly_fee", feat.get("monthly_fee", 12.99))),
            "mrr_at_risk": float(pred.get("mrr_at_risk", round(float(pred.get("churn_probability", 0.0)) * float(feat.get("monthly_fee", 12.99)), 2))),
            "arr_at_risk": float(pred.get("arr_at_risk", round(float(pred.get("mrr_at_risk", 0.0)) * 12, 2))),
            "churn_type_risk": str(pred.get("churn_type_risk", "voluntary_engagement")),
            "features": {
                "age": int(feat.get("age", 0)),
                "watch_hours": float(feat.get("watch_hours", 0)),
                "last_login_days": int(feat.get("last_login_days", 0)),
                "monthly_fee": float(feat.get("monthly_fee", 0)),
                "number_of_profiles": int(feat.get("number_of_profiles", 0)),
                "avg_watch_time_per_day": float(feat.get("avg_watch_time_per_day", 0)),
                "subscription_type": str(feat.get("subscription_type", "")),
                "region": str(feat.get("region", "")),
                "device": str(feat.get("device", "")),
                "payment_method": str(feat.get("payment_method", "")),
                "gender": str(feat.get("gender", "")),
                "favorite_genre": str(feat.get("favorite_genre", "")),
                "watch_hours_per_profile": float(feat.get("watch_hours_per_profile", 0)),
                "fee_per_watch_hour": float(feat.get("fee_per_watch_hour", 0)),
                "login_recency_score": float(feat.get("login_recency_score", 0)),
            },
        }

    def get_customer_explanation(self, customer_id: str) -> dict[str, Any]:
        """Get SHAP explanation for a customer's prediction."""
        if self._explainer is None:
            return {"error": "Explainer not available"}
        if self._features is None:
            return {"error": "No feature data available"}

        feat_row = self._features[self._features["customer_id"] == customer_id]
        if feat_row.empty:
            return {"error": f"Customer {customer_id} not found"}

        config = get_feature_config()
        feature_cols = config.numerical_features + config.categorical_features

        try:
            X = feat_row[feature_cols]
            explanation = self._explainer.explain_local(X, customer_id)
            return explanation.model_dump()
        except Exception as e:
            logger.error(f"Explanation failed for {customer_id}: {e}")
            return {"error": str(e)}

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        """Online prediction endpoint."""
        if self._production_model is None:
            return {"error": "Model not available"}

        config = get_feature_config()
        feature_cols = config.numerical_features + config.categorical_features

        try:
            X = pd.DataFrame([features])[feature_cols]
            result = predict_single(
                self._production_model, X,
                customer_id=features.get("customer_id", "unknown"),
                model_version="1.0.0",
            )
            return result.model_dump()
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {"error": str(e)}

    def simulate(self, customer_id: str, scenario: dict[str, Any]) -> dict[str, Any]:
        """Run a what-if simulation."""
        if self._production_model is None:
            return {"error": "Model not available"}
        if self._features is None:
            return {"error": "No feature data available"}

        feat_row = self._features[self._features["customer_id"] == customer_id]
        if feat_row.empty:
            return {"error": f"Customer {customer_id} not found"}

        config = get_feature_config()
        feature_cols = config.numerical_features + config.categorical_features

        # Baseline features
        baseline = feat_row[feature_cols].copy()

        # Apply scenario modifications
        scenario_df = baseline.copy()
        for key, value in scenario.items():
            if key in scenario_df.columns:
                scenario_df[key] = value

        try:
            result = simulate_scenario(
                self._production_model,
                baseline,
                scenario_df,
                customer_id,
                model_version="1.0.0",
            )
            return result.model_dump()
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            return {"error": str(e)}

    def get_cohorts(self, group_by: str = "subscription_type") -> dict[str, Any]:
        """Get cohort analysis data."""
        if self._predictions is None:
            return {"error": "No data available"}

        if group_by in self._predictions.columns:
            merged = self._predictions
        else:
            source_df = self._raw_data if (self._raw_data is not None and group_by in self._raw_data.columns) else self._features
            if source_df is None or group_by not in source_df.columns:
                return {
                    "group_by": group_by,
                    "cohorts": {},
                    "available_dimensions": [
                        "subscription_type", "region", "device",
                        "payment_method", "gender", "favorite_genre",
                    ],
                }

            # Merge predictions with features
            merged = self._predictions.merge(
                source_df[["customer_id", group_by]],
                on="customer_id",
                how="left",
            )

        cohorts = {}
        for group_val in merged[group_by].dropna().unique():
            group_data = merged[merged[group_by] == group_val]
            risk_dist = {}
            for band in RiskBand:
                count = (group_data["risk_band"] == band.value).sum()
                risk_dist[band.value] = {
                    "count": int(count),
                    "percentage": round(count / len(group_data) * 100, 1) if len(group_data) > 0 else 0,
                }

            cohorts[str(group_val)] = {
                "total": len(group_data),
                "avg_churn_probability": round(float(group_data["churn_probability"].mean()), 4),
                "risk_distribution": risk_dist,
            }

        return {
            "group_by": group_by,
            "cohorts": cohorts,
            "available_dimensions": [
                "subscription_type", "region", "device",
                "payment_method", "gender", "favorite_genre",
            ],
        }

    def get_models(self) -> dict[str, Any]:
        """Get model metadata."""
        if self._model_comparison is None:
            return {"error": "No model comparison data available"}
        return self._model_comparison

    def get_model_metrics(self, model_version: str = "1.0.0") -> dict[str, Any]:
        """Get metrics for a specific model version."""
        if self._evaluation_results is None:
            return {"error": "No evaluation results available"}
        return self._evaluation_results

    def get_data_quality(self) -> dict[str, Any]:
        """Get data quality monitoring results."""
        if self._raw_data is None:
            return {"error": "No data available"}
        return get_data_quality_summary(self._raw_data)

    def get_drift(self) -> dict[str, Any]:
        """Get feature drift analysis."""
        ref_df = self._training_features if self._training_features is not None else self._features
        if ref_df is None or self._features is None:
            return {
                "status": "insufficient_data",
                "message": "Insufficient historical observations for drift analysis",
                "features": [],
            }

        raw_drift = compute_feature_drift(ref_df, self._features)
        drift_list = []
        for feat, metrics in raw_drift.get("features", {}).items():
            drift_list.append({
                "feature": feat,
                "psi": metrics.get("psi", 0.0),
                "ks_statistic": metrics.get("ks_statistic", 0.0),
                "threshold": metrics.get("psi_threshold", 0.2),
                "status": "drift_detected" if metrics.get("is_drifted") else "stable",
            })

        is_drifted = len(raw_drift.get("drifted_features", [])) > 0
        return {
            "status": "drift_detected" if is_drifted else "stable",
            "message": "No significant population drift detected across monitored features." if not is_drifted else f"Drift detected in {len(raw_drift.get('drifted_features', []))} features",
            "features": drift_list,
            "raw_drift": raw_drift,
        }

    def get_performance(self) -> dict[str, Any]:
        """Get model performance monitoring."""
        if self._evaluation_results is None:
            return {"error": "No evaluation results available"}

        return {
            "test_metrics": self._evaluation_results.get("test_metrics", {}),
            "validation_metrics": self._evaluation_results.get("validation_metrics", {}),
            "calibration": self._evaluation_results.get("calibration", {}),
            "model_selection": self._evaluation_results.get("model_selection", {}),
        }

    def get_customer_list(
        self,
        risk_band: Optional[str] = None,
        churn_type: Optional[str] = None,
        sort_by: str = "churn_probability",
        order: str = "desc",
        limit: int = 50,
        offset: int = 0,
        search: Optional[str] = None,
    ) -> dict[str, Any]:
        """Get filtered and sorted customer list with financial prioritization."""
        if self._predictions is None:
            return {"customers": [], "total": 0}

        df = self._predictions.copy()

        if risk_band:
            df = df[df["risk_band"] == risk_band]
        if churn_type and "churn_type_risk" in df.columns:
            df = df[df["churn_type_risk"] == churn_type]
        if search:
            df = df[df["customer_id"].str.contains(search, na=False)]

        total = len(df)
        sort_col = sort_by if sort_by in df.columns else "churn_probability"
        ascending = (order.lower() == "asc")
        df = df.sort_values(sort_col, ascending=ascending)
        df = df.iloc[offset:offset + limit]

        return {
            "customers": df.to_dict(orient="records"),
            "total": total,
            "limit": limit,
            "offset": offset,
        }


# Singleton instance
_model_service: Optional[ModelService] = None


def get_model_service() -> ModelService:
    """Get or create the singleton ModelService."""
    global _model_service
    if _model_service is None:
        _model_service = ModelService()
        _model_service.load()
    return _model_service
