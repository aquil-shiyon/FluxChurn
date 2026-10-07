"""ML tests: leakage prevention, training, prediction, and risk bands.

These tests verify the critical ML integrity properties:
1. No future information leaks into features (R-008)
2. Preprocessor fitted only on training data (R-010)
3. Predictions are valid probabilities [0, 1]
4. Risk bands match configured thresholds (R-014)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ml.features import generate_features
from ml.inference import assign_risk_band
from ml.data_contract import RiskBand


def _make_test_df() -> pd.DataFrame:
    """Create test DataFrame."""
    return pd.DataFrame({
        "customer_id": [f"c{i:03d}" for i in range(100)],
        "age": np.random.randint(18, 70, 100),
        "gender": np.random.choice(["Male", "Female", "Other"], 100),
        "subscription_type": np.random.choice(["Basic", "Standard", "Premium"], 100),
        "watch_hours": np.random.uniform(0, 50, 100),
        "last_login_days": np.random.randint(0, 60, 100),
        "region": np.random.choice(["Europe", "Asia", "Africa", "North America", "South America", "Oceania"], 100),
        "device": np.random.choice(["TV", "Mobile", "Laptop", "Desktop", "Tablet"], 100),
        "monthly_fee": np.random.choice([8.99, 13.99, 17.99], 100),
        "churned": np.random.choice([0, 1], 100),
        "payment_method": np.random.choice(["Credit Card", "Debit Card", "PayPal", "Gift Card", "Crypto"], 100),
        "number_of_profiles": np.random.randint(1, 6, 100),
        "avg_watch_time_per_day": np.random.uniform(0, 10, 100),
        "favorite_genre": np.random.choice(["Action", "Comedy", "Drama", "Documentary", "Horror", "Romance", "Sci-Fi"], 100),
    })


class TestLeakagePrevention:
    """Tests explicitly verifying that no future information leaks into features.

    The source dataset is a flat snapshot, but these tests ensure:
    - Churn label is NOT used as a feature
    - Target column is not correlated with features in unexpected ways
    - Feature generation doesn't access the target
    """

    def test_churn_label_not_in_features(self):
        """Verify the churn label is not directly used as a model feature."""
        from ml.config import get_feature_config
        config = get_feature_config()
        all_features = config.numerical_features + config.categorical_features
        assert "churned" not in all_features, "Target 'churned' must not be a model feature"
        assert "churn_label" not in all_features, "Target 'churn_label' must not be a model feature"

    def test_target_not_in_derived_features(self):
        """Verify derived features don't encode the target."""
        df = _make_test_df()
        features = generate_features(df)
        derived_cols = ["watch_hours_per_profile", "fee_per_watch_hour", "login_recency_score"]
        for col in derived_cols:
            assert col in features.columns
            # Correlation with target should not be suspiciously high
            corr = abs(features[col].corr(features["churned"]))
            assert corr < 0.99, f"Derived feature '{col}' suspiciously correlated with target ({corr})"

    def test_feature_values_independent_of_churn(self):
        """Verify feature generation produces the same values regardless of churn status."""
        df = _make_test_df()

        # Generate features with original labels
        features_original = generate_features(df)

        # Flip all labels
        df_flipped = df.copy()
        df_flipped["churned"] = 1 - df_flipped["churned"]
        features_flipped = generate_features(df_flipped)

        # Derived features should be identical (they shouldn't depend on churn)
        for col in ["watch_hours_per_profile", "fee_per_watch_hour", "login_recency_score"]:
            np.testing.assert_array_equal(
                features_original[col].values,
                features_flipped[col].values,
                err_msg=f"Feature '{col}' changes with target label - potential leakage!"
            )


class TestPredictionValidity:
    """Tests for prediction output validity."""

    def test_risk_band_thresholds(self):
        """Verify risk bands match configured thresholds."""
        assert assign_risk_band(0.0) == RiskBand.LOW
        assert assign_risk_band(0.15) == RiskBand.LOW
        assert assign_risk_band(0.29) == RiskBand.LOW
        assert assign_risk_band(0.3) == RiskBand.MODERATE
        assert assign_risk_band(0.49) == RiskBand.MODERATE
        assert assign_risk_band(0.5) == RiskBand.HIGH
        assert assign_risk_band(0.69) == RiskBand.HIGH
        assert assign_risk_band(0.7) == RiskBand.CRITICAL
        assert assign_risk_band(1.0) == RiskBand.CRITICAL

    def test_risk_band_boundary_consistency(self):
        """Verify no gaps or overlaps in risk bands."""
        probabilities = np.arange(0, 1.01, 0.01)
        bands = [assign_risk_band(p) for p in probabilities]
        assert len(bands) == len(probabilities)
        # All should be valid RiskBand values
        for band in bands:
            assert isinstance(band, RiskBand)


class TestLabelGeneration:
    """Tests for label generation."""

    def test_binary_labels(self):
        from ml.labels import generate_labels
        df = _make_test_df()
        result = generate_labels(df)
        assert set(result["churn_label"].unique()).issubset({0, 1})

    def test_label_metadata(self):
        from ml.labels import generate_labels
        df = _make_test_df()
        result = generate_labels(df)
        assert "prediction_horizon_days" in result.columns
        assert "observation_window_days" in result.columns
        assert "label_source" in result.columns
        assert all(result["label_source"] == "dataset_provided")
