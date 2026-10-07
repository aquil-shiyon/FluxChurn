"""Unit tests for feature engineering module."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ml.features import generate_features


def _make_test_df() -> pd.DataFrame:
    """Create test DataFrame mimicking normalized data."""
    return pd.DataFrame({
        "customer_id": ["c001", "c002", "c003", "c004"],
        "age": [25, 45, 60, 30],
        "gender": ["Male", "Female", "Other", "Male"],
        "subscription_type": ["Basic", "Standard", "Premium", "Basic"],
        "watch_hours": [10.0, 20.0, 0.0, 50.0],
        "last_login_days": [2, 30, 60, 0],
        "region": ["Europe", "Asia", "Africa", "North America"],
        "device": ["TV", "Mobile", "Laptop", "Desktop"],
        "monthly_fee": [8.99, 13.99, 17.99, 8.99],
        "churned": [0, 1, 1, 0],
        "payment_method": ["Credit Card", "PayPal", "Crypto", "Debit Card"],
        "number_of_profiles": [2, 3, 1, 5],
        "avg_watch_time_per_day": [1.5, 3.0, 0.0, 5.0],
        "favorite_genre": ["Action", "Drama", "Comedy", "Sci-Fi"],
    })


class TestDerivedFeatures:
    """Tests for derived feature calculations."""

    def test_watch_hours_per_profile(self):
        df = _make_test_df()
        result = generate_features(df)
        # 10 / 2 = 5.0
        assert result.loc[0, "watch_hours_per_profile"] == pytest.approx(5.0)
        # 20 / 3 ≈ 6.667
        assert result.loc[1, "watch_hours_per_profile"] == pytest.approx(20.0 / 3)
        # 50 / 5 = 10.0
        assert result.loc[3, "watch_hours_per_profile"] == pytest.approx(10.0)

    def test_fee_per_watch_hour(self):
        df = _make_test_df()
        result = generate_features(df)
        # 8.99 / (10 + 1) ≈ 0.817
        assert result.loc[0, "fee_per_watch_hour"] == pytest.approx(8.99 / 11, rel=0.01)
        # 17.99 / (0 + 1) = 17.99 (zero watch hours)
        assert result.loc[2, "fee_per_watch_hour"] == pytest.approx(17.99 / 1, rel=0.01)

    def test_login_recency_score_range(self):
        df = _make_test_df()
        result = generate_features(df)
        # Score should be between 0 and 1
        assert result["login_recency_score"].min() >= 0.0
        assert result["login_recency_score"].max() <= 1.0

    def test_login_recency_score_ordering(self):
        df = _make_test_df()
        result = generate_features(df)
        # Customer with 0 login days should have lowest recency score
        assert result.loc[3, "login_recency_score"] == 0.0
        # Customer with 60 login days should have highest recency score
        assert result.loc[2, "login_recency_score"] == 1.0


class TestFeatureConsistency:
    """Tests for feature engineering determinism."""

    def test_deterministic_output(self):
        df = _make_test_df()
        result1 = generate_features(df)
        result2 = generate_features(df)
        pd.testing.assert_frame_equal(result1, result2)

    def test_no_null_derived_features(self):
        df = _make_test_df()
        result = generate_features(df)
        derived = ["watch_hours_per_profile", "fee_per_watch_hour", "login_recency_score"]
        for col in derived:
            assert result[col].isnull().sum() == 0, f"NaN found in {col}"

    def test_original_columns_preserved(self):
        df = _make_test_df()
        result = generate_features(df)
        for col in df.columns:
            assert col in result.columns, f"Original column '{col}' missing"
