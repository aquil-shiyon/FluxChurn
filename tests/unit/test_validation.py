"""Unit tests for data validation module."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ml.validation import ValidationReport, validate_raw_data


def _make_valid_df() -> pd.DataFrame:
    """Create a minimal valid test DataFrame."""
    return pd.DataFrame({
        "customer_id": ["c001", "c002", "c003"],
        "age": [25, 45, 60],
        "gender": ["Male", "Female", "Other"],
        "subscription_type": ["Basic", "Standard", "Premium"],
        "watch_hours": [10.0, 20.0, 5.0],
        "last_login_days": [2, 10, 30],
        "region": ["Europe", "Asia", "Africa"],
        "device": ["TV", "Mobile", "Laptop"],
        "monthly_fee": [8.99, 13.99, 17.99],
        "churned": [0, 1, 0],
        "payment_method": ["Credit Card", "PayPal", "Crypto"],
        "number_of_profiles": [2, 3, 1],
        "avg_watch_time_per_day": [1.5, 3.0, 0.5],
        "favorite_genre": ["Action", "Drama", "Comedy"],
    })


class TestSchemaValidation:
    """Tests for required columns and schema checks."""

    def test_valid_data_passes(self):
        df = _make_valid_df()
        report = validate_raw_data(df)
        assert report.passed

    def test_missing_column_fails(self):
        df = _make_valid_df().drop(columns=["churned"])
        report = validate_raw_data(df)
        assert any(
            i.check == "required_columns" and i.severity == "error"
            for i in report.issues
        )

    def test_extra_column_is_info(self):
        df = _make_valid_df()
        df["extra_col"] = 1
        report = validate_raw_data(df)
        assert any(
            i.check == "extra_columns" and i.severity == "info"
            for i in report.issues
        )


class TestMissingValues:
    """Tests for missing value detection."""

    def test_no_missing_reports_info(self):
        df = _make_valid_df()
        report = validate_raw_data(df)
        assert any(
            i.check == "missing_values" and "No missing values" in i.message
            for i in report.issues
        )

    def test_missing_values_detected(self):
        df = _make_valid_df()
        df.loc[0, "watch_hours"] = None
        report = validate_raw_data(df)
        assert any(
            i.check == "missing_values" and "watch_hours" in i.message
            for i in report.issues
        )


class TestDuplicates:
    """Tests for duplicate detection."""

    def test_duplicate_customer_ids(self):
        df = _make_valid_df()
        df.loc[2, "customer_id"] = "c001"  # Duplicate
        report = validate_raw_data(df)
        assert any(
            i.check == "duplicate_customer_ids"
            for i in report.issues
        )


class TestNumericalRanges:
    """Tests for impossible numerical values."""

    def test_negative_watch_hours(self):
        df = _make_valid_df()
        df.loc[0, "watch_hours"] = -5.0
        report = validate_raw_data(df)
        assert any(
            i.check == "numerical_range" and "Watch hours" in i.message
            for i in report.issues
        )

    def test_age_below_minimum(self):
        df = _make_valid_df()
        df.loc[0, "age"] = 10
        report = validate_raw_data(df)
        assert any(
            i.check == "numerical_range" and "Age" in i.message
            for i in report.issues
        )


class TestCategoricalValues:
    """Tests for categorical validity."""

    def test_invalid_subscription_type(self):
        df = _make_valid_df()
        df.loc[0, "subscription_type"] = "Super Premium"
        report = validate_raw_data(df)
        assert any(
            i.check == "categorical_values" and "subscription_type" in i.message
            for i in report.issues
        )


class TestTargetVariable:
    """Tests for churn target validation."""

    def test_valid_binary_target(self):
        df = _make_valid_df()
        report = validate_raw_data(df)
        assert any(
            i.check == "class_distribution"
            for i in report.issues
        )

    def test_non_binary_target(self):
        df = _make_valid_df()
        df.loc[0, "churned"] = 2
        report = validate_raw_data(df)
        assert any(
            i.check == "target_variable" and i.severity == "error"
            for i in report.issues
        )
