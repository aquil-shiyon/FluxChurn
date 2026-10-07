"""End-to-end integration tests for the ML pipeline and inference lifecycle."""
from pathlib import Path
import pytest
import pandas as pd

from ml.config import PROJECT_ROOT
from ml.ingestion import load_raw_data
from ml.validation import validate_raw_data
from ml.features import generate_features
from ml.inference import predict_batch, get_risk_distribution
from pipelines.inference.run_inference import run_batch_inference


def test_raw_data_ingestion_and_validation():
    df_raw = load_raw_data()
    assert len(df_raw) > 0
    assert "customer_id" in df_raw.columns

    report = validate_raw_data(df_raw)
    assert report.total_records == len(df_raw)
    assert report.passed


def test_feature_engineering_integrity():
    df_raw = load_raw_data()
    df_features = generate_features(df_raw)

    assert len(df_features) == len(df_raw)
    assert "watch_hours_per_profile" in df_features.columns
    assert "fee_per_watch_hour" in df_features.columns
    assert "login_recency_score" in df_features.columns

    # Check for NaN generation
    assert not df_features["watch_hours_per_profile"].isna().any()
    assert not df_features["fee_per_watch_hour"].isna().any()


def test_batch_inference_pipeline_execution(tmp_path):
    output_parquet = tmp_path / "test_predictions.parquet"
    results = run_batch_inference(output_path=output_parquet)

    assert len(results) > 0
    assert output_parquet.exists()

    # Verify persisted parquet
    loaded = pd.read_parquet(output_parquet)
    assert len(loaded) == len(results)
    assert "churn_probability" in loaded.columns
    assert "risk_band" in loaded.columns

    # Probability bounds
    assert (loaded["churn_probability"] >= 0.0).all()
    assert (loaded["churn_probability"] <= 1.0).all()
