"""Batch inference pipeline for FlixChurn.

Executes the automated batch scoring lifecycle:
1. Load raw/latest customer dataset
2. Run data validation & quality gate
3. Generate feature engineering matrix
4. Load active production model artifact
5. Generate calibrated churn probabilities and risk bands
6. Persist scored predictions artifact

Run: python -m pipelines.inference.run_inference
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from ml.config import PROJECT_ROOT, get_feature_config, get_prediction_config
from ml.features import generate_features
from ml.inference import get_risk_distribution, predict_batch
from ml.ingestion import load_raw_data
from ml.validation import validate_raw_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("inference_pipeline")


def run_batch_inference(
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
    model_path: str | Path | None = None,
) -> pd.DataFrame:
    """Execute end-to-end batch inference on latest customer population.

    Args:
        input_path: Optional path to raw dataset. Defaults to raw data directory.
        output_path: Optional path for latest predictions parquet. Defaults to data/processed.
        model_path: Optional path to model artifact. Defaults to data/models/model_production.joblib.

    Returns:
        DataFrame of scored customers with metadata and predictions.
    """
    start_time = datetime.now(timezone.utc)
    logger.info("=" * 60)
    logger.info("FLIXCHURN BATCH INFERENCE PIPELINE")
    logger.info("=" * 60)

    # 1. Ingestion
    logger.info("Step 1: Loading customer data...")
    df_raw = load_raw_data(input_path)
    logger.info(f"Loaded {len(df_raw)} customer records.")

    # 2. Validation
    logger.info("Step 2: Validating data quality...")
    val_report = validate_raw_data(df_raw)
    if not val_report.passed:
        logger.warning(f"Data validation reported errors: {val_report.error_count} errors, {val_report.warning_count} warnings")
    else:
        logger.info(f"Data validation passed: {val_report.summary()}")

    # 3. Feature Engineering
    logger.info("Step 3: Engineering feature table...")
    df_features = generate_features(df_raw)
    feature_config = get_feature_config()
    feature_cols = feature_config.all_feature_names
    logger.info(f"Generated {len(df_features)} records across {len(feature_cols)} features.")

    # 4. Load Production Model
    logger.info("Step 4: Loading production model artifact...")
    if model_path is None:
        model_path = PROJECT_ROOT / "data" / "models" / "production_model.joblib"
    else:
        model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Production model artifact not found at {model_path}. Train a model first.")

    model_pipeline = joblib.load(model_path)
    model_version = "1.0.0"

    # 5. Batch Scoring
    logger.info("Step 5: Generating predictions and risk classifications...")
    predictions = predict_batch(
        pipeline=model_pipeline,
        df=df_features,
        feature_columns=feature_cols,
        model_version=model_version,
    )

    # Merge features and raw dimensional context with predictions without column duplicates
    cols_to_add = [c for c in df_features.columns if c not in predictions.columns]
    merged_output = pd.concat([
        predictions,
        df_features[cols_to_add],
    ], axis=1)

    # Add raw categorical attributes if present in df_raw for cohort analytics
    cohort_cols = ["gender", "subscription_type", "region", "device", "payment_method", "favorite_genre"]
    for col in cohort_cols:
        if col in df_raw.columns and col not in merged_output.columns:
            merged_output[col] = df_raw[col].values

    # 6. Save Predictions Artifact
    if output_path is None:
        output_dir = PROJECT_ROOT / "data" / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "latest_predictions.parquet"
    else:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Step 6: Saving predictions to {output_file}...")
    merged_output.to_parquet(output_file, index=False)

    # Summary
    duration = (datetime.now(timezone.utc) - start_time).total_seconds()
    dist = get_risk_distribution(predictions)
    logger.info("-" * 60)
    logger.info(f"Batch Inference Finished in {duration:.2f}s")
    logger.info(f"Total Scored: {dist['total_eligible']}")
    logger.info(f"High/Critical Risk Customers: {dist['high_risk_count']}")
    logger.info(f"Average Churn Probability: {dist['avg_churn_probability']:.2%}")
    logger.info("=" * 60)

    return merged_output


if __name__ == "__main__":
    run_batch_inference()
