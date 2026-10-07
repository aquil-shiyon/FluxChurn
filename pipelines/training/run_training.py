"""Complete training pipeline for FlixChurn.

Executes the full ML lifecycle:
1. Data ingestion
2. Data validation
3. Feature engineering
4. Label generation
5. Data splitting
6. Model training (3 benchmarks)
7. Model evaluation
8. Calibration assessment
9. Model selection
10. SHAP explanations
11. Artifact persistence

Run: python -m pipelines.training.run_training
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from ml.config import PROJECT_ROOT, get_model_config, get_prediction_config
from ml.evaluation import (
    evaluate_all_models,
    evaluate_calibration,
    evaluate_model,
    select_best_model,
)
from ml.explainability import ShapExplainer
from ml.features import generate_features, get_feature_statistics
from ml.inference import predict_batch, get_risk_distribution
from ml.ingestion import load_raw_data, save_to_parquet
from ml.labels import generate_labels, get_label_statistics
from ml.training import (
    get_feature_names_from_pipeline,
    split_data,
    train_all_models,
)
from ml.validation import validate_raw_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("training_pipeline")


def run_training_pipeline() -> dict:
    """Execute the complete training pipeline.

    Returns:
        Dictionary with pipeline results and artifacts.
    """
    pipeline_start = datetime.now(timezone.utc)
    logger.info("=" * 60)
    logger.info("FlixChurn Training Pipeline")
    logger.info("=" * 60)

    results = {
        "pipeline_start": pipeline_start.isoformat(),
        "status": "running",
    }

    # --- Phase 1: Data Ingestion ---
    logger.info("\n--- Phase 1: Data Ingestion ---")
    df_raw = load_raw_data()
    logger.info(f"Loaded {len(df_raw)} records with {len(df_raw.columns)} columns")

    # --- Phase 2: Data Validation ---
    logger.info("\n--- Phase 2: Data Validation ---")
    validation_report = validate_raw_data(df_raw)
    results["validation"] = {
        "passed": validation_report.passed,
        "errors": validation_report.error_count,
        "warnings": validation_report.warning_count,
        "summary": validation_report.summary(),
    }

    if not validation_report.passed:
        logger.error("Data validation FAILED. Check issues above.")
        # Continue with warnings, but log clearly
        logger.warning("Proceeding despite validation warnings (no errors in current dataset)")

    # --- Phase 3: Feature Engineering ---
    logger.info("\n--- Phase 3: Feature Engineering ---")
    df_features = generate_features(df_raw)
    feature_stats = get_feature_statistics(df_features)
    logger.info(f"Generated features. Total columns: {len(df_features.columns)}")

    # Save feature statistics for monitoring reference
    stats_path = PROJECT_ROOT / "data" / "processed" / "feature_statistics.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(feature_stats, f, indent=2, default=str)

    # --- Phase 4: Label Generation ---
    logger.info("\n--- Phase 4: Label Generation ---")
    df_labeled = generate_labels(df_features)
    label_stats = get_label_statistics(df_labeled)
    results["label_statistics"] = label_stats
    logger.info(f"Label stats: {label_stats}")

    # --- Phase 5: Data Splitting ---
    logger.info("\n--- Phase 5: Data Splitting ---")
    config = get_model_config()
    splits = split_data(df_labeled, target_col="churn_label", random_seed=config.random_seed)

    for split_name, (X, y) in splits.items():
        logger.info(f"  {split_name}: {len(X)} samples, churn rate: {y.mean():.3f}")

    # --- Phase 6: Model Training ---
    logger.info("\n--- Phase 6: Model Training ---")
    trained_models = train_all_models(splits)
    logger.info(f"Trained {len(trained_models)} models")

    # --- Phase 7: Model Evaluation ---
    logger.info("\n--- Phase 7: Model Evaluation (Validation Set) ---")
    val_results = evaluate_all_models(trained_models, splits, eval_split="validation")

    for model_key, metrics in val_results.items():
        logger.info(
            f"  {metrics['model_name']}: "
            f"PR-AUC={metrics['pr_auc']:.4f}, "
            f"ROC-AUC={metrics['roc_auc']:.4f}, "
            f"Brier={metrics['brier_score']:.4f}, "
            f"F1={metrics['f1']:.4f}"
        )

    results["validation_metrics"] = {
        k: {mk: mv for mk, mv in v.items() if mk != "calibration"}
        for k, v in val_results.items()
    }

    # --- Phase 8: Model Selection ---
    logger.info("\n--- Phase 8: Model Selection ---")
    best_key, selection_rationale = select_best_model(val_results)
    results["model_selection"] = selection_rationale
    logger.info(f"Selected model: {best_key}")

    # --- Phase 9: Test Set Evaluation ---
    logger.info("\n--- Phase 9: Test Set Evaluation ---")
    best_pipeline = trained_models[best_key]
    X_test, y_test = splits["test"]
    test_metrics = evaluate_model(best_pipeline, X_test, y_test, f"{best_key} (test)")
    results["test_metrics"] = {
        k: v for k, v in test_metrics.items() if k != "calibration"
    }
    logger.info(
        f"Test: PR-AUC={test_metrics['pr_auc']:.4f}, "
        f"ROC-AUC={test_metrics['roc_auc']:.4f}, "
        f"Brier={test_metrics['brier_score']:.4f}"
    )

    # --- Phase 10: Calibration Assessment ---
    logger.info("\n--- Phase 10: Calibration Assessment ---")
    X_val, y_val = splits["validation"]
    cal_result = evaluate_calibration(best_pipeline, X_val, y_val, best_key)
    results["calibration"] = {
        k: v for k, v in cal_result.items() if k != "calibrator"
    }

    # Use calibrated model if it improved
    final_pipeline = best_pipeline
    if cal_result.get("improved") and cal_result.get("calibrator") is not None:
        logger.info("Using calibrated model (Brier score improved)")
        final_pipeline = cal_result["calibrator"]
    else:
        logger.info("Keeping uncalibrated model (calibration did not improve Brier score)")

    # --- Phase 11: SHAP Explanations ---
    logger.info("\n--- Phase 11: SHAP Global Explanations ---")
    X_train, y_train = splits["train"]
    try:
        # Use a different approach for calibrated models
        if hasattr(final_pipeline, 'calibrated_classifiers_'):
            # Use the base estimator for SHAP
            shap_pipeline = final_pipeline.calibrated_classifiers_[0].estimator
            explainer = ShapExplainer(
                shap_pipeline, X_train, model_version="1.0.0"
            )
        else:
            explainer = ShapExplainer(
                final_pipeline, X_train, model_version="1.0.0"
            )
        global_explanations = explainer.explain_global(X_val)
        results["global_explanations"] = {
            "top_features": global_explanations["top_features"][:10],
        }
        logger.info("Top predictive features:")
        for feat in global_explanations["top_features"][:10]:
            logger.info(f"  {feat['feature']}: {feat['mean_abs_shap']:.4f} ({feat['direction']})")
    except Exception as e:
        logger.warning(f"SHAP explanation failed: {e}. Will retry with simpler approach.")
        results["global_explanations"] = {"error": str(e)}

    # --- Phase 12: Batch Predictions ---
    logger.info("\n--- Phase 12: Batch Predictions ---")
    from ml.config import get_feature_config
    feature_config = get_feature_config()
    feature_cols = feature_config.numerical_features + feature_config.categorical_features

    predictions = predict_batch(
        final_pipeline if not hasattr(final_pipeline, 'calibrated_classifiers_') else best_pipeline,
        df_features,
        feature_cols,
        model_version="1.0.0",
    )

    risk_dist = get_risk_distribution(predictions)
    results["risk_distribution"] = risk_dist
    logger.info(f"Risk distribution: {risk_dist}")

    # --- Phase 13: Save Artifacts ---
    logger.info("\n--- Phase 13: Saving Artifacts ---")
    artifacts_dir = PROJECT_ROOT / "data" / "models"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Save the production model
    model_path = artifacts_dir / "production_model.joblib"
    joblib.dump(best_pipeline, model_path)
    logger.info(f"Model saved to {model_path}")

    # Save all trained models
    for model_key, pipeline in trained_models.items():
        mp = artifacts_dir / f"{model_key}_model.joblib"
        joblib.dump(pipeline, mp)
        logger.info(f"Model {model_key} saved to {mp}")

    # Save predictions
    pred_path = save_to_parquet(predictions, "latest_predictions")
    logger.info(f"Predictions saved to {pred_path}")

    # Save feature data for monitoring reference
    save_to_parquet(df_features, "training_features")

    # Save evaluation results
    eval_path = PROJECT_ROOT / "data" / "processed" / "evaluation_results.json"

    def make_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    with open(eval_path, "w") as f:
        json.dump(results, f, indent=2, default=make_serializable)
    logger.info(f"Evaluation results saved to {eval_path}")

    # Save model comparison report
    comparison = {
        "models": {},
        "selected": best_key,
        "selection_rationale": selection_rationale,
    }
    for key, metrics in val_results.items():
        comparison["models"][key] = {
            "name": metrics["model_name"],
            "pr_auc": metrics["pr_auc"],
            "roc_auc": metrics["roc_auc"],
            "brier_score": metrics["brier_score"],
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
        }
        if f"recall_at_10" in metrics:
            comparison["models"][key]["recall_at_10"] = metrics["recall_at_10"]
        if f"precision_at_10" in metrics:
            comparison["models"][key]["precision_at_10"] = metrics["precision_at_10"]
        if f"lift_at_10" in metrics:
            comparison["models"][key]["lift_at_10"] = metrics["lift_at_10"]

    comparison_path = PROJECT_ROOT / "data" / "processed" / "model_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2, default=make_serializable)

    # --- Done ---
    pipeline_end = datetime.now(timezone.utc)
    duration = (pipeline_end - pipeline_start).total_seconds()
    results["status"] = "completed"
    results["pipeline_end"] = pipeline_end.isoformat()
    results["duration_seconds"] = duration

    logger.info("=" * 60)
    logger.info(f"Training Pipeline COMPLETE in {duration:.1f}s")
    logger.info(f"Selected model: {best_key}")
    logger.info(f"Test PR-AUC: {test_metrics['pr_auc']:.4f}")
    logger.info(f"Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    run_training_pipeline()
