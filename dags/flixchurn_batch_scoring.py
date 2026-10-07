"""Airflow DAG: Daily FlixChurn Batch Scoring and Risk Categorization."""
from datetime import datetime, timedelta

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
except ImportError:
    # Fallback placeholder if airflow package is not in the active local test environment
    DAG = None
    PythonOperator = None


def task_validate_input():
    from ml.ingestion import load_raw_data
    from ml.validation import validate_raw_data

    df_raw = load_raw_data()
    report = validate_raw_data(df_raw)
    if not report.passed:
        raise ValueError(f"Data validation failed: {report.summary()}")
    return f"Validated {len(df_raw)} records"


def task_run_batch_inference():
    from pipelines.inference.run_inference import run_batch_inference
    df_results = run_batch_inference()
    return f"Scored {len(df_results)} customers"


def task_monitor_drift():
    from ml.config import PROJECT_ROOT
    from ml.monitoring import compute_feature_drift
    import pandas as pd

    proc_dir = PROJECT_ROOT / "data" / "processed"
    pred_path = proc_dir / "latest_predictions.parquet"
    train_path = proc_dir / "training_features.parquet"

    if pred_path.exists() and train_path.exists():
        df_pred = pd.read_parquet(pred_path)
        df_train = pd.read_parquet(train_path)
        drift = compute_feature_drift(df_train, df_pred)
        return f"Drift evaluated: {len(drift.get('drifted_features', []))} features drifted"
    return "Skipped drift check: missing baseline data"


if DAG is not None:
    default_args = {
        "owner": "mlops",
        "depends_on_past": False,
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    }

    dag = DAG(
        "flixchurn_batch_scoring",
        default_args=default_args,
        description="Daily automated batch scoring and churn risk categorization",
        schedule_interval="0 2 * * *",  # Daily at 02:00 UTC
        start_date=datetime(2026, 1, 1),
        catchup=False,
        tags=["flixchurn", "batch-inference", "mlops"],
    )

    t1 = PythonOperator(
        task_id="validate_raw_data",
        python_callable=task_validate_input,
        dag=dag,
    )

    t2 = PythonOperator(
        task_id="run_batch_scoring",
        python_callable=task_run_batch_inference,
        dag=dag,
    )

    t3 = PythonOperator(
        task_id="monitor_feature_drift",
        python_callable=task_monitor_drift,
        dag=dag,
    )

    t1 >> t2 >> t3
