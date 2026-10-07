"""Airflow DAG: Automated Model Retraining, Champion Evaluation, and Deployment."""
from datetime import datetime, timedelta

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
except ImportError:
    DAG = None
    PythonOperator = None


def task_retrain_models():
    from pipelines.training.run_training import run_training_pipeline
    results = run_training_pipeline()
    return f"Retraining complete. Selected model: {results.get('selected_model')}"


if DAG is not None:
    default_args = {
        "owner": "mlops",
        "depends_on_past": False,
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=10),
    }

    dag = DAG(
        "flixchurn_model_retrain",
        default_args=default_args,
        description="Weekly model retraining, champion benchmarking, and artifact deployment",
        schedule_interval="0 4 * * 0",  # Weekly Sundays at 04:00 UTC
        start_date=datetime(2026, 1, 1),
        catchup=False,
        tags=["flixchurn", "retraining", "mlops"],
    )

    t_train = PythonOperator(
        task_id="execute_retraining_pipeline",
        python_callable=task_retrain_models,
        dag=dag,
    )
