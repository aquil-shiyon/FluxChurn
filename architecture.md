# System Architecture

# AI Customer Churn Prediction & Retention Intelligence Platform

**Project:** FlixChurn
**Architecture Style:** ML-centric modular architecture with batch + online inference
**Deployment Model:** Containerized
**Primary Language:** Python
**Frontend:** React + TypeScript
**Backend:** FastAPI
**ML:** scikit-learn + gradient boosting + SHAP
**Experiment Tracking:** MLflow
**Orchestration:** Apache Airflow
**Data:** Parquet/DuckDB for analytical processing + PostgreSQL for application/prediction data

---

# 1. Architecture Objective

The architecture must demonstrate an end-to-end production-oriented ML system rather than a notebook connected directly to a frontend.

The major separation is:

```text
DATA
 ↓
DATA VALIDATION
 ↓
FEATURE ENGINEERING
 ↓
LABEL GENERATION
 ↓
MODEL TRAINING
 ↓
MODEL EVALUATION
 ↓
MODEL REGISTRY
 ↓
BATCH / ONLINE INFERENCE
 ↓
ANALYTICS API
 ↓
REACT APPLICATION
```

Monitoring runs across the pipeline.

---

# 2. High-Level Architecture

```text
                     ┌─────────────────────┐
                     │ Raw Customer Data   │
                     │ CSV / Parquet       │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │ Data Validation     │
                     │ Schema + Quality    │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │ Feature Engineering │
                     │ Time-aware features │
                     └──────────┬──────────┘
                                │
                    ┌───────────┴────────────┐
                    ▼                        ▼
          ┌──────────────────┐      ┌──────────────────┐
          │ Label Generation │      │ Feature Dataset  │
          └────────┬─────────┘      └────────┬─────────┘
                   └────────────┬────────────┘
                                ▼
                     ┌─────────────────────┐
                     │ Temporal Split      │
                     └──────────┬──────────┘
                                ▼
                     ┌─────────────────────┐
                     │ Model Training      │
                     │ LR / RF / Boosting  │
                     └──────────┬──────────┘
                                ▼
                     ┌─────────────────────┐
                     │ Model Evaluation    │
                     │ PR-AUC / Recall@K   │
                     │ Calibration / Lift  │
                     └──────────┬──────────┘
                                ▼
                     ┌─────────────────────┐
                     │ MLflow Registry     │
                     └──────────┬──────────┘
                                │
                 ┌──────────────┴──────────────┐
                 ▼                             ▼
       ┌──────────────────┐          ┌──────────────────┐
       │ Batch Prediction │          │ Online Inference │
       │ Airflow Pipeline │          │ FastAPI          │
       └────────┬─────────┘          └────────┬─────────┘
                │                             │
                └─────────────┬───────────────┘
                              ▼
                    ┌─────────────────────┐
                    │ PostgreSQL          │
                    │ Predictions         │
                    │ Metrics             │
                    │ Metadata            │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ FastAPI Analytics   │
                    │ + Explanation APIs  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ React + TypeScript  │
                    │ Analyst Workbench   │
                    └─────────────────────┘
```

---

# 3. Repository Architecture

```text
FlixChurn/
│
├── apps/
│   ├── api/
│   │   ├── routes/
│   │   ├── schemas/
│   │   ├── services/
│   │   ├── dependencies/
│   │   └── main.py
│   │
│   └── web/
│       ├── src/
│       └── package.json
│
├── ml/
│   ├── ingestion/
│   ├── validation/
│   ├── features/
│   ├── labels/
│   ├── training/
│   ├── evaluation/
│   ├── inference/
│   ├── explainability/
│   └── monitoring/
│
├── pipelines/
│   ├── training/
│   └── inference/
│
├── dags/
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── sample/
│
├── configs/
│   ├── model.yaml
│   ├── features.yaml
│   ├── prediction.yaml
│   └── application.yaml
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── ml/
│   └── api/
│
├── notebooks/
│   └── exploratory/
│
├── scripts/
│
├── docker/
│
├── docker-compose.yml
├── pyproject.toml
├── README.md
└── .env.example
```

---

# 4. Data Layer

## 4.1 Raw Data

Raw files must remain immutable.

Example:

```text
data/raw/
```

No preprocessing should overwrite raw data.

---

# 4.2 Analytical Layer

Use Parquet for feature datasets.

Use DuckDB for local analytical querying.

Advantages:

* columnar storage
* efficient local analytics
* reproducibility
* no dependency on a cloud warehouse
* realistic data-engineering workflow

---

# 4.3 Application Data

PostgreSQL stores:

```text
prediction_runs
customer_predictions
model_metadata
model_metrics
feature_metadata
data_quality_results
monitoring_metrics
```

The database is used for analytical application state, not as a generic CRUD demonstration.

---

# 5. Data Pipeline

```text
Raw Dataset
    ↓
Schema Validation
    ↓
Quality Validation
    ↓
Cleaning
    ↓
Event Normalization
    ↓
Feature Aggregation
    ↓
Label Generation
    ↓
Feature Dataset
```

---

# 6. Feature Engineering

Feature engineering must be time-aware.

For prediction timestamp `T`, a feature may only use information where:

```text
event_timestamp < T
```

Future events must never contribute to the feature vector.

---

## 6.1 Feature Categories

### Recency

```text
days_since_last_activity
days_since_last_payment
days_since_last_session
```

### Frequency

```text
sessions_7d
sessions_30d
active_days_7d
active_days_30d
```

### Intensity

```text
watch_minutes_7d
watch_minutes_30d
average_session_duration
```

### Engagement

```text
completion_rate
content_interaction_count
engagement_frequency
```

### Trend

```text
watch_time_7d_vs_previous_7d
active_days_7d_vs_previous_7d
session_change_rate
```

### Subscription

```text
tenure_days
plan_type
plan_change_count
```

### Payment

```text
payment_failure_count
payment_retry_count
```

Only features supported by the actual dataset may be implemented.

---

# 7. Label Generation

For each observation timestamp:

```text
[Historical Observation Window]
              |
              V
       Prediction Point
              |
              V
       [Outcome Window]
```

Example:

```text
60 days historical behavior
            ↓
      prediction date
            ↓
       30 day outcome
```

Label:

```text
1 = churned within outcome window
0 = did not churn within outcome window
```

Customers with insufficient outcome observation must be excluded from evaluation rather than incorrectly labeled as non-churners.

---

# 8. Temporal Data Splitting

Recommended:

```text
                    TIME
────────────────────────────────────────────>

TRAIN                VALIDATION          TEST
████████████████      ████████            ███████
```

No random shuffling across time for the primary evaluation.

This reduces temporal leakage.

---

# 9. Model Architecture

The initial model benchmark should include:

## Baseline

Logistic Regression

Purpose:

* interpretable baseline
* probability benchmark
* sanity check

## Nonlinear Model

Random Forest

Purpose:

* nonlinear interactions
* benchmark against linear model

## Primary Candidate

Gradient Boosting:

```text
XGBoost / LightGBM
```

Use one based on environment compatibility.

The implementation should avoid unnecessarily supporting multiple gradient boosting frameworks.

---

# 10. Model Selection

Model selection must consider:

```text
PR-AUC
Recall@Top-K
Precision@Top-K
Calibration
Brier Score
Lift
Operational usefulness
```

ROC-AUC is useful but cannot be the only selection criterion.

---

# 11. Probability Calibration

Raw classifier scores must not automatically be treated as reliable probabilities.

Evaluate calibration.

Potential methods:

```text
Platt scaling
Isotonic regression
```

Only use calibration if validation demonstrates that it improves probability reliability without unacceptable degradation.

---

# 12. Explainability Architecture

Use SHAP for supported models.

Two modes:

### Global

Identify features that consistently contribute to model predictions.

### Local

Explain a specific customer prediction.

```text
Customer Features
       ↓
Model
       ↓
Prediction
       ↓
SHAP Explainer
       ↓
Feature Contributions
```

---

# 13. Inference Architecture

## 13.1 Batch

Used for population-level scoring.

```text
Airflow
   ↓
Load eligible customers
   ↓
Generate current features
   ↓
Load registered model
   ↓
Generate probabilities
   ↓
Generate explanations where required
   ↓
Persist predictions
```

## 13.2 Online

Used for interactive scenario prediction.

```text
React
  ↓
FastAPI
  ↓
Feature validation
  ↓
Model service
  ↓
Prediction
  ↓
Explanation
  ↓
Response
```

---

# 14. API Structure

Suggested endpoints:

```text
GET  /health

GET  /api/v1/overview
GET  /api/v1/risk-distribution
GET  /api/v1/cohorts
GET  /api/v1/drivers

GET  /api/v1/customers/{customer_id}/risk
GET  /api/v1/customers/{customer_id}/explanation

POST /api/v1/predict
POST /api/v1/simulate

GET  /api/v1/models
GET  /api/v1/models/{model_version}/metrics

GET  /api/v1/monitoring/data-quality
GET  /api/v1/monitoring/drift
GET  /api/v1/monitoring/performance
```

The endpoints are analytical/query/inference interfaces rather than CRUD resources.

---

# 15. API Contracts

Pydantic models must validate all API requests and responses.

Example prediction response:

```json
{
  "customer_id": "C10291",
  "churn_probability": 0.784,
  "risk_band": "high",
  "prediction_horizon_days": 30,
  "model_version": "1.2.0",
  "prediction_timestamp": "2026-10-01T10:30:00Z"
}
```

---

# 16. Monitoring Architecture

Monitoring is divided into three layers.

## Data Monitoring

```text
missingness
schema changes
range violations
category changes
distribution drift
```

## Prediction Monitoring

```text
probability distribution
risk-band distribution
prediction volume
```

## Model Performance Monitoring

Once actual outcomes become available:

```text
PR-AUC
ROC-AUC
Recall
Precision
Brier Score
Calibration
Lift
```

---

# 17. Drift Detection

Potential techniques:

* PSI
* KS statistic
* distribution comparison
* categorical frequency divergence

The system should record the method and threshold used.

Do not label a feature as "drifted" without an explicit statistical rule.

---

# 18. MLflow

MLflow should track:

```text
experiment
run
parameters
metrics
artifacts
model
dataset metadata
feature version
```

The registered model should have explicit version identifiers.

---

# 19. Airflow

Airflow should orchestrate:

### Training DAG

```text
validate_data
    ↓
prepare_features
    ↓
generate_labels
    ↓
split_data
    ↓
train_models
    ↓
evaluate_models
    ↓
register_model
```

### Prediction DAG

```text
validate_current_data
    ↓
generate_current_features
    ↓
load_production_model
    ↓
batch_predict
    ↓
persist_predictions
    ↓
calculate_monitoring_metrics
```

---

# 20. Frontend Architecture

React + TypeScript.

Suggested structure:

```text
src/
├── app/
├── pages/
├── components/
├── features/
│   ├── overview/
│   ├── customers/
│   ├── cohorts/
│   ├── simulator/
│   ├── models/
│   └── monitoring/
├── services/
├── hooks/
├── types/
└── utils/
```

The frontend should never contain ML logic.

---

# 21. Frontend Data Flow

```text
React UI
   ↓
API Client
   ↓
FastAPI
   ↓
Service Layer
   ↓
PostgreSQL / Model Service
```

---

# 22. Deployment

Use Docker Compose for local deployment.

Services:

```text
frontend
backend
postgres
mlflow
airflow
airflow-db
```

DuckDB/Parquet remain part of the analytical pipeline.

---

# 23. Environment Separation

Configuration must be externalized.

Example:

```text
.env
.env.example
configs/
```

Never hard-code:

* database credentials
* secret keys
* model paths
* environment-specific URLs
* API credentials

---

# 24. Testing Strategy

## Unit Tests

Test:

* feature calculations
* label generation
* validation rules
* threshold logic
* risk-band mapping
* API schemas

## ML Tests

Test:

* no future timestamps used
* train/test temporal separation
* feature consistency
* deterministic preprocessing
* model artifact loading

## Integration Tests

Test:

```text
API → service → database
API → model inference
pipeline → prediction storage
```

---

# 25. Critical Architectural Principle

The architecture must preserve this separation:

```text
Business Definition
        ↓
Data
        ↓
Feature Engineering
        ↓
Model
        ↓
Prediction
        ↓
Explanation (SHAP)
        ↓
Financial Risk Engine ($VaR / MRR at Risk)
        ↓
Business Action / Retention Simulation
```

The system must never reverse this process by inventing business explanations from UI requirements.

---

# 26. Financial Risk Engine & Churn Mechanism Layer

To translate statistical probabilities into executive business metrics, the system implements a Financial Exposure & Churn Mechanism layer:

1. **Revenue Exposure ($Value-at-Risk)**:
   - `$MRR at Risk = monthly_fee * P(churn)`
   - `$ARR at Risk = monthly_fee * 12 * P(churn)`
   - Allows aggregation of financial loss across individual subscribers, subscription tiers, and geographic cohorts.

2. **Churn Mechanism Decomposition**:
   - **Voluntary Churn Risk**: High churn probability accompanied by decaying platform engagement (low watch hours, low login frequency, low content diversity). Remediation: Personalized content push, re-engagement campaigns.
   - **Involuntary Churn Risk**: High churn probability accompanied by payment friction (recent payment failures, failed transaction count). Remediation: Automated card update notifications, smart dunning grace periods.

