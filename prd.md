# Product Requirements Document

# AI Customer Churn Prediction & Retention Intelligence Platform

**Project Name:** FlixChurn
**Domain:** Subscription Streaming / OTT
**Product Type:** AI/ML Decision-Support Platform
**Primary User:** Business Analyst / Customer Retention Analyst
**Secondary Users:** Product Manager, CRM/Retention Team, Data Scientist, Business Leadership
**Author:** Aquil Shiyon
**Status:** Implemented & Production-Ready
**Version:** 1.2

---

## 1. Executive Summary

ChurnIQ is an AI-powered customer churn prediction and analysis platform designed for a subscription-based streaming business.

The platform analyzes historical customer behavior, subscription activity, engagement, payment behavior, customer-support interactions, and other available signals to:

1. Estimate the probability that an active customer will churn within a defined prediction horizon.
2. Segment customers by churn risk.
3. Explain the major behavioral factors contributing to an individual prediction.
4. Identify patterns and cohorts associated with elevated churn.
5. Simulate how changes in selected customer behaviors could affect predicted churn risk.
6. Provide business-oriented insights for retention decision-making.
7. Monitor model performance and data quality over time.

The system is intentionally **not a CRUD application**.

The primary product value is the complete ML lifecycle:

**Raw Data → Validation → Feature Engineering → Label Generation → Model Training → Evaluation → Explainability → Batch Prediction → API Inference → Business Analytics → Monitoring**

---

# 2. Business Problem

A streaming subscription business loses recurring revenue when customers cancel or fail to renew.

A simple historical churn report answers:

> "How many customers churned?"

It does not adequately answer:

> "Which currently active customers are at elevated risk of churning, why are they at risk, and what behavioral patterns characterize that risk?"

The platform therefore needs a predictive system capable of transforming customer behavior into actionable churn-risk intelligence.

---

# 3. Product Objective

Build an end-to-end production-style machine learning system that predicts customer churn probability and converts model outputs into interpretable business intelligence.

The project must demonstrate practical competence in:

* Data engineering
* Exploratory data analysis
* Feature engineering
* Supervised machine learning
* Model selection
* Imbalanced classification
* Probability calibration
* Explainable AI
* Model evaluation
* API development
* Batch inference
* Data validation
* Model versioning
* Monitoring
* Visualization
* Deployment

---

# 4. Explicit Scope Decisions

The following are **project decisions**, not claims about Netflix's actual systems.

## 4.1 Business Model

The initial implementation will model a subscription-based streaming service.

Customers have an active subscription and generate behavioral signals through platform usage.

## 4.2 Churn Definition

For the initial implementation:

**Churn = a customer who becomes inactive/cancels/non-renews within the defined churn window following the observation period.**

The exact operational definition must be configurable.

The system must NOT hard-code "churn" as merely `subscription_status = cancelled`, because different businesses may define churn differently.

## 4.3 Prediction Horizon

Initial project configuration:

**Predict whether an active customer will churn within the next 30 days.**

This must be represented as a configurable parameter rather than scattered throughout the codebase.

## 4.4 Observation Window

Initial project configuration:

**Use the previous 60 days of customer behavior to generate predictive features.**

This is a project decision and must be configurable.

## 4.5 Prediction Population

Predictions should primarily be generated for customers who are:

* currently active
* eligible for prediction
* sufficiently represented in the observation period
* not already churned
* not excluded by data-quality rules

---

# 5. Target Users

## 5.1 Business Analyst

Needs to:

* understand overall churn risk
* identify high-risk customer cohorts
* investigate behavioral patterns
* compare risk across plans/segments
* understand model explanations
* monitor changes over time

## 5.2 Retention/Product Manager

Needs to:

* identify populations requiring retention attention
* understand major churn drivers
* analyze customer segments
* evaluate hypothetical behavioral scenarios

## 5.3 Data Scientist

Needs to:

* inspect model performance
* compare model versions
* inspect feature importance
* investigate data drift
* evaluate calibration
* identify model degradation

---

# 6. Core Product Capabilities

## 6.1 Churn Risk Prediction

For each eligible customer:

```text
customer_id
prediction_timestamp
model_version
churn_probability
risk_band
prediction_horizon
```

Example risk bands:

* Low
* Moderate
* High
* Critical

Risk bands must be derived from configurable probability thresholds rather than arbitrary UI labels.

---

# 6.2 Risk Distribution

The platform must show:

* total eligible customers
* number of high-risk customers
* percentage of population at high risk
* average predicted churn probability
* predicted churn volume
* risk distribution
* risk trend over time

---

# 6.3 Customer Risk Investigation

An analyst should be able to select a customer and inspect:

### Customer profile

* Customer ID
* Subscription plan
* Tenure
* Region, if available and appropriate
* Subscription status
* Payment characteristics

### Engagement

* sessions
* watch time
* active days
* content completion
* days since last activity
* engagement trend

### Behavioral changes

Examples:

```text
Watch time ↓ 38%
Active days ↓ 25%
Days since last session ↑ 9 days
Completion rate ↓ 12%
```

### Model output

```text
Churn probability: 78.4%
Risk band: High
Prediction horizon: 30 days
Model version: x.y.z
```

### Explanation

Show the strongest factors contributing to the prediction.

---

# 6.4 Explainable AI

The platform must provide local explanations for individual predictions.

Preferred approach:

**SHAP**

The system should expose:

* top positive churn contributors
* top negative churn contributors
* feature value
* contribution direction
* contribution magnitude

Example:

```text
Higher days since last activity      +0.21
Reduced watch time                   +0.14
Lower active days                    +0.09
Longer tenure                        -0.06
```

The UI must distinguish between:

> "Feature contributed to the model prediction"

and:

> "This feature caused the customer to churn."

The model cannot establish causality merely from predictive association.

---

# 6.5 Cohort Analysis

Users must be able to analyze churn risk across dimensions such as:

* subscription plan
* tenure bucket
* engagement bucket
* region where available
* acquisition cohort where available
* payment behavior
* usage intensity
* content engagement category

The system must avoid arbitrary demographic segmentation unless the dataset and business case explicitly justify it.

---

# 6.6 Churn Driver Analysis

The platform should aggregate model explanations to identify recurring predictive patterns.

Examples:

```text
Long inactivity period
Declining engagement
Low active-day frequency
Reduced content completion
Payment instability
Low recent usage
```

The output must be described as:

**predictive drivers**

rather than causal drivers.

---

# 6.7 What-If Risk Simulator

The platform should allow an analyst to modify selected behavioral features and observe how the model prediction changes.

Example:

Current:

```text
Watch time: 4.2 hrs/week
Active days: 2
Days since activity: 11
Predicted churn: 72%
```

Scenario:

```text
Watch time: 7 hrs/week
Active days: 4
Days since activity: 4
Predicted churn: 48%
```

The interface must clearly state:

> "This is a model simulation, not evidence that changing the behavior will cause churn probability to decrease."

Only features that are valid, mutable, and within realistic ranges may be exposed.

---

# 6.8 Model Evaluation

The platform must expose:

* ROC-AUC
* PR-AUC
* precision
* recall
* F1
* confusion matrix
* calibration curve
* Brier score
* lift
* precision@top-k
* recall@top-k

Because churn is commonly imbalanced, **accuracy must not be treated as the primary model metric**.

---

# 6.9 Model Comparison

The system should support comparison between candidate models.

Initial candidate models:

1. Logistic Regression
2. Random Forest
3. XGBoost / LightGBM equivalent gradient boosting model

The final model must be selected using predefined evaluation criteria.

The system must not automatically select a model merely because it has the highest ROC-AUC.

---

# 6.10 Data Quality Monitoring

Before model inference/training, validate:

* missing values
* duplicate records
* invalid timestamps
* impossible numerical values
* categorical inconsistencies
* target leakage
* feature availability
* unexpected distributions

Data-quality failures should be visible and logged.

---

# 6.11 Model Monitoring

The system should monitor:

* prediction distribution
* feature drift
* data drift
* prediction drift
* model performance when labels become available
* calibration degradation

Model monitoring should distinguish:

**Data drift**

from

**Concept/performance drift**

---

# 6.12 Financial Value-at-Risk ($VaR) & Churn Mechanism Segmentation

The platform must bridge statistical risk models with financial metrics:

1. **Revenue Exposure ($Value-at-Risk)**:
   - Calculate Monthly Recurring Revenue at Risk: `$MRR at Risk = monthly_fee * P(churn)`
   - Calculate Annual Recurring Revenue at Risk: `$ARR at Risk = monthly_fee * 12 * P(churn)`
   - Aggregate financial loss totals across the subscriber base, plan tiers, and regional cohorts.

2. **Churn Mechanism Decomposition**:
   - **Voluntary Churn**: Subscribers at risk due to declining usage or content dissatisfaction (`watch_hours < 15`, `login_frequency < 5`).
   - **Involuntary Churn**: Subscribers at risk due to payment friction or billing failures (`payment_failures_last_90d >= 1`, `failed_transaction_count >= 1`).
   - Enable targeted retention workflows tailored to each specific churn vector.

---

# 7. Functional Requirements

## FR-01 Data Ingestion

The system must ingest structured customer activity data.

Supported initial format:

```text
CSV / Parquet
```

The ingestion layer must not depend on a single hard-coded dataset.

---

## FR-02 Data Validation

Every dataset must pass schema and quality validation before entering the ML pipeline.

---

## FR-03 Feature Generation

Features must be generated from timestamped behavioral data.

Examples:

```text
watch_time_7d
watch_time_30d
active_days_7d
active_days_30d
sessions_7d
sessions_30d
days_since_last_activity
completion_rate_30d
engagement_trend
payment_failure_count
tenure_days
```

Features must be generated using historical information available **before the prediction timestamp**.

---

## FR-04 Label Generation

The system must generate training labels using future information relative to the observation timestamp.

This separation must be explicit.

Example:

```text
Observation Window
       ↓
Feature Generation
       ↓
Prediction Timestamp
       ↓
30-Day Outcome Window
       ↓
Churn Label
```

---

## FR-05 Train/Test Splitting

For timestamped customer behavior, the preferred evaluation strategy is temporal splitting.

Example:

```text
Past ----------------------------> Future

Training          Validation      Test
```

Random splitting should not be used as the default for temporal churn prediction.

---

## FR-06 Model Training

Training must be reproducible.

Each training run should record:

* dataset version
* feature version
* model parameters
* training timestamp
* evaluation metrics
* model artifact
* code/version identifier

---

## FR-07 Prediction API

Expose an inference API.

Example endpoint concept:

```http
POST /predict
```

Input:

```json
{
  "customer_features": {}
}
```

Output:

```json
{
  "churn_probability": 0.784,
  "risk_band": "high",
  "model_version": "1.2.0",
  "prediction_horizon_days": 30
}
```

---

## FR-08 Batch Prediction

Support batch prediction for the current customer population.

The batch process must generate versioned prediction outputs.

---

## FR-09 Explanation API

The backend must provide explanation information for a prediction.

---

## FR-10 Analytics

The application must aggregate prediction data for business analysis.

---

## FR-11 Monitoring

System and model metrics must be persisted and exposed through the monitoring interface.

---

# 8. Non-Functional Requirements

## NFR-01 Reproducibility

A model should be reproducible from:

```text
dataset version
+
feature version
+
configuration
+
code version
+
model parameters
```

---

## NFR-02 Explainability

Predictions must be explainable at both:

* global level
* individual customer level

---

## NFR-03 Reliability

Invalid data must fail safely rather than silently producing predictions.

---

## NFR-04 Observability

The system must provide structured logging for:

* ingestion
* validation
* training
* inference
* API requests
* model failures

---

## NFR-05 Security

Customer identifiers must not be unnecessarily exposed.

Sensitive information must not appear in application logs.

---

## NFR-06 Performance

The API should be capable of low-latency single-customer inference for normal analytical usage.

Exact latency targets must be benchmarked rather than fabricated.

---

# 9. Business Metrics

The platform should allow the business to evaluate:

```text
Customers at high risk
Predicted churn volume
Predicted churn rate
Risk concentration
Churn risk by cohort
Churn risk by plan
Top predictive drivers
Retention opportunity population
```

If customer revenue/CLV data becomes available, the system may additionally calculate:

```text
Expected revenue at risk
```

This must not be fabricated when revenue data is unavailable.

---

# 10. Success Criteria

The project is considered technically successful when:

1. The entire ML pipeline can run from raw dataset to predictions.
2. No future information leaks into model features.
3. Model performance is evaluated using appropriate classification and calibration metrics.
4. Predictions are probabilistic rather than only binary.
5. Individual predictions have interpretable explanations.
6. Batch and API inference produce consistent results.
7. Model versions are traceable.
8. Data quality failures are detectable.
9. The UI communicates predictive uncertainty appropriately.
10. The application demonstrates a complete AI product rather than a model notebook.

---

# 11. Explicit Non-Goals

The first version will NOT include:

* generic customer CRUD
* customer profile editing
* admin CRUD screens
* arbitrary database management
* generic authentication-heavy architecture
* payment processing
* recommendation engine
* content recommendation
* automatic customer messaging
* automatic discounts
* autonomous retention campaigns
* causal inference claims
* fake real-time streaming behavior

---

# 12. Assumptions and Unknowns

The following are intentionally explicit.

### Assumption A1

A suitable historical customer/activity dataset is available or will be created using a documented schema.

### Assumption A2

The dataset contains timestamped behavioral information.

### Assumption A3

A churn outcome can be defined consistently.

### Unknown U1

Actual Netflix internal data is unavailable and must not be assumed.

### Unknown U2

Actual Netflix churn definitions, business thresholds, feature distributions, infrastructure, and model performance are unknown.

### Unknown U3

The actual retention economics are unknown.

Therefore, the project must use a documented public dataset or a clearly documented synthetic dataset.

---

# 13. Initial Dataset Contract

The implementation should support a normalized event-oriented structure such as:

```text
customer_id
event_timestamp
event_type
watch_time_minutes
session_duration
content_completion_rate
subscription_plan
subscription_status
payment_status
```

Additional fields may be introduced only when justified by the available dataset.

The ML feature layer must abstract the raw schema so that another compatible dataset can be substituted without rewriting the model layer.

---

# 14. Product Principle

The system is not intended to answer:

> "Who will definitely churn?"

It should answer:

> "Based on the available historical evidence, which currently active customers have elevated predicted churn risk, what signals are associated with that prediction, and how reliable is the model?"

That distinction is central to the product.
