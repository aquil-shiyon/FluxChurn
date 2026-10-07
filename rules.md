# Engineering Rules

# ChurnIQ

These rules are mandatory for implementation.

---

# 1. Product Rules

## R-001 — This is an ML product

The application must demonstrate a genuine machine-learning lifecycle.

A dashboard over static CSV data is insufficient.

---

## R-002 — No CRUD architecture

Do not build:

```text
Create Customer
Edit Customer
Delete Customer
Customer Management
Admin CRUD
```

The customer entity exists primarily as an analytical subject.

---

## R-003 — No fake AI

Do not generate:

* hard-coded predictions
* random risk scores
* manually written "AI insights"
* fake SHAP values
* fabricated model metrics
* fake drift statistics
* hard-coded dashboard numbers presented as live results

---

# 2. Data Rules

## R-004 — Never assume the dataset schema

Inspect the actual dataset before implementing feature extraction.

If the dataset does not contain a required feature:

* do not fabricate it
* identify an appropriate derivation
* or exclude the feature

---

## R-005 — Preserve raw data

Raw data must never be overwritten.

---

## R-006 — Data lineage

Every generated feature dataset should be traceable to its source data.

---

## R-007 — Timestamp integrity

All time-based features must use only information available before the prediction timestamp.

---

# 3. Leakage Rules

## R-008 — No target leakage

The following are prohibited as features when they encode future outcomes:

```text
cancellation_date
post-churn status
future payment events
future engagement
future support outcomes
```

unless they legitimately existed before prediction.

---

## R-009 — Temporal validation

Primary evaluation must use time-aware splitting.

---

## R-010 — No preprocessing leakage

Transformers must be fitted only on training data.

Example:

```text
Scaler.fit(train)
Scaler.transform(train)
Scaler.transform(validation)
Scaler.transform(test)
```

Never:

```text
Scaler.fit(all_data)
```

---

# 4. ML Rules

## R-011 — Baseline first

A simple baseline must exist before complex models.

---

## R-012 — Accuracy is not the primary metric

For imbalanced churn classification, report:

```text
PR-AUC
Recall
Precision
F1
Recall@K
Precision@K
Calibration
Brier Score
Lift
```

---

## R-013 — Probability means probability

If the application displays:

```text
78% churn probability
```

the number must come from a calibrated/evaluated probabilistic model.

Do not call arbitrary classifier scores "probability."

---

## R-014 — Thresholds must be configurable

Risk thresholds must live in configuration.

Do not scatter:

```python
if probability > 0.7:
```

throughout the codebase.

---

## R-015 — Model selection must be documented

The final model must have an auditable reason for selection.

---

# 5. Explainability Rules

## R-016 — SHAP is not causality

The UI must never say:

> "This feature caused the customer to churn."

Instead:

> "This feature contributed to the model's prediction."

---

## R-017 — Explanation consistency

The explanation shown for a customer must correspond to the same model version used for the prediction.

---

## R-018 — What-if simulation disclaimer

Scenario simulation must be explicitly described as model-based simulation, not causal prediction.

---

# 6. Backend Rules

## R-019 — API-first architecture

Frontend components must communicate through API/service interfaces.

Do not directly query PostgreSQL from React.

---

## R-020 — Strong schemas

Use Pydantic models for API input/output.

---

## R-021 — Service separation

Keep:

```text
route
service
repository/data access
ML service
```

separated.

---

## R-022 — No business logic inside routes

FastAPI route handlers should orchestrate services rather than contain large ML/data-processing functions.

---

# 7. Frontend Rules

## R-023 — No generic admin dashboard

Do not use generic layouts containing:

```text
Sidebar
Dashboard
Users
Settings
Reports
```

unless each section has an analytical purpose.

---

## R-024 — Information hierarchy

The interface must prioritize:

```text
Risk
Why risk exists
Where risk is concentrated
What changed
How reliable the model is
```

---

## R-025 — Dense but readable

This is an analyst tool, not a consumer entertainment interface.

Use high information density without creating visual clutter.

---

## R-026 — Avoid dashboard decoration

Do not add charts merely to make the UI look sophisticated.

Every visualization must answer a business question.

---

# 8. UX Rules

## R-027 — Analytical navigation

Primary navigation should reflect workflows:

```text
Risk Overview
Customer Intelligence
Cohorts
Simulator
Model Lab
Monitoring
```

---

## R-028 — Customer investigation

The customer view should feel like an analytical investigation workspace, not a profile page.

---

## R-029 — Model transparency

Model version, prediction timestamp and prediction horizon should be discoverable.

---

# 9. Code Quality Rules

## R-030 — Type safety

Use type hints throughout Python.

Use TypeScript types throughout the frontend.

---

## R-031 — Configuration

Do not hard-code environment-specific configuration.

---

## R-032 — Logging

Use structured logs.

Never log secrets or unnecessary customer-sensitive data.

---

## R-033 — Error handling

Errors must be explicit and actionable.

Do not silently return empty data when a pipeline or model fails.

---

# 10. Testing Rules

## R-034

Critical feature engineering functions require unit tests.

## R-035

Label generation requires tests covering boundary dates.

## R-036

Inference requires schema validation tests.

## R-037

The API requires integration tests.

## R-038

Temporal leakage tests must exist.

---

# 11. Documentation Rules

The repository must explain:

* business problem
* dataset
* churn definition
* feature definitions
* model selection
* evaluation
* architecture
* API
* deployment
* limitations

---

# 12. Anti-Vibe-Coding Rules

The implementation must NOT:

* generate hundreds of unnecessary files
* use meaningless abstractions
* create placeholder functionality and call it complete
* invent enterprise infrastructure without purpose
* copy generic SaaS dashboard templates
* use AI-generated comments everywhere
* create excessive animations
* use fake metrics
* hide missing functionality behind polished UI

Every major component must have a technical reason to exist.

---

# 13. Portfolio Quality Rule

The final repository must allow a technical interviewer to trace:

```text
Raw data
→ feature engineering
→ label construction
→ training
→ evaluation
→ model registry
→ inference
→ explanation
→ API
→ frontend
→ monitoring
```

without relying on undocumented notebook magic.

---

# 14. Definition of Done

The project is not complete until:

* data pipeline runs
* training pipeline runs
* evaluation runs
* model artifact is versioned
* batch prediction works
* API prediction works
* SHAP explanation works
* simulator works
* dashboard consumes real API data
* monitoring produces real measurements
* tests pass
* Docker deployment works
* README explains architecture and decisions
