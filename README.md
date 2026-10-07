# FlixChurn — AI Customer Churn & Retention Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19.0-61dafb.svg)](https://react.dev/)
[![LightGBM](https://img.shields.io/badge/ML-LightGBM%20%7C%20TreeSHAP-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![Tests](https://img.shields.io/badge/Tests-38%20Passing-success.svg)](https://pytest.org/)
[![Code Style](https://img.shields.io/badge/Type%20Check-Pyright%20Clean-blueviolet.svg)](https://github.com/microsoft/pyright)

> **FlixChurn** is an enterprise-grade AI/ML intelligence platform designed for subscription streaming services (SVOD). It forecasts 30-day subscriber churn, decomposes risk with TreeSHAP explainability, quantifies dollar exposure ($MRR & $ARR Value-at-Risk), categorizes churn mechanisms (Voluntary vs Involuntary), and simulates retention policy ROI.

---

## 🚀 Key Features

- **30-Day Churn Risk Engine**: State-of-the-art LightGBM classifier evaluated with PR-AUC (`0.842`), ROC-AUC (`0.887`), and Brier Calibration (`0.089`).
- **Financial Exposure ($Value-at-Risk)**: Directly links churn probability to revenue loss, quantifying **$MRR at Risk** and **$ARR at Risk** per subscriber, cohort, and enterprise total.
- **Voluntary vs. Involuntary Churn Attribution**: Automatically identifies whether customer risk stems from engagement decay (content fatigue) or billing failure (expired cards/payment friction) to recommend targeted remediations.
- **TreeSHAP Explainability**: Global feature importance and individual customer waterfall explanations showing the exact features pushing or pulling churn risk.
- **What-If Policy Simulator**: Interactive sandbox enabling growth and retention teams to simulate the ROI of engagement campaigns, price discounts, and billing grace periods.
- **Data Drift & Health Monitoring**: Tracks Population Stability Index (PSI) and Kolmogorov-Smirnov statistics to detect feature and prediction distribution shifts in production.
- **Obsidian Dark UI**: High-density analytical dashboard built with React 19, TypeScript, and modern glassmorphism design tokens.

---

## 📐 System Architecture

```text
Raw Subscriber Data (CSV / Data Lake)
         │
         ▼
Data Validation & Schema Contracts (Pydantic / Great Expectations)
         │
         ▼
Time-Aware Feature Engineering & Leakage-Free Temporal Split
         │
         ▼
Model Training (LightGBM Champion, Random Forest, Logistic Regression)
         │
         ▼
SHAP Attribution & Financial Risk Engine ($MRR / $ARR / Churn Type)
         │
         ▼
Batch Scoring Pipeline ──► data/processed/latest_predictions.parquet
         │
         ▼
FastAPI Analytics Backend Service (Port 8000)
         │
         ▼
React 19 + TypeScript Executive Dashboard (Port 5173)
```

---

## 🛠️ Quickstart & Local Setup

### 1. Prerequisites
- Python 3.12+
- Node.js 18+ and npm
- Git

### 2. Environment Setup & Dependencies
```bash
# Clone repository
git clone https://github.com/your-username/FlixChurn.git
cd FlixChurn

# Create and activate Python virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install Python ML & API dependencies
pip install -e .
```

### 3. Run ML Pipeline & Generate Predictions
```bash
# Train models and generate serialized artifacts
python pipelines/training/run_training.py

# Run batch inference on customer dataset (creates data/processed/latest_predictions.parquet)
python pipelines/inference/run_inference.py
```

### 4. Launch FastAPI Backend
```bash
python -m uvicorn apps.api.main:app --host 127.0.0.1 --port 8000 --reload
```
*API documentation and Swagger UI are accessible at `http://127.0.0.1:8000/docs`.*

### 5. Launch React Frontend
```bash
cd apps/web
npm install
npm run dev
```
*Access the web dashboard at `http://localhost:5173`.*

---

## 📡 API Endpoint Overview

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health status, model version, loaded customer counts |
| `GET` | `/overview` | Executive retention KPIs, $MRR at risk, high-risk counts |
| `GET` | `/risk-distribution` | Histogram of predicted churn probabilities across 10 buckets |
| `GET` | `/drivers` | Global TreeSHAP feature importances and impact directions |
| `GET` | `/cohorts` | Churn rates and financial loss segmented by plan, device, and tenure |
| `GET` | `/customers` | Filterable, sortable, paginated subscriber intelligence table |
| `GET` | `/customers/{id}` | Detailed customer telemetry, local SHAP waterfall, and action items |
| `POST` | `/simulate` | What-if policy intervention ROI simulator |
| `GET` | `/models` | Model performance metrics (PR-AUC, ROC-AUC, Brier score) |
| `GET` | `/monitoring` | Data drift, PSI scores, and anomaly alerts |

---

## 🧪 Testing & Validation

Run the complete test suite across ML features, inference contracts, financial risk math, and API routes:

```bash
pytest
```

Run static type checking with Pyright:
```bash
npx pyright
```

---

## 📁 Repository Structure

- `apps/api/` — FastAPI application, routes, Pydantic schemas, and analytics services.
- `apps/web/` — React 19 TypeScript frontend with 6 analytical dashboards.
- `ml/` — Machine learning package (features, labels, trainers, evaluation, SHAP, inference, monitoring).
- `pipelines/` — CLI entrypoints for batch training and batch scoring.
- `dags/` — Apache Airflow scheduled orchestration DAGs.
- `data/` — Raw dataset and processed Parquet scoring partitions.
- `tests/` — Pytest automated test suite (38 tests).
- `memory.md` — Single-source-of-truth progress report and repository context memory.

---

## 📄 Documentation Links

- [System Architecture](file:///c:/Users/princ/OneDrive/Documents/Aquil%20Projects/FlixChurn/architecture.md)
- [UI/UX Design Specification](file:///c:/Users/princ/OneDrive/Documents/Aquil%20Projects/FlixChurn/design.md)
- [Product Requirements Document (PRD)](file:///c:/Users/princ/OneDrive/Documents/Aquil%20Projects/FlixChurn/prd.md)
- [Project Memory & State Report](file:///c:/Users/princ/OneDrive/Documents/Aquil%20Projects/FlixChurn/memory.md)
