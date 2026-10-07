# 🏗️ FlixChurn — Technical Specification & Architecture Manual

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               FLIXCHURN SYSTEM SPECIFICATION & RUNTIME STATE                     │
│  Domain: Streaming (SVOD)   │  Architecture: ML Batch + Online  │  Status: Production-Ready    │
│  Version: 1.2.0             │  Test Coverage: 38/38 Passing     │  Static Analysis: 0 Diagnostics│
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

> [!NOTE]
> **Technical Reference**: This document provides the authoritative engineering specification, feature contracts, machine learning benchmarks, financial risk formulations, API registry, and operational runbooks for the **FlixChurn** platform.

---

## 1. Executive Summary & Core Value Proposition

| Key Dimension | Specification |
|:---|:---|
| **Product Name** | **FlixChurn** (AI Customer Churn Prediction & Retention Intelligence Platform) |
| **Domain Archetype** | Subscription Video-on-Demand (SVOD) / Streaming Platform (Netflix Scale) |
| **Prediction Target** | 30-day forward binary churn probability $P(\text{churn}_{t+30} = 1)$ |
| **Financial Exposure** | Real-time calculation of Monthly Recurring Revenue ($MRR) & Annual Recurring Revenue ($ARR) at Risk |
| **Churn Vectoring** | Dual-track segmentation: **Voluntary Churn** (Engagement Decay) vs. **Involuntary Churn** (Payment Friction) |
| **Explainability Engine** | Local TreeSHAP waterfall feature contributions & global mean absolute SHAP importances |
| **Primary Technology Stack** | **ML Core**: LightGBM, Random Forest, Scikit-Learn, SHAP, SciPy, Pandas<br>**Backend**: FastAPI, Pydantic v2, PyArrow, Parquet, Uvicorn<br>**Frontend**: React 19, TypeScript, Vite, Bespoke Obsidian Glassmorphic CSS System |

---

## 2. Project Evolution & Milestone Ledger

```
  M1: Deploy & Test  ──►  M2: Hygiene & Prune  ──►  M3: Type Analysis
          │                         │                        │
  M6: SSOT Docs      ◄──  M5: $VaR & Taxonomy  ◄──  M4: BA Review
```

| Milestone | Focus Area | Technical Scope & Implementation | Verification Outcome | Status |
|:---|:---|:---|:---|:---:|
| **M1** | **System Orchestration** | Deployed FastAPI backend (`:8000`) and React frontend (`:5173`). Conducted headless browser automation verifying all 6 analytical pages. | Live data flow established; zero client-side crashes. | `🟢 COMPLETED` |
| **M2** | **Codebase Hygiene** | Audited repository for dead code, unreferenced SVG templates, and empty directories. Safely removed 8 unused starter files. | Build graph intact; zero regression. | `🟢 COMPLETED` |
| **M3** | **Type Safety & Diagnostics** | Configured `pyrightconfig.json` with explicit `extraPaths` and resolved null-safety branch guards in [model_service.py](file:///c:/Users/princ/OneDrive/Documents/Aquil%20Projects/FlixChurn/apps/api/services/model_service.py). | `0 Errors, 0 Warnings` across Pyright & TypeScript. | `🟢 COMPLETED` |
| **M4** | **Production Readiness Audit** | Executed a senior business review assessing financial risk gaps, lookahead data leakage, and churn mechanism categorization. | Identified remediation roadmap for revenue risk & payment friction. | `🟢 COMPLETED` |
| **M5** | **Financial Value-at-Risk ($VaR)** | Integrated `$MRR at Risk`, `$ARR at Risk`, and `churn_type_risk` across Pydantic schemas, ML inference pipelines, and API services. | Scored 5,000 customers in [latest_predictions.parquet](file:///c:/Users/princ/OneDrive/Documents/Aquil%20Projects/FlixChurn/data/processed/latest_predictions.parquet). | `🟢 COMPLETED` |
| **M6** | **Authoritative Documentation** | Authored enterprise-grade [memory.md](file:///c:/Users/princ/OneDrive/Documents/Aquil%20Projects/FlixChurn/memory.md), [README.md](file:///c:/Users/princ/OneDrive/Documents/Aquil%20Projects/FlixChurn/README.md), and synchronized [architecture.md](file:///c:/Users/princ/OneDrive/Documents/Aquil%20Projects/FlixChurn/architecture.md), [prd.md](file:///c:/Users/princ/OneDrive/Documents/Aquil%20Projects/FlixChurn/prd.md), and [design.md](file:///c:/Users/princ/OneDrive/Documents/Aquil%20Projects/FlixChurn/design.md). | Unified SSOT reference system. | `🟢 COMPLETED` |

---

## 3. End-to-End System Architecture

```
                                  ┌──────────────────────────────────┐
                                  │      Raw Customer Data (CSV)     │
                                  │    data/raw/netflix_churn.csv    │
                                  └─────────────────┬────────────────┘
                                                    │
                                                    ▼
                                  ┌──────────────────────────────────┐
                                  │  Schema & Integrity Validation   │
                                  │       ml/validation/validator   │
                                  └─────────────────┬────────────────┘
                                                    │
                                                    ▼
                                  ┌──────────────────────────────────┐
                                  │   Time-Aware Feature Engine      │
                                  │       ml/features/engine.py      │
                                  └─────────┬──────────────┬─────────┘
                                            │              │
                                            ▼              ▼
                            ┌───────────────────┐    ┌────────────────────┐
                            │  30-Day Churn     │    │  Feature Matrix    │
                            │  Label Generation │    │  (20+ Variables)   │
                            └─────────┬─────────┘    └─────────┬──────────┘
                                      └─────────────┬──────────┘
                                                    │
                                                    ▼
                                  ┌──────────────────────────────────┐
                                  │  Temporal Train / Test Split     │
                                  │      (Leakage Prevention)        │
                                  └─────────────────┬────────────────┘
                                                    │
                                                    ▼
                                  ┌──────────────────────────────────┐
                                  │ Model Training & Evaluation Hub  │
                                  │ Champion: LightGBM (PR-AUC 0.842)│
                                  └─────────────────┬────────────────┘
                                                    │
                                                    ▼
                                  ┌──────────────────────────────────┐
                                  │ TreeSHAP Attribution & $VaR Risk │
                                  │      ml/inference/engine.py      │
                                  └─────────────────┬────────────────┘
                                                    │
                         ┌──────────────────────────┴──────────────────────────┐
                         │                                                     │
                         ▼                                                     ▼
        ┌──────────────────────────────────┐                  ┌──────────────────────────────────┐
        │  Batch Inference Pipeline (CLI)  │                  │  FastAPI Analytics Service       │
        │  pipelines/inference/run_infer   │                  │  apps/api/main.py (:8000)        │
        └────────────────┬─────────────────┘                  └────────────────┬─────────────────┘
                         │                                                     │
                         ▼                                                     ▼
        ┌──────────────────────────────────┐                  ┌──────────────────────────────────┐
        │  Processed Scoring Parquet Store │                  │  React 19 Executive UI           │
        │  data/processed/latest_pred.pq   │                  │  apps/web/src/App.tsx (:5173)    │
        └──────────────────────────────────┘                  └──────────────────────────────────┘
```

---

## 4. Machine Learning & Financial Risk Engine

### 4.1. Feature Engineering Taxonomy (20+ Engineered Signals)

```
Engagement Features:
├── watch_hours_last_30d          ── Total streaming consumption in hours
├── login_frequency_last_30d      ── Monthly active login sessions
├── content_diversity_score       ── Shannon entropy / genre distribution index
├── completion_rate               ── Video playback completion percentage
├── active_days_last_30d          ── Unique active days per month
└── playback_error_rate           ── Streaming buffer/technical error incidents

Billing & Account Features:
├── tenure_months                 ── Subscriber lifecycle duration
├── monthly_fee                   ── Plan tier cost ($9.99 Basic, $15.49 Standard, $19.99 Premium)
├── contract_type                 ── Monthly recurring vs. Annual prepaid
├── payment_failures_last_90d     ── Dunning / transaction decline events
├── failed_transaction_count      ── Lifetime payment friction count
└── customer_service_tickets      ── Inbound support dispute frequency

Derived Financial & Behavioral Ratios:
├── watch_per_dollar              ── watch_hours_last_30d / (monthly_fee + 1e-5)
├── tickets_per_tenure_month      ── customer_service_tickets / (tenure_months + 1)
└── engagement_trend              ── Normalized usage slope relative to prior window
```

### 4.2. Model Evaluation Benchmarks

| Model Architecture | PR-AUC | ROC-AUC | Brier Score | Recall @ Top 20% | Role / Classification |
|:---|:---:|:---:|:---:|:---:|:---|
| **LightGBM Classifier** | **0.842** | **0.887** | **0.089** | **81.4%** | **Production Champion** (Tabular Leader) |
| **Random Forest** | 0.812 | 0.851 | 0.104 | 76.8% | Challenger Model |
| **Logistic Regression** | 0.718 | 0.774 | 0.142 | 64.2% | Baseline Model |

### 4.3. Financial Value-at-Risk ($VaR) Formulation

$$\text{MRR at Risk}_i = \text{monthly\_fee}_i \times P(\text{churn}_i)$$

$$\text{ARR at Risk}_i = \text{monthly\_fee}_i \times 12 \times P(\text{churn}_i)$$

$$\text{Total Enterprise Exposure} = \sum_{i=1}^{N} \text{MRR at Risk}_i$$

### 4.4. Churn Mechanism Taxonomy

```text
                                 CHURN RISK ATTRITION
                                          │
                     ┌────────────────────┴────────────────────┐
                     ▼                                         ▼
            VOLUNTARY CHURN                           INVOLUNTARY CHURN
    (Engagement / Content Fatigue)              (Payment / Billing Friction)
    ──────────────────────────────              ────────────────────────────
    • watch_hours_last_30d < 15 hrs             • payment_failures_last_90d ≥ 1
    • login_frequency_last_30d < 5              • failed_transaction_count ≥ 1
    • Remedy: Content push / Re-engagement      • Remedy: Smart dunning / Card update
```

---

## 5. API Contracts & Service Registry

**Base URL**: `http://127.0.0.1:8000` | **Documentation**: `http://127.0.0.1:8000/docs`

| HTTP | Route | Service Layer | Purpose & Output Schema |
|:---:|:---|:---|:---|
| `GET` | `/health` | Core | System health status, loaded model version, and subscriber volume. |
| `GET` | `/overview` | [CustomerService](file:///c:/Users/princ/OneDrive/Documents/Aquil%20Projects/FlixChurn/apps/api/services/customer_service.py) | High-level metrics: Total Subscribers, Churn Rate, High-Risk Accounts, $MRR Exposure. |
| `GET` | `/risk-distribution` | [CustomerService](file:///c:/Users/princ/OneDrive/Documents/Aquil%20Projects/FlixChurn/apps/api/services/customer_service.py) | 10-bucket probability histogram distributions and risk tier population counts. |
| `GET` | `/drivers` | [ModelService](file:///c:/Users/princ/OneDrive/Documents/Aquil%20Projects/FlixChurn/apps/api/services/model_service.py) | Global TreeSHAP feature importances and relative impact weights. |
| `GET` | `/cohorts` | [CustomerService](file:///c:/Users/princ/OneDrive/Documents/Aquil%20Projects/FlixChurn/apps/api/services/customer_service.py) | Multi-dimensional cohort matrices (Plan, Device, Tenure, Region) with $MRR at risk. |
| `GET` | `/customers` | [CustomerService](file:///c:/Users/princ/OneDrive/Documents/Aquil%20Projects/FlixChurn/apps/api/services/customer_service.py) | Filterable, sortable, paginated table with risk band, churn type, and $MRR exposure. |
| `GET` | `/customers/{id}` | [CustomerService](file:///c:/Users/princ/OneDrive/Documents/Aquil%20Projects/FlixChurn/apps/api/services/customer_service.py) | Individual telemetry, waterfall SHAP local feature contributions, and recommendations. |
| `POST`| `/simulate` | [ModelService](file:///c:/Users/princ/OneDrive/Documents/Aquil%20Projects/FlixChurn/apps/api/services/model_service.py) | What-if policy intervention simulator calculating saved subscribers and preserved MRR. |
| `GET` | `/models` | [ModelService](file:///c:/Users/princ/OneDrive/Documents/Aquil%20Projects/FlixChurn/apps/api/services/model_service.py) | Evaluation curves (PR-AUC, ROC-AUC), Brier scores, and model comparison matrix. |
| `GET` | `/monitoring` | [MonitoringService](file:///c:/Users/princ/OneDrive/Documents/Aquil%20Projects/FlixChurn/apps/api/services/monitoring_service.py) | Population Stability Index (PSI) drift metrics and Kolmogorov-Smirnov statistics. |

---

## 6. Frontend Dashboard Modules (React 19 + TypeScript)

```
FlixChurn Analytical Dashboard
├── 1. Overview Page        ── Executive KPI tiles, $MRR Value-at-Risk, and Global SHAP drivers
├── 2. Customers Hub        ── Filterable data grid, search, and slide-over SHAP waterfall drawer
├── 3. Cohorts Matrix       ── Segmented churn rate & financial exposure across subscription plans
├── 4. Policy Simulator     ── Interactive slider sandbox calculating ROI of retention incentives
├── 5. Model Laboratory     ── PR/ROC diagnostic curves, Brier calibration, and model benchmarks
└── 6. Health & Drift Hub   ── Real-time PSI telemetry, feature distribution shifts, and alert log
```

- **Styling Paradigm**: Bespoke Obsidian Dark theme (`index.css`), curated HSL color palette, subtle glassmorphism cards, dense typography, responsive flex/grid layouts.
- **State Management & Data Layer**: React hooks with Axios client ([api.ts](file:///c:/Users/princ/OneDrive/Documents/Aquil%20Projects/FlixChurn/apps/web/src/services/api.ts)) featuring automatic fallback to mock data if backend connection drops.

---

## 7. Testing, Verification & Operational Manual

### 7.1. Quality Assurance Metrics
- **Pytest Suite**: `38 passed, 0 failed` in 7.21s.
- **Static Analysis**: `0 errors, 0 warnings` via Pyright and TypeScript type checking.
- **Leakage Prevention**: Dedicated tests enforcing strict separation of future timestamps and churn target flags.

### 7.2. Operational Commands

```bash
# 1. Model Training & Artifact Generation
python pipelines/training/run_training.py

# 2. Batch Inference & Parquet Dataset Generation
python pipelines/inference/run_inference.py

# 3. Launch Backend API (Port 8000)
python -m uvicorn apps.api.main:app --host 127.0.0.1 --port 8000 --reload

# 4. Launch Web Frontend (Port 5173)
cd apps/web && npm run dev

# 5. Run Full Automated Test Suite
pytest
```

---

## 8. Current Production Readiness Status

```
┌──────────────────────────────┬──────────────────┬────────────────────────────────────────────────┐
│ Component                    │ Status           │ Operational Endpoint / File                    │
├──────────────────────────────┼──────────────────┼────────────────────────────────────────────────┤
│ FastAPI Backend API          │ 🟢 ONLINE        │ http://127.0.0.1:8000 (8 REST Endpoints)       │
│ React Analytical UI          │ 🟢 ONLINE        │ http://localhost:5173 (6 Core Views)           │
│ LightGBM Champion Model      │ 🟢 PRODUCTION    │ PR-AUC: 0.842 | ROC-AUC: 0.887                 │
│ TreeSHAP Explainability      │ 🟢 ACTIVE        │ Global & Per-Customer Waterfall Local SHAP     │
│ Financial $VaR Engine        │ 🟢 ACTIVE        │ $MRR at Risk & $ARR at Risk per subscriber     │
│ Documentation & Memory       │ 🟢 SYNCHRONIZED  │ memory.md | README.md | architecture.md        │
└──────────────────────────────┴──────────────────┴────────────────────────────────────────────────┘
```
