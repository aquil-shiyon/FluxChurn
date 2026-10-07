# UI/UX Design Specification

# FlixChurn — Stitch AI Build Prompt

Design and generate the frontend interface for **FlixChurn**, an AI-powered customer churn prediction and retention intelligence platform for a subscription streaming business.

This is an **analyst-grade AI/ML product**, not a generic SaaS dashboard and not a CRUD application.

The interface should look like a serious internal decision-support system used by data analysts, product managers and retention teams.

---

# 1. Design Direction

Create a sophisticated analytical workspace with:

* high information density
* restrained visual hierarchy
* strong typography
* precise data visualization
* subtle depth
* minimal decorative elements
* clear model transparency
* professional enterprise analytics aesthetic

Do NOT make it look like:

* a generic admin panel
* a banking dashboard
* a CRM
* a social media application
* a consumer streaming UI
* a template-generated SaaS product

---

# 2. Visual Personality

Desired characteristics:

```text
Analytical
Precise
Technical
Premium
Quiet
Data-centric
Professional
Evidence-driven
```

Avoid:

```text
Playful
Cartoonish
Over-animated
Neon
Excessive gradients
Huge cards
Generic KPI tiles
```

Use a dark-first analytical interface with strong contrast and restrained accent usage.

Do not overuse the Netflix brand identity or Netflix-specific visual assets.

This is a fictional streaming-business analytics product.

---

# 3. Global Layout

Desktop-first application.

Recommended structure:

```text
┌───────────────────────────────────────────────────────────────┐
│ ChurnIQ       Prediction: 30 Days     Model v1.2.0     User │
├──────────────┬────────────────────────────────────────────────┤
│              │                                                │
│ Risk         │                                                │
│ Overview     │                 MAIN WORKSPACE                  │
│              │                                                │
│ Customers    │                                                │
│              │                                                │
│ Cohorts      │                                                │
│              │                                                │
│ Simulator    │                                                │
│              │                                                │
│ Model Lab    │                                                │
│              │                                                │
│ Monitoring   │                                                │
│              │                                                │
└──────────────┴────────────────────────────────────────────────┘
```

---

# 4. Primary Navigation

Use:

1. Risk Overview
2. Customer Intelligence
3. Cohorts
4. Risk Simulator
5. Model Lab
6. Monitoring

Do not include:

```text
Users
Products
Orders
Settings
CRUD Management
```

unless required by the actual application.

---

# 5. Screen 1 — Risk Overview

Purpose:

Answer:

> "What is the current churn-risk situation?"

---

## Header

Show:

```text
Customer Churn Intelligence
30-day prediction horizon

Last scored:
[date/time]

Model:
v1.2.0
```

---

## Risk Summary

Use compact analytical metrics:

```text
Eligible Customers
High-Risk Customers
Average Churn Probability
Predicted Churn Volume
```

Do not use oversized marketing-style cards.

---

## Risk Distribution

Visualization:

Horizontal or vertical distribution:

```text
LOW       ███████████████
MODERATE  █████████
HIGH      █████
CRITICAL  ██
```

Allow clicking a risk band to filter downstream analysis.

---

## Risk Trend

Line chart:

```text
Predicted churn risk over time
```

Allow:

* 7 days
* 30 days
* 90 days

---

## Risk Concentration

Show which cohorts contain the highest concentration of elevated risk.

Potential dimensions:

```text
Plan
Tenure
Engagement
Region
```

Only display dimensions supported by actual data.

---

## Predictive Drivers

Use a ranked horizontal bar chart.

Example:

```text
Days since activity             █████████████
Declining watch time            █████████
Low active-day frequency        ███████
Payment failures                █████
Low completion rate             ████
```

Label clearly:

**Top Predictive Features**

not:

**Causes of Churn**

---

# 6. Screen 2 — Customer Intelligence

Purpose:

Investigate individual customer risk.

---

## Search

Use:

```text
Search customer ID
```

Do not create generic customer management functionality.

---

## Customer Risk Header

Display:

```text
Customer C10291

HIGH RISK

78.4%
Predicted churn probability

30-day horizon

Model v1.2.0
```

---

## Behavioral Timeline

Show a timeline of:

* activity
* watch time
* sessions
* subscription events
* payment events

Use a compact analytical visualization.

---

## Engagement Trend

Display:

```text
Watch time
Active days
Sessions
Completion rate
```

Use synchronized trend charts where appropriate.

---

## Risk Explanation

Create a prominent explanation panel:

```text
WHY THIS PREDICTION?

Feature                       Impact

Days since last activity     ↑ High
Watch-time decline           ↑ High
Active days                  ↑ Medium
Tenure                       ↓ Low risk
```

Use directional indicators.

Do not use unexplained AI language.

---

## Feature Detail

Allow analysts to inspect:

```text
Feature
Current Value
Reference/Population Value
Contribution
Direction
```

---

# 7. Screen 3 — Cohort Explorer

Purpose:

Find groups with elevated predicted risk.

---

## Controls

Provide analytical filters:

```text
Date
Subscription plan
Tenure
Risk band
Engagement level
Region
```

Filters should dynamically update the analysis.

---

## Main Visualization

Create a cohort comparison matrix.

Example:

```text
                 LOW       MODERATE     HIGH

New users        62%        23%         15%
Mid tenure       71%        19%         10%
Long tenure      78%        14%          8%
```

Do not rank cohorts as "best" or "worst".

Use neutral analytical terminology.

---

## Cohort Trend

Show how risk changes over time.

---

# 8. Screen 4 — Risk Simulator

Purpose:

Explore model sensitivity to hypothetical behavioral changes.

Layout:

```text
CURRENT STATE             SCENARIO

Watch time                 Watch time
[4.2 hrs]                  [7.0 hrs]

Active days                Active days
[2]                        [4]

Days inactive              Days inactive
[11]                       [4]

             ↓

       MODEL PREDICTION

Current     72%
Scenario    48%
Change      -24 pp
```

Use sliders only for valid continuous/numerical features.

Categorical features should use controlled selectors.

---

## Mandatory Disclaimer

Place near the result:

> Scenario output represents model sensitivity, not a causal estimate of how changing customer behavior will affect churn.

---

# 9. Screen 5 — Model Lab

Purpose:

Allow technical users to inspect model quality.

---

## Model Header

Display:

```text
Production Model
Gradient Boosting

Version: 1.2.0
Training date: [date]
Dataset version: [version]
Feature version: [version]
```

---

## Metrics

Display:

```text
PR-AUC
ROC-AUC
Recall
Precision
F1
Brier Score
Recall@Top-K
Lift@Top-K
```

---

## Calibration

Create calibration plot:

```text
Predicted probability
vs
Observed frequency
```

---

## Confusion Matrix

Use a clean matrix.

---

## Model Comparison

Allow comparison between:

```text
Logistic Regression
Random Forest
Gradient Boosting
```

Do not visually declare a "winner".

Show the metrics and let the analyst interpret them.

---

# 10. Screen 6 — Monitoring

Purpose:

Determine whether the ML system remains trustworthy.

---

## Data Quality

Display:

```text
Schema status
Missingness
Duplicate records
Invalid values
Freshness
```

---

## Feature Drift

Show:

```text
Feature
Drift score
Threshold
Status
```

Example:

```text
days_since_activity
PSI: 0.21
Threshold: 0.20
Status: REVIEW
```

---

## Prediction Drift

Show:

```text
Current probability distribution
Reference probability distribution
```

---

## Performance Monitoring

When actual labels become available:

```text
PR-AUC trend
Recall trend
Calibration trend
Brier score trend
```

Do not fabricate unavailable historical metrics.

---

# 11. Interaction Rules

Use interactions that support investigation:

* hover for exact values
* click chart elements to filter
* drill from cohort → customers
* drill from customer → explanation
* model version selection
* date-range selection
* scenario simulation
* metric tooltips

Avoid:

* excessive animations
* animated backgrounds
* unnecessary page transitions
* decorative charts

---

# 12. Data States

Every analytical component must support:

### Loading

Use skeletons appropriate to chart type.

### Empty

Explain why data is unavailable.

### Error

Show meaningful error messages.

### Stale

Clearly indicate stale prediction/model data.

---

# 13. Responsive Behaviour

Primary target:

```text
1440px desktop
```

Support:

```text
1280px
1024px
```

Do not sacrifice analytical readability merely to make the interface mobile-first.

---

# 14. Accessibility

Use:

* sufficient contrast
* keyboard navigation
* semantic labels
* non-color indicators
* readable chart annotations

Do not communicate risk exclusively through red/green colors.

---

# 15. Component Style

Prefer:

* compact cards
* analytical tables
* restrained borders
* subtle separators
* consistent spacing
* precise chart labels
* meaningful empty states

Avoid:

* oversized cards
* excessive rounded containers
* floating glassmorphism everywhere
* giant hero sections
* decorative illustrations

---

# 16. Design Principle

Every visible component must answer one of these questions:

```text
What is the current risk?
Where is risk concentrated?
Why does the model predict this?
How has risk changed?
How reliable is the model?
What happens under a hypothetical scenario?
Is the model/data still trustworthy?
```

If a component does not answer one of these questions, remove it.

---

# 17. Implementation Status & Component Registry

The frontend is fully implemented in React 19 + TypeScript with live API connections to the FastAPI backend:

* **Executive Overview (`OverviewPage.tsx`)**: High-level KPIs, $MRR Value-at-Risk, probability distribution histogram, and global SHAP drivers.
* **Customer Intelligence (`CustomersPage.tsx`)**: Filterable data table, pagination, search, risk tier filtering, and interactive slide-over detail drawer with waterfall SHAP breakdown.
* **Cohort Matrix (`CohortsPage.tsx`)**: Churn probability and $MRR exposure segmented across plans, devices, tenure, and region.
* **What-If Simulator (`SimulatorPage.tsx`)**: Interactive scenario sliders calculating saved subscribers and preserved MRR.
* **Model Laboratory (`ModelLabPage.tsx`)**: PR-AUC, ROC-AUC, Brier score calibration, and model comparison.
* **Health & Drift Monitor (`MonitoringPage.tsx`)**: PSI stability gauge, feature drift detection, and data health telemetry.

