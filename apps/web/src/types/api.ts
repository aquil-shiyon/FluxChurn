/* API types matching the FastAPI backend contracts. */

/* --- Risk Bands --- */
export type RiskBand = 'low' | 'moderate' | 'high' | 'critical';

/* --- Overview --- */
export interface OverviewData {
  prediction_horizon_days: number;
  model_version: string;
  last_scored: string | null;
  total_customers: number;
  risk_distribution: Record<RiskBand, RiskBandInfo>;
  avg_churn_probability: number;
  predicted_churn_volume: number;
  high_risk_count: number;
  high_risk_percentage?: number;
  total_mrr_at_risk?: number;
  total_arr_at_risk?: number;
  high_risk_mrr_at_risk?: number;
  voluntary_risk_count?: number;
  involuntary_risk_count?: number;
}

export interface RiskBandInfo {
  count: number;
  percentage: number;
}

/* --- Drivers --- */
export interface DriversData {
  top_features: DriverFeature[];
}

export interface DriverFeature {
  feature: string;
  mean_abs_shap: number;
  direction: string;
}

/* --- Customer List --- */
export interface CustomerListData {
  customers: CustomerSummary[];
  total: number;
  limit: number;
  offset: number;
}

export interface CustomerSummary {
  customer_id: string;
  churn_probability: number;
  risk_band: RiskBand;
  prediction_horizon_days: number;
  model_version: string;
  prediction_timestamp: string;
  monthly_fee?: number;
  mrr_at_risk?: number;
  arr_at_risk?: number;
  churn_type_risk?: string;
}

/* --- Customer Risk --- */
export interface CustomerRiskData {
  customer_id: string;
  churn_probability: number;
  risk_band: RiskBand;
  prediction_horizon_days: number;
  model_version: string;
  prediction_timestamp: string;
  monthly_fee?: number;
  mrr_at_risk?: number;
  arr_at_risk?: number;
  churn_type_risk?: string;
  features: Record<string, number | string>;
}

/* --- Customer Explanation --- */
export interface ExplanationData {
  customer_id: string;
  model_version: string;
  base_value: number;
  predicted_value: number;
  top_contributors: ExplanationFeature[];
}

export interface ExplanationFeature {
  feature: string;
  feature_value: number | string;
  contribution: number;
  direction: string;
}

/* --- Cohorts --- */
export interface CohortsData {
  group_by: string;
  cohorts: Record<string, CohortInfo>;
  available_dimensions: string[];
}

export interface CohortInfo {
  total: number;
  avg_churn_probability: number;
  risk_distribution: Record<RiskBand, RiskBandInfo>;
}

/* --- Simulation --- */
export interface SimulationResult {
  customer_id: string;
  baseline_probability: number;
  scenario_probability: number;
  probability_change_pp: number;
  risk_band_baseline: RiskBand;
  risk_band_scenario: RiskBand;
  model_version: string;
  monthly_fee?: number;
  mrr_saved?: number;
  annual_value_preserved?: number;
  disclaimer: string;
}

/* --- Models --- */
export interface ModelComparisonData {
  models: Record<string, ModelMetrics>;
  selected: string;
  selection_rationale: Record<string, unknown>;
}

export interface ModelMetrics {
  name: string;
  pr_auc: number;
  roc_auc: number;
  brier_score: number;
  f1: number;
  precision: number;
  recall: number;
  recall_at_10?: number;
  precision_at_10?: number;
  lift_at_10?: number;
}

/* --- Model Evaluation --- */
export interface EvaluationData {
  test_metrics: Record<string, number>;
  validation_metrics: Record<string, Record<string, number>>;
  calibration: Record<string, unknown>;
  model_selection: Record<string, unknown>;
}

/* --- Monitoring --- */
export interface DataQualityData {
  total_records: number;
  missing_values: Record<string, number>;
  duplicate_count: number;
  schema_valid: boolean;
  range_violations: Record<string, unknown>;
  issues: string[];
}

export interface DriftData {
  status: string;
  message?: string;
  features?: DriftFeature[];
  summary?: Record<string, unknown>;
}

export interface DriftFeature {
  feature: string;
  psi: number;
  ks_statistic: number;
  threshold: number;
  status: string;
}

export interface PerformanceData {
  test_metrics: Record<string, number>;
  validation_metrics: Record<string, Record<string, number>>;
  calibration: Record<string, unknown>;
  model_selection: Record<string, unknown>;
}
