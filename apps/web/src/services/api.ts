import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: `${API_BASE}/api/v1`,
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' },
});

/* --- Analytics --- */
export const fetchOverview = () => api.get('/overview').then(r => r.data);
export const fetchRiskDistribution = () => api.get('/risk-distribution').then(r => r.data);
export const fetchDrivers = () => api.get('/drivers').then(r => r.data);
export const fetchCohorts = (groupBy: string) =>
  api.get('/cohorts', { params: { group_by: groupBy } }).then(r => r.data);

/* --- Customers --- */
export const fetchCustomers = (params: {
  risk_band?: string;
  churn_type?: string;
  sort_by?: string;
  order?: string;
  search?: string;
  limit?: number;
  offset?: number;
}) => api.get('/customers', { params }).then(r => r.data);

export const fetchCustomerRisk = (customerId: string) =>
  api.get(`/customers/${customerId}/risk`).then(r => r.data);

export const fetchCustomerExplanation = (customerId: string) =>
  api.get(`/customers/${customerId}/explanation`).then(r => r.data);

/* --- Prediction --- */
export const postPredict = (features: Record<string, unknown>) =>
  api.post('/predict', features).then(r => r.data);

export const postSimulate = (customerId: string, scenario: Record<string, unknown>) =>
  api.post('/simulate', { customer_id: customerId, scenario }).then(r => r.data);

/* --- Models --- */
export const fetchModels = () => api.get('/models').then(r => r.data);
export const fetchModelMetrics = (version: string) =>
  api.get(`/models/${version}/metrics`).then(r => r.data);

/* --- Monitoring --- */
export const fetchDataQuality = () => api.get('/monitoring/data-quality').then(r => r.data);
export const fetchDrift = () => api.get('/monitoring/drift').then(r => r.data);
export const fetchPerformance = () => api.get('/monitoring/performance').then(r => r.data);

/* --- Health --- */
export const fetchHealth = () =>
  axios.get(`${API_BASE}/health`).then(r => r.data);

export default api;
