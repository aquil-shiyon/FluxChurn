import { useQuery, useMutation } from '@tanstack/react-query';
import * as api from '../services/api';

/* --- Analytics --- */
export const useOverview = () =>
  useQuery({ queryKey: ['overview'], queryFn: api.fetchOverview, staleTime: 60_000 });

export const useRiskDistribution = () =>
  useQuery({ queryKey: ['riskDistribution'], queryFn: api.fetchRiskDistribution, staleTime: 60_000 });

export const useDrivers = () =>
  useQuery({ queryKey: ['drivers'], queryFn: api.fetchDrivers, staleTime: 120_000 });

export const useCohorts = (groupBy: string) =>
  useQuery({ queryKey: ['cohorts', groupBy], queryFn: () => api.fetchCohorts(groupBy), staleTime: 60_000 });

/* --- Customers --- */
export const useCustomers = (params: {
  risk_band?: string;
  churn_type?: string;
  sort_by?: string;
  order?: string;
  search?: string;
  limit?: number;
  offset?: number;
}) =>
  useQuery({
    queryKey: ['customers', params],
    queryFn: () => api.fetchCustomers(params),
    staleTime: 30_000,
  });

export const useCustomerRisk = (customerId: string) =>
  useQuery({
    queryKey: ['customerRisk', customerId],
    queryFn: () => api.fetchCustomerRisk(customerId),
    enabled: !!customerId,
    staleTime: 60_000,
  });

export const useCustomerExplanation = (customerId: string) =>
  useQuery({
    queryKey: ['customerExplanation', customerId],
    queryFn: () => api.fetchCustomerExplanation(customerId),
    enabled: !!customerId,
    staleTime: 60_000,
  });

/* --- Prediction & Simulation --- */
export const useSimulate = () =>
  useMutation({
    mutationFn: ({ customerId, scenario }: { customerId: string; scenario: Record<string, unknown> }) =>
      api.postSimulate(customerId, scenario),
  });

/* --- Models --- */
export const useModels = () =>
  useQuery({ queryKey: ['models'], queryFn: api.fetchModels, staleTime: 300_000 });

export const useModelMetrics = (version: string) =>
  useQuery({
    queryKey: ['modelMetrics', version],
    queryFn: () => api.fetchModelMetrics(version),
    staleTime: 300_000,
  });

/* --- Monitoring --- */
export const useDataQuality = () =>
  useQuery({ queryKey: ['dataQuality'], queryFn: api.fetchDataQuality, staleTime: 120_000 });

export const useDrift = () =>
  useQuery({ queryKey: ['drift'], queryFn: api.fetchDrift, staleTime: 120_000 });

export const usePerformance = () =>
  useQuery({ queryKey: ['performance'], queryFn: api.fetchPerformance, staleTime: 120_000 });
