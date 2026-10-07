import type { RiskBand } from '../types/api';

/** Format a probability as a percentage string. */
export function formatPct(value: number, decimals = 1): string {
  return `${(value * 100).toFixed(decimals)}%`;
}

/** Format a number with locale-aware separators. */
export function formatNumber(value: number): string {
  return value.toLocaleString('en-US');
}

/** Human-readable feature name from snake_case key. */
export function humanize(key: string): string {
  return key
    .replace(/_/g, ' ')
    .replace(/\b\w/g, l => l.toUpperCase());
}

/** Get CSS class for a risk band. */
export function riskClass(band: RiskBand | string): string {
  return band.toLowerCase() as string;
}

/** Get risk band color variable. */
export function riskColor(band: RiskBand | string): string {
  const colors: Record<string, string> = {
    low: 'var(--risk-low)',
    moderate: 'var(--risk-moderate)',
    high: 'var(--risk-high)',
    critical: 'var(--risk-critical)',
  };
  return colors[band.toLowerCase()] ?? 'var(--text-secondary)';
}

/** Risk band background color variable. */
export function riskBgColor(band: RiskBand | string): string {
  const colors: Record<string, string> = {
    low: 'var(--risk-low-bg)',
    moderate: 'var(--risk-moderate-bg)',
    high: 'var(--risk-high-bg)',
    critical: 'var(--risk-critical-bg)',
  };
  return colors[band.toLowerCase()] ?? 'transparent';
}

/** Chart palette getter. */
export const CHART_COLORS = [
  'var(--chart-1)',
  'var(--chart-2)',
  'var(--chart-3)',
  'var(--chart-4)',
  'var(--chart-5)',
  'var(--chart-6)',
];

/** Hex chart colors for Recharts (can't use CSS vars directly). */
export const CHART_COLORS_HEX = ['#6391ff', '#34d399', '#fbbf24', '#f97316', '#a78bfa', '#ec4899'];

export const RISK_COLORS_HEX: Record<string, string> = {
  low: '#34d399',
  moderate: '#fbbf24',
  high: '#f97316',
  critical: '#ef4444',
};

/** Ordered risk bands for consistent display. */
export const RISK_BAND_ORDER: RiskBand[] = ['low', 'moderate', 'high', 'critical'];
