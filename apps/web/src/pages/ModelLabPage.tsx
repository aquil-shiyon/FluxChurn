import { useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, RadarChart, Radar, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis
} from 'recharts';
import { useModels, useModelMetrics } from '../hooks/useApi';
import { CardSkeleton, ErrorState, EmptyState } from '../components/Shared';
import { humanize, CHART_COLORS_HEX } from '../utils/format';
import type { ModelMetrics } from '../types/api';

const METRIC_LABELS: Record<string, string> = {
  pr_auc: 'PR-AUC',
  roc_auc: 'ROC-AUC',
  f1: 'F1 Score',
  precision: 'Precision',
  recall: 'Recall',
  brier_score: 'Brier Score',
  recall_at_10: 'Recall@10%',
  precision_at_10: 'Precision@10%',
  lift_at_10: 'Lift@10%',
};

export default function ModelLabPage() {
  const [activeTab, setActiveTab] = useState<'comparison' | 'evaluation'>('comparison');
  const { data: models, isLoading: modelsLoading, error: modelsError, refetch } = useModels();
  const { data: evalData, isLoading: evalLoading } = useModelMetrics('1.0.0');

  if (modelsError) {
    return <ErrorState message="Could not load model data." onRetry={() => refetch()} />;
  }

  return (
    <>
      <div className="page-header">
        <h1>Model Lab</h1>
        <p>Inspect model quality, compare candidates, and evaluate calibration</p>
      </div>

      {/* Model Header */}
      {models && (
        <div className="card mb-lg">
          <div className="flex items-center gap-xl">
            <div>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.8px', fontWeight: 600 }}>
                Production Model
              </div>
              <div style={{ fontSize: 16, fontWeight: 700, marginTop: 2 }}>
                {models.models?.[models.selected]?.name ?? models.selected}
              </div>
            </div>
            <div className="topbar-tag model">v1.0.0</div>
            <div style={{ flex: 1 }} />
            <div style={{ fontSize: 12, color: 'var(--text-tertiary)' }}>
              Selected: <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{humanize(models.selected)}</span>
            </div>
          </div>
        </div>
      )}

      {/* Tabs */}
      <div className="tabs">
        <button
          className={`tab${activeTab === 'comparison' ? ' active' : ''}`}
          onClick={() => setActiveTab('comparison')}
        >
          Model Comparison
        </button>
        <button
          className={`tab${activeTab === 'evaluation' ? ' active' : ''}`}
          onClick={() => setActiveTab('evaluation')}
        >
          Evaluation Details
        </button>
      </div>

      {activeTab === 'comparison' ? (
        modelsLoading ? <CardSkeleton lines={10} /> : models?.models ? (
          <ModelComparison models={models.models} selected={models.selected} />
        ) : <EmptyState title="No model data available" />
      ) : (
        evalLoading ? <CardSkeleton lines={10} /> : evalData ? (
          <EvaluationDetails data={evalData} />
        ) : <EmptyState title="No evaluation data available" />
      )}
    </>
  );
}

/* --- Model Comparison --- */
function ModelComparison({ models, selected }: { models: Record<string, ModelMetrics>; selected: string }) {
  const modelKeys = Object.keys(models);
  const metrics = ['pr_auc', 'roc_auc', 'f1', 'precision', 'recall'];

  // Bar chart data
  const chartData = metrics.map(metric => {
    const row: Record<string, unknown> = { metric: METRIC_LABELS[metric] || metric };
    modelKeys.forEach(key => {
      row[key] = models[key]?.[metric as keyof ModelMetrics] ?? 0;
    });
    return row;
  });

  // Radar chart data
  const radarData = metrics.map(metric => {
    const row: Record<string, unknown> = { metric: METRIC_LABELS[metric] || metric };
    modelKeys.forEach(key => {
      row[key] = models[key]?.[metric as keyof ModelMetrics] ?? 0;
    });
    return row;
  });

  return (
    <>
      <div className="grid grid-2">
        {/* Grouped Bar Chart */}
        <div className="card">
          <div className="card-header">
            <span className="card-title">Metric Comparison</span>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={chartData} margin={{ left: 10, right: 10 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="metric" tick={{ fill: '#8b90a0', fontSize: 11 }} />
              <YAxis domain={[0, 1]} tick={{ fill: '#5a5f70', fontSize: 11 }} />
              <Tooltip
                content={({ active, payload, label }) => {
                  if (!active || !payload) return null;
                  return (
                    <div className="custom-tooltip">
                      <div className="label">{label}</div>
                      {payload.map((p, i) => (
                        <div key={i} style={{ display: 'flex', gap: 8, marginTop: 2 }}>
                          <span style={{ color: p.color, fontWeight: 500 }}>{humanize(p.dataKey as string)}</span>
                          <span className="value">{(p.value as number)?.toFixed(4)}</span>
                        </div>
                      ))}
                    </div>
                  );
                }}
              />
              {modelKeys.map((key, i) => (
                <Bar key={key} dataKey={key} fill={CHART_COLORS_HEX[i]} barSize={20} radius={[4, 4, 0, 0]} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Radar Chart */}
        <div className="card">
          <div className="card-header">
            <span className="card-title">Model Profile</span>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="var(--border-subtle)" />
              <PolarAngleAxis dataKey="metric" tick={{ fill: '#8b90a0', fontSize: 10 }} />
              <PolarRadiusAxis domain={[0, 1]} tick={{ fill: '#5a5f70', fontSize: 10 }} />
              {modelKeys.map((key, i) => (
                <Radar
                  key={key}
                  name={humanize(key)}
                  dataKey={key}
                  stroke={CHART_COLORS_HEX[i]}
                  fill={CHART_COLORS_HEX[i]}
                  fillOpacity={0.1}
                  strokeWidth={2}
                />
              ))}
              <Tooltip
                content={({ active, payload, label }) => {
                  if (!active || !payload) return null;
                  return (
                    <div className="custom-tooltip">
                      <div className="label">{label}</div>
                      {payload.map((p, i) => (
                        <div key={i} style={{ display: 'flex', gap: 8, marginTop: 2 }}>
                          <span style={{ color: p.color, fontWeight: 500 }}>{p.name}</span>
                          <span className="value">{(p.value as number)?.toFixed(4)}</span>
                        </div>
                      ))}
                    </div>
                  );
                }}
              />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Full comparison table */}
      <div className="card mt-xl">
        <div className="card-header">
          <span className="card-title">Full Metrics Table</span>
        </div>
        <div className="table-wrapper">
          <table className="data-table">
            <thead>
              <tr>
                <th>Model</th>
                {Object.keys(METRIC_LABELS).map(m => (
                  <th key={m}>{METRIC_LABELS[m]}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {modelKeys.map(key => {
                const m = models[key];
                const isSelected = key === selected;
                return (
                  <tr key={key} style={isSelected ? { background: 'var(--accent-muted)' } : undefined}>
                    <td style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                      {m.name}
                      {isSelected && <span style={{ marginLeft: 8, fontSize: 10, color: 'var(--accent)' }}>SELECTED</span>}
                    </td>
                    {Object.keys(METRIC_LABELS).map(metric => {
                      const val = m[metric as keyof ModelMetrics];
                      return (
                        <td key={metric} className="num">
                          {typeof val === 'number' ? val.toFixed(4) : '—'}
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}

/* --- Evaluation Details --- */
function EvaluationDetails({ data }: { data: Record<string, unknown> }) {
  const testMetrics = (data.test_metrics ?? {}) as Record<string, number>;
  const calibration = (data.calibration ?? {}) as Record<string, unknown>;
  const confusion = testMetrics.confusion_matrix as unknown;

  return (
    <>
      {/* Test Set Metrics */}
      <div className="card mb-lg">
        <div className="card-header">
          <span className="card-title">Test Set Performance</span>
          <span className="card-subtitle">Final evaluation on held-out test data</span>
        </div>
        <div className="grid grid-4">
          {Object.entries(testMetrics)
            .filter(([key]) => typeof testMetrics[key] === 'number' && key !== 'confusion_matrix')
            .slice(0, 8)
            .map(([key, val]) => (
              <div key={key} style={{ padding: 'var(--space-sm)' }}>
                <div className="kpi-label">{METRIC_LABELS[key] || humanize(key)}</div>
                <div className="kpi-value mono" style={{ fontSize: 20 }}>
                  {typeof val === 'number' ? (key === 'brier_score' ? val.toFixed(4) : val.toFixed(4)) : '—'}
                </div>
              </div>
            ))}
        </div>
      </div>

      {/* Calibration */}
      {calibration && Object.keys(calibration).length > 0 && (
        <div className="card mb-lg">
          <div className="card-header">
            <span className="card-title">Calibration Assessment</span>
          </div>
          <div className="grid grid-3">
            {Object.entries(calibration)
              .filter(([, val]) => typeof val === 'number' || typeof val === 'boolean' || typeof val === 'string')
              .map(([key, val]) => (
                <div key={key} style={{ padding: 'var(--space-sm)' }}>
                  <div className="kpi-label">{humanize(key)}</div>
                  <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-primary)', marginTop: 4 }}>
                    {typeof val === 'boolean'
                      ? (val ? '✓ Yes' : '✗ No')
                      : typeof val === 'number'
                        ? val.toFixed(4)
                        : String(val)}
                  </div>
                </div>
              ))}
          </div>
        </div>
      )}

      {/* Confusion Matrix */}
      {confusion && Array.isArray(confusion) && (
        <div className="card">
          <div className="card-header">
            <span className="card-title">Confusion Matrix</span>
            <span className="card-subtitle">Test set predictions vs actual outcomes</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'center', padding: 'var(--space-xl)' }}>
            <div className="confusion-matrix">
              <div className="cm-header" />
              <div className="cm-header">Predicted 0</div>
              <div className="cm-header">Predicted 1</div>
              <div className="cm-header" style={{ writingMode: 'vertical-rl', transform: 'rotate(180deg)' }}>Actual 0</div>
              <div className="cm-cell tn">{(confusion as number[][])[0]?.[0] ?? '—'}</div>
              <div className="cm-cell fp">{(confusion as number[][])[0]?.[1] ?? '—'}</div>
              <div className="cm-header" style={{ writingMode: 'vertical-rl', transform: 'rotate(180deg)' }}>Actual 1</div>
              <div className="cm-cell fn">{(confusion as number[][])[1]?.[0] ?? '—'}</div>
              <div className="cm-cell tp">{(confusion as number[][])[1]?.[1] ?? '—'}</div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
