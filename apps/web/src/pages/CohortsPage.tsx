import { useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, Legend
} from 'recharts';
import { useCohorts } from '../hooks/useApi';
import { CardSkeleton, ErrorState, EmptyState } from '../components/Shared';
import { formatPct, formatNumber, humanize, RISK_COLORS_HEX, RISK_BAND_ORDER } from '../utils/format';
import type { CohortInfo } from '../types/api';

const DIMENSIONS = [
  { value: 'subscription_type', label: 'Subscription Plan' },
  { value: 'region', label: 'Region' },
  { value: 'device', label: 'Device' },
  { value: 'payment_method', label: 'Payment Method' },
  { value: 'gender', label: 'Gender' },
  { value: 'favorite_genre', label: 'Favorite Genre' },
];

export default function CohortsPage() {
  const [dimension, setDimension] = useState('subscription_type');
  const { data, isLoading, error, refetch } = useCohorts(dimension);

  if (error) {
    return <ErrorState message="Could not load cohort data." onRetry={() => refetch()} />;
  }

  return (
    <>
      <div className="page-header">
        <h1>Cohort Explorer</h1>
        <p>Analyze churn risk across customer segments and identify elevated-risk groups</p>
      </div>

      {/* Controls */}
      <div className="flex items-center gap-lg mb-lg">
        <span style={{ fontSize: 12, color: 'var(--text-tertiary)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px' }}>
          Group by
        </span>
        <div className="flex gap-sm" style={{ flexWrap: 'wrap' }}>
          {DIMENSIONS.map(d => (
            <button
              key={d.value}
              className={`btn btn-ghost btn-sm${dimension === d.value ? ' active' : ''}`}
              onClick={() => setDimension(d.value)}
            >
              {d.label}
            </button>
          ))}
        </div>
      </div>

      {isLoading ? (
        <div className="grid grid-2">
          <CardSkeleton lines={8} /><CardSkeleton lines={8} />
        </div>
      ) : !data?.cohorts || Object.keys(data.cohorts).length === 0 ? (
        <EmptyState title="No cohort data available" />
      ) : (
        <>
          {/* Stacked Bar Chart */}
          <div className="card mb-lg">
            <div className="card-header">
              <span className="card-title">Risk Distribution by {humanize(dimension)}</span>
            </div>
            <CohortStackedChart cohorts={data.cohorts} />
          </div>

          {/* Avg Probability Bar Chart */}
          <div className="card mb-lg">
            <div className="card-header">
              <span className="card-title">Average Churn Probability by {humanize(dimension)}</span>
            </div>
            <CohortProbabilityChart cohorts={data.cohorts} />
          </div>

          {/* Cohort Table */}
          <div className="card">
            <div className="card-header">
              <span className="card-title">Cohort Summary</span>
              <span className="card-subtitle">{Object.keys(data.cohorts).length} groups</span>
            </div>
            <div className="table-wrapper">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>{humanize(dimension)}</th>
                    <th>Customers</th>
                    <th>Avg Prob</th>
                    {RISK_BAND_ORDER.map(band => (
                      <th key={band} style={{ textAlign: 'center' }}>
                        <span style={{ color: RISK_COLORS_HEX[band] }}>
                          {band.charAt(0).toUpperCase() + band.slice(1)}
                        </span>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(data.cohorts)
                    .sort((a, b) => (b[1] as CohortInfo).avg_churn_probability - (a[1] as CohortInfo).avg_churn_probability)
                    .map(([name, info]) => {
                      const cohort = info as CohortInfo;
                      return (
                        <tr key={name}>
                          <td style={{ fontWeight: 500, color: 'var(--text-primary)' }}>{name}</td>
                          <td className="num">{formatNumber(cohort.total)}</td>
                          <td className="num">{formatPct(cohort.avg_churn_probability)}</td>
                          {RISK_BAND_ORDER.map(band => (
                            <td key={band} className="num" style={{ textAlign: 'center' }}>
                              {cohort.risk_distribution[band]?.count ?? 0}
                              <span style={{ color: 'var(--text-muted)', marginLeft: 4, fontSize: 11 }}>
                                ({cohort.risk_distribution[band]?.percentage?.toFixed(1) ?? 0}%)
                              </span>
                            </td>
                          ))}
                        </tr>
                      );
                    })}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </>
  );
}

/* --- Sub-components --- */

function CohortStackedChart({ cohorts }: { cohorts: Record<string, CohortInfo> }) {
  const chartData = Object.entries(cohorts).map(([name, info]) => {
    const row: Record<string, unknown> = { name };
    RISK_BAND_ORDER.forEach(band => {
      row[band] = info.risk_distribution[band]?.percentage ?? 0;
    });
    return row;
  });

  return (
    <ResponsiveContainer width="100%" height={Math.max(240, chartData.length * 46)}>
      <BarChart data={chartData} layout="vertical" margin={{ left: 20, right: 20 }}>
        <CartesianGrid strokeDasharray="3 3" horizontal={false} />
        <XAxis type="number" domain={[0, 100]} tick={{ fill: '#5a5f70', fontSize: 11 }} tickFormatter={v => `${v}%`} />
        <YAxis type="category" dataKey="name" width={130} tick={{ fill: '#8b90a0', fontSize: 12 }} />
        <Tooltip
          content={({ active, payload, label }) => {
            if (!active || !payload) return null;
            return (
              <div className="custom-tooltip">
                <div className="label">{label}</div>
                {payload.map((p, i) => (
                  <div key={i} style={{ display: 'flex', gap: 8, marginTop: 2 }}>
                    <span style={{ color: p.color, fontWeight: 500 }}>{humanize(p.dataKey as string)}</span>
                    <span className="value">{(p.value as number)?.toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            );
          }}
        />
        <Legend
          formatter={(value: string) => <span style={{ fontSize: 11, color: 'var(--text-secondary)' }}>{humanize(value)}</span>}
        />
        {RISK_BAND_ORDER.map(band => (
          <Bar key={band} dataKey={band} stackId="risk" fill={RISK_COLORS_HEX[band]} barSize={22} />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}

function CohortProbabilityChart({ cohorts }: { cohorts: Record<string, CohortInfo> }) {
  const chartData = Object.entries(cohorts)
    .map(([name, info]) => ({
      name,
      probability: info.avg_churn_probability,
    }))
    .sort((a, b) => b.probability - a.probability);

  return (
    <ResponsiveContainer width="100%" height={Math.max(200, chartData.length * 38)}>
      <BarChart data={chartData} layout="vertical" margin={{ left: 20, right: 30 }}>
        <CartesianGrid strokeDasharray="3 3" horizontal={false} />
        <XAxis type="number" domain={[0, 1]} tick={{ fill: '#5a5f70', fontSize: 11 }} tickFormatter={v => formatPct(v as number, 0)} />
        <YAxis type="category" dataKey="name" width={130} tick={{ fill: '#8b90a0', fontSize: 12 }} />
        <Tooltip
          content={({ active, payload }) => {
            if (!active || !payload?.[0]) return null;
            return (
              <div className="custom-tooltip">
                <div className="label">{payload[0].payload.name}</div>
                <div className="value">{formatPct(payload[0].payload.probability)}</div>
              </div>
            );
          }}
        />
        <Bar dataKey="probability" radius={[0, 4, 4, 0]} barSize={20}>
          {chartData.map((entry, i) => (
            <Cell key={i} fill={entry.probability >= 0.5 ? '#f97316' : '#6391ff'} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
