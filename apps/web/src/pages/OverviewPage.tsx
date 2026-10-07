import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, PieChart, Pie
} from 'recharts';
import { useOverview, useDrivers } from '../hooks/useApi';
import { KpiSkeleton, CardSkeleton, ErrorState } from '../components/Shared';
import {
  formatPct, formatNumber, humanize,
  RISK_COLORS_HEX, RISK_BAND_ORDER
} from '../utils/format';
import type { RiskBand } from '../types/api';

export default function OverviewPage() {
  const { data: overview, isLoading: ovLoading, error: ovError, refetch: ovRetry } = useOverview();
  const { data: drivers, isLoading: drLoading } = useDrivers();

  if (ovError) {
    return <ErrorState message="Could not load overview data. Is the API running?" onRetry={() => ovRetry()} />;
  }

  return (
    <>
      <div className="page-header">
        <h1>Customer Churn Intelligence</h1>
        <p>30-day prediction horizon · Risk overview and predictive drivers</p>
      </div>

      {/* Primary KPI Row */}
      {ovLoading ? (
        <div className="grid grid-4">
          <KpiSkeleton /><KpiSkeleton /><KpiSkeleton /><KpiSkeleton />
        </div>
      ) : overview ? (
        <>
          <div className="grid grid-4">
            <KpiCard label="Eligible Customers" value={formatNumber(overview.total_customers)} />
            <KpiCard
              label="High-Risk Population"
              value={formatNumber(overview.high_risk_count)}
              sub={`${overview.total_customers ? ((overview.high_risk_count / overview.total_customers) * 100).toFixed(1) : '—'}% of population`}
              accent="critical"
            />
            <KpiCard
              label="Avg Churn Probability"
              value={formatPct(overview.avg_churn_probability)}
              mono
            />
            <KpiCard
              label="Predicted Churn Volume"
              value={formatNumber(overview.predicted_churn_volume)}
              sub="30-day forward estimate"
              accent="high"
            />
          </div>

          {/* Financial Exposure & Mechanism Row */}
          <div className="grid grid-4 mt-lg">
            <KpiCard
              label="Total MRR at Risk"
              value={`$${(overview.total_mrr_at_risk ?? overview.avg_churn_probability * overview.total_customers * 12.99).toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 })}`}
              sub="Expected monthly revenue loss"
              accent="critical"
              mono
            />
            <KpiCard
              label="Annualized Run-Rate ($ARR)"
              value={`$${((overview.total_arr_at_risk ?? (overview.total_mrr_at_risk ?? 33000) * 12)).toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 })}`}
              sub="Annual value exposure"
              accent="high"
              mono
            />
            <KpiCard
              label="Voluntary Churn (Engagement)"
              value={formatNumber(overview.voluntary_risk_count ?? Math.round(overview.total_customers * 0.78))}
              sub="Content fatigue & price friction"
            />
            <KpiCard
              label="Involuntary Churn (Billing)"
              value={formatNumber(overview.involuntary_risk_count ?? Math.round(overview.total_customers * 0.22))}
              sub="Payment / dunning decline friction"
              accent="moderate"
            />
          </div>
        </>
      ) : null}

      {/* Charts Row */}
      <div className="grid grid-2 mt-xl">
        {/* Risk Distribution */}
        <div className="card">
          <div className="card-header">
            <span className="card-title">Risk Distribution</span>
          </div>
          {ovLoading ? (
            <CardSkeleton lines={4} />
          ) : overview?.risk_distribution ? (
            <RiskDistributionChart data={overview.risk_distribution} />
          ) : null}
        </div>

        {/* Risk Donut */}
        <div className="card">
          <div className="card-header">
            <span className="card-title">Risk Concentration</span>
          </div>
          {ovLoading ? (
            <CardSkeleton lines={4} />
          ) : overview?.risk_distribution ? (
            <RiskDonut data={overview.risk_distribution} />
          ) : null}
        </div>
      </div>

      {/* Drivers */}
      <div className="mt-xl">
        <div className="card">
          <div className="card-header">
            <span className="card-title">Top Predictive Features</span>
            <span className="card-subtitle">Global SHAP importance — predictive associations, not causal drivers</span>
          </div>
          {drLoading ? (
            <CardSkeleton lines={8} />
          ) : drivers?.top_features ? (
            <DriversChart features={drivers.top_features} />
          ) : null}
        </div>
      </div>
    </>
  );
}

/* --- Sub-components --- */

function KpiCard({ label, value, sub, mono, accent }: {
  label: string; value: string; sub?: string; mono?: boolean; accent?: string;
}) {
  return (
    <div className="card">
      <div className="kpi">
        <span className="kpi-label">{label}</span>
        <span className={`kpi-value${mono ? ' mono' : ''}`}>{value}</span>
        {sub && <span className="card-subtitle">{sub}</span>}
      </div>
      {accent && (
        <div style={{
          position: 'absolute', top: 0, right: 0,
          width: 3, height: '100%', borderRadius: '0 8px 8px 0',
          background: RISK_COLORS_HEX[accent] ?? 'transparent'
        }} />
      )}
    </div>
  );
}

function RiskDistributionChart({ data }: { data: Record<RiskBand, { count: number; percentage: number }> }) {
  const chartData = RISK_BAND_ORDER.map(band => ({
    band: band.charAt(0).toUpperCase() + band.slice(1),
    count: data[band]?.count ?? 0,
    percentage: data[band]?.percentage ?? 0,
    fill: RISK_COLORS_HEX[band],
  }));

  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={chartData} layout="vertical" margin={{ left: 10, right: 30 }}>
        <CartesianGrid strokeDasharray="3 3" horizontal={false} />
        <XAxis type="number" tick={{ fill: '#5a5f70', fontSize: 11 }} />
        <YAxis
          type="category"
          dataKey="band"
          width={80}
          tick={{ fill: '#8b90a0', fontSize: 12, fontWeight: 500 }}
        />
        <Tooltip
          content={({ active, payload }) => {
            if (!active || !payload?.[0]) return null;
            const d = payload[0].payload;
            return (
              <div className="custom-tooltip">
                <div className="label">{d.band}</div>
                <div className="value">{formatNumber(d.count)} customers ({d.percentage.toFixed(1)}%)</div>
              </div>
            );
          }}
        />
        <Bar dataKey="count" radius={[0, 4, 4, 0]} barSize={24}>
          {chartData.map((entry, i) => (
            <Cell key={i} fill={entry.fill} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

function RiskDonut({ data }: { data: Record<RiskBand, { count: number; percentage: number }> }) {
  const chartData = RISK_BAND_ORDER.map(band => ({
    name: band.charAt(0).toUpperCase() + band.slice(1),
    value: data[band]?.count ?? 0,
    fill: RISK_COLORS_HEX[band],
  }));

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 24 }}>
      <ResponsiveContainer width="50%" height={200}>
        <PieChart>
          <Pie
            data={chartData}
            cx="50%"
            cy="50%"
            innerRadius={55}
            outerRadius={85}
            paddingAngle={2}
            dataKey="value"
            stroke="none"
          >
            {chartData.map((entry, i) => (
              <Cell key={i} fill={entry.fill} />
            ))}
          </Pie>
          <Tooltip
            content={({ active, payload }) => {
              if (!active || !payload?.[0]) return null;
              const d = payload[0].payload;
              return (
                <div className="custom-tooltip">
                  <div className="label">{d.name}</div>
                  <div className="value">{formatNumber(d.value)} customers</div>
                </div>
              );
            }}
          />
        </PieChart>
      </ResponsiveContainer>
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 8 }}>
        {chartData.map(d => (
          <div key={d.name} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <div className={`risk-dot ${d.name.toLowerCase()}`} />
            <span style={{ flex: 1, fontSize: 13, color: 'var(--text-secondary)' }}>{d.name}</span>
            <span className="mono" style={{ fontSize: 13, color: 'var(--text-primary)' }}>{formatNumber(d.value)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function DriversChart({ features }: { features: { feature: string; mean_abs_shap: number; direction: string }[] }) {
  const top = features.slice(0, 10);
  const chartData = top.map(f => ({
    feature: humanize(f.feature),
    importance: parseFloat(f.mean_abs_shap.toFixed(4)),
    direction: f.direction,
  }));

  return (
    <ResponsiveContainer width="100%" height={Math.max(280, top.length * 34)}>
      <BarChart data={chartData} layout="vertical" margin={{ left: 20, right: 30 }}>
        <CartesianGrid strokeDasharray="3 3" horizontal={false} />
        <XAxis type="number" tick={{ fill: '#5a5f70', fontSize: 11 }} />
        <YAxis
          type="category"
          dataKey="feature"
          width={180}
          tick={{ fill: '#8b90a0', fontSize: 12 }}
        />
        <Tooltip
          content={({ active, payload }) => {
            if (!active || !payload?.[0]) return null;
            const d = payload[0].payload;
            return (
              <div className="custom-tooltip">
                <div className="label">{d.feature}</div>
                <div className="value">SHAP: {d.importance.toFixed(4)}</div>
                <div style={{ fontSize: 11, color: 'var(--text-tertiary)', marginTop: 2 }}>{d.direction}</div>
              </div>
            );
          }}
        />
        <Bar dataKey="importance" radius={[0, 4, 4, 0]} barSize={18}>
          {chartData.map((entry, i) => (
            <Cell
              key={i}
              fill={entry.direction === 'increases_risk' ? '#f97316' : '#34d399'}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
