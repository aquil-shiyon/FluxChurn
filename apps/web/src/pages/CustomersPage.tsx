import { useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell
} from 'recharts';
import { useCustomers, useCustomerRisk, useCustomerExplanation } from '../hooks/useApi';
import { CardSkeleton, ErrorState, EmptyState } from '../components/Shared';
import { formatPct, formatNumber, humanize, riskColor } from '../utils/format';
import type { CustomerSummary, ExplanationFeature } from '../types/api';

export default function CustomersPage() {
  const [search, setSearch] = useState('');
  const [riskFilter, setRiskFilter] = useState<string>('');
  const [churnTypeFilter, setChurnTypeFilter] = useState<string>('');
  const [sortBy, setSortBy] = useState<string>('mrr_at_risk');
  const [selectedId, setSelectedId] = useState<string>('');
  const [page, setPage] = useState(0);

  const limit = 25;
  const { data, isLoading, error, refetch } = useCustomers({
    search: search || undefined,
    risk_band: riskFilter || undefined,
    churn_type: churnTypeFilter || undefined,
    sort_by: sortBy,
    order: 'desc',
    limit,
    offset: page * limit,
  });

  if (error) {
    return <ErrorState message="Could not load customer data." onRetry={() => refetch()} />;
  }

  return (
    <div style={{ display: 'flex', gap: 'var(--space-xl)', height: '100%' }}>
      {/* Left Panel — Customer List */}
      <div style={{ flex: selectedId ? '0 0 460px' : 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
        <div className="page-header">
          <h1>Customer Intelligence & Financial Prioritization</h1>
          <p>Investigate individual subscriber exposure, churn mechanisms, and local SHAP explanations</p>
        </div>

        {/* Filters & Financial Sorters */}
        <div className="flex gap-md mb-lg" style={{ flexWrap: 'wrap', alignItems: 'center' }}>
          <div className="input-group" style={{ maxWidth: 220 }}>
            <svg className="input-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
              <circle cx="11" cy="11" r="8" /><path d="M21 21l-4.35-4.35" />
            </svg>
            <input
              className="input"
              placeholder="Search customer ID..."
              value={search}
              onChange={e => { setSearch(e.target.value); setPage(0); }}
            />
          </div>

          <select
            className="select"
            value={riskFilter}
            onChange={e => { setRiskFilter(e.target.value); setPage(0); }}
          >
            <option value="">All risk bands</option>
            <option value="critical">Critical Risk</option>
            <option value="high">High Risk</option>
            <option value="moderate">Moderate Risk</option>
            <option value="low">Low Risk</option>
          </select>

          <select
            className="select"
            value={churnTypeFilter}
            onChange={e => { setChurnTypeFilter(e.target.value); setPage(0); }}
          >
            <option value="">All Mechanisms</option>
            <option value="voluntary_engagement">Voluntary (Engagement)</option>
            <option value="involuntary_billing">Involuntary (Billing)</option>
          </select>

          <select
            className="select"
            value={sortBy}
            onChange={e => { setSortBy(e.target.value); setPage(0); }}
            style={{ fontWeight: 600, color: 'var(--accent)' }}
          >
            <option value="mrr_at_risk">Sort: Highest $MRR at Risk</option>
            <option value="churn_probability">Sort: Highest Churn Probability</option>
          </select>
        </div>

        {/* Table */}
        {isLoading ? (
          <CardSkeleton lines={12} />
        ) : !data?.customers?.length ? (
          <EmptyState title="No customers found" description="Try adjusting your search or filters." />
        ) : (
          <>
            <div className="card" style={{ overflow: 'auto', flex: 1 }}>
              <div className="table-wrapper">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Customer ID</th>
                      <th>Risk %</th>
                      <th>$MRR at Risk</th>
                      <th>Mechanism</th>
                      <th>Risk Tier</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.customers.map((c: CustomerSummary) => (
                      <tr
                        key={c.customer_id}
                        className="clickable"
                        onClick={() => setSelectedId(c.customer_id)}
                        style={c.customer_id === selectedId ? { background: 'var(--accent-muted)' } : undefined}
                      >
                        <td style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                          {c.customer_id}
                        </td>
                        <td className="num font-mono">{formatPct(c.churn_probability)}</td>
                        <td className="num font-mono" style={{ color: 'var(--risk-critical)', fontWeight: 600 }}>
                          ${(c.mrr_at_risk ?? (c.churn_probability * 12.99)).toFixed(2)}/mo
                        </td>
                        <td>
                          <span style={{
                            fontSize: 10,
                            padding: '2px 6px',
                            borderRadius: 4,
                            background: c.churn_type_risk === 'involuntary_billing' ? 'rgba(251, 191, 36, 0.15)' : 'rgba(99, 145, 255, 0.15)',
                            color: c.churn_type_risk === 'involuntary_billing' ? 'var(--warning)' : 'var(--accent)',
                            fontWeight: 500,
                          }}>
                            {c.churn_type_risk === 'involuntary_billing' ? 'Billing' : 'Voluntary'}
                          </span>
                        </td>
                        <td>
                          <span className={`risk-badge ${c.risk_band}`}>{c.risk_band}</span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Pagination */}
            <div className="flex items-center justify-between mt-lg" style={{ padding: '0 4px' }}>
              <span style={{ fontSize: 12, color: 'var(--text-tertiary)' }}>
                {formatNumber(data.total)} total · Showing {page * limit + 1}–{Math.min((page + 1) * limit, data.total)}
              </span>
              <div className="flex gap-sm">
                <button
                  className="btn btn-ghost btn-sm"
                  disabled={page === 0}
                  onClick={() => setPage(p => p - 1)}
                >
                  Previous
                </button>
                <button
                  className="btn btn-ghost btn-sm"
                  disabled={(page + 1) * limit >= data.total}
                  onClick={() => setPage(p => p + 1)}
                >
                  Next
                </button>
              </div>
            </div>
          </>
        )}
      </div>

      {/* Right Panel — Customer Detail */}
      {selectedId && (
        <CustomerDetailPanel
          customerId={selectedId}
          onClose={() => setSelectedId('')}
        />
      )}
    </div>
  );
}

/* --- Customer Detail Panel --- */
function CustomerDetailPanel({ customerId, onClose }: { customerId: string; onClose: () => void }) {
  const { data: risk, isLoading: riskLoading } = useCustomerRisk(customerId);
  const { data: explanation, isLoading: expLoading } = useCustomerExplanation(customerId);

  return (
    <div style={{
      flex: 1,
      background: 'var(--bg-card)',
      border: '1px solid var(--border-subtle)',
      borderRadius: 'var(--radius-lg)',
      padding: 'var(--space-xl)',
      overflow: 'auto',
      minWidth: 0,
    }}>
      {/* Header */}
      <div className="flex items-center justify-between mb-lg">
        <div>
          <div style={{ fontSize: 11, color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.8px', fontWeight: 600 }}>
            Subscriber Profile & Exposure
          </div>
          <h2 style={{ fontSize: 18, fontWeight: 700, marginTop: 4 }}>{customerId}</h2>
        </div>
        <button className="btn btn-ghost btn-sm" onClick={onClose}>✕ Close</button>
      </div>

      {riskLoading ? (
        <CardSkeleton lines={6} />
      ) : risk && !risk.error ? (
        <>
          {/* Financial Risk Header */}
          <div className="card" style={{ marginBottom: 'var(--space-lg)' }}>
            <div className="grid grid-3" style={{ gap: 'var(--space-md)' }}>
              <div style={{ textAlign: 'center', padding: 'var(--space-sm)' }}>
                <div style={{
                  fontSize: 32, fontWeight: 700, fontFamily: 'var(--font-mono)',
                  color: riskColor(risk.risk_band), lineHeight: 1.1,
                }}>
                  {formatPct(risk.churn_probability)}
                </div>
                <div style={{ fontSize: 11, color: 'var(--text-tertiary)', marginTop: 4 }}>30-Day Probability</div>
                <span className={`risk-badge ${risk.risk_band} mt-2`} style={{ fontSize: 11 }}>
                  {risk.risk_band.toUpperCase()}
                </span>
              </div>

              <div style={{ textAlign: 'center', padding: 'var(--space-sm)', borderLeft: '1px solid var(--border-subtle)', borderRight: '1px solid var(--border-subtle)' }}>
                <div style={{ fontSize: 26, fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--danger)', lineHeight: 1.1 }}>
                  ${(risk.mrr_at_risk ?? (risk.churn_probability * (typeof risk.features?.monthly_fee === 'number' ? risk.features.monthly_fee : 12.99))).toFixed(2)}
                </div>
                <div style={{ fontSize: 11, color: 'var(--text-tertiary)', marginTop: 4 }}>Monthly MRR at Risk</div>
                <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 4 }}>
                  ${((risk.arr_at_risk ?? (risk.mrr_at_risk ?? 10) * 12)).toFixed(0)} Annualized ($ARR)
                </div>
              </div>

              <div style={{ textAlign: 'center', padding: 'var(--space-sm)' }}>
                <div style={{ fontSize: 13, fontWeight: 600, marginTop: 2, color: risk.churn_type_risk === 'involuntary_billing' ? 'var(--warning)' : 'var(--accent)' }}>
                  {risk.churn_type_risk === 'involuntary_billing' ? '💳 Involuntary Billing Friction' : '📉 Voluntary Engagement Decline'}
                </div>
                <div style={{ fontSize: 11, color: 'var(--text-tertiary)', marginTop: 4 }}>Primary Mechanism</div>
                <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 4 }}>
                  Plan Fee: ${typeof risk.features?.monthly_fee === 'number' ? risk.features.monthly_fee.toFixed(2) : '12.99'}/mo
                </div>
              </div>
            </div>
          </div>

          {/* Features Grid */}
          <div style={{ marginBottom: 'var(--space-lg)' }}>
            <div className="card-title" style={{ marginBottom: 'var(--space-md)' }}>Subscriber Attributes</div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 'var(--space-sm)' }}>
              {Object.entries(risk.features || {}).map(([key, val]) => (
                <div key={key} style={{
                  padding: 'var(--space-sm) var(--space-md)',
                  background: 'var(--bg-tertiary)',
                  borderRadius: 'var(--radius-sm)',
                }}>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                    {humanize(key)}
                  </div>
                  <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)', marginTop: 2 }}>
                    {typeof val === 'number' ? (val % 1 ? val.toFixed(2) : val) : String(val ?? '')}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </>
      ) : (
        <EmptyState title="Customer not found" />
      )}

      {/* Explanation */}
      {expLoading ? (
        <CardSkeleton lines={8} />
      ) : explanation?.top_contributors ? (
        <div>
          <div className="card-title" style={{ marginBottom: 4 }}>Why This Risk Score? (SHAP Waterfall)</div>
          <div className="card-subtitle" style={{ marginBottom: 'var(--space-md)' }}>
            Feature impact relative to baseline probability of {formatPct(explanation.base_value)}
          </div>
          <div style={{ width: '100%', height: 220 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={explanation.top_contributors}
                layout="vertical"
                margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" horizontal={false} />
                <XAxis type="number" stroke="var(--text-muted)" fontSize={11} />
                <YAxis
                  type="category"
                  dataKey="feature"
                  stroke="var(--text-muted)"
                  fontSize={11}
                  tickFormatter={humanize}
                  width={100}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'var(--bg-card)',
                    borderColor: 'var(--border-default)',
                    borderRadius: 'var(--radius-md)',
                    color: 'var(--text-primary)',
                    fontSize: 12,
                  }}
                  formatter={(val: unknown) => [typeof val === 'number' ? val.toFixed(4) : String(val), 'SHAP Impact']}
                />
                <Bar dataKey="contribution" radius={[0, 4, 4, 0]}>
                  {explanation.top_contributors.map((entry: ExplanationFeature, idx: number) => (
                    <Cell
                      key={`exp-${idx}`}
                      fill={entry.contribution > 0 ? 'var(--risk-critical)' : 'var(--risk-low)'}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      ) : null}
    </div>
  );
}
