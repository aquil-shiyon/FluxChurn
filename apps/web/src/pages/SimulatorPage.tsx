import { useState } from 'react';
import { useCustomers, useCustomerRisk, useSimulate } from '../hooks/useApi';
import { CardSkeleton, EmptyState, Disclaimer } from '../components/Shared';
import { formatPct, riskColor } from '../utils/format';
import type { SimulationResult, CustomerSummary } from '../types/api';

const SIMULATABLE = [
  { key: 'watch_hours', label: 'Watch Hours', min: 0, max: 120, step: 1, type: 'continuous' },
  { key: 'last_login_days', label: 'Days Since Last Login', min: 0, max: 60, step: 1, type: 'integer' },
  { key: 'avg_watch_time_per_day', label: 'Avg Watch Time (hrs/day)', min: 0, max: 24, step: 0.5, type: 'continuous' },
  { key: 'number_of_profiles', label: 'Number of Profiles', min: 1, max: 5, step: 1, type: 'integer' },
];

const SUBSCRIPTION_OPTIONS = ['Basic', 'Standard', 'Premium'];

export default function SimulatorPage() {
  const [selectedId, setSelectedId] = useState('');
  const [search, setSearch] = useState('');
  const [scenario, setScenario] = useState<Record<string, number | string>>({});
  const [result, setResult] = useState<SimulationResult | null>(null);

  const { data: customers, isLoading: listLoading } = useCustomers({
    search: search || undefined,
    limit: 15,
    offset: 0,
  });
  const { data: risk, isLoading: riskLoading } = useCustomerRisk(selectedId);
  const simulate = useSimulate();

  const handleSelect = (id: string) => {
    setSelectedId(id);
    setScenario({});
    setResult(null);
  };

  const handleSimulate = () => {
    if (!selectedId || Object.keys(scenario).length === 0) return;
    simulate.mutate(
      { customerId: selectedId, scenario },
      { onSuccess: (data) => setResult(data) },
    );
  };

  const currentFeatures = risk?.features ?? {};

  return (
    <>
      <div className="page-header">
        <h1>Risk Simulator</h1>
        <p>Explore model sensitivity to hypothetical behavioral changes</p>
      </div>

      <Disclaimer>
        Scenario output represents model sensitivity, not a causal estimate of how
        changing customer behavior will affect churn.
      </Disclaimer>

      <div style={{ display: 'flex', gap: 'var(--space-xl)', marginTop: 'var(--space-xl)' }}>
        {/* Customer Selector */}
        <div style={{ width: 280, flexShrink: 0 }}>
          <div className="card-title" style={{ marginBottom: 'var(--space-md)' }}>Select Customer</div>
          <div className="input-group" style={{ marginBottom: 'var(--space-md)' }}>
            <svg className="input-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
              <circle cx="11" cy="11" r="8" /><path d="M21 21l-4.35-4.35" />
            </svg>
            <input
              className="input"
              placeholder="Search customer..."
              value={search}
              onChange={e => setSearch(e.target.value)}
            />
          </div>

          {listLoading ? (
            <CardSkeleton lines={6} />
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 2, maxHeight: 400, overflow: 'auto' }}>
              {customers?.customers?.map((c: CustomerSummary) => (
                <div
                  key={c.customer_id}
                  onClick={() => handleSelect(c.customer_id)}
                  style={{
                    padding: 'var(--space-sm) var(--space-md)',
                    borderRadius: 'var(--radius-md)',
                    cursor: 'pointer',
                    background: c.customer_id === selectedId ? 'var(--accent-muted)' : 'transparent',
                    border: c.customer_id === selectedId ? '1px solid var(--border-accent)' : '1px solid transparent',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    transition: 'all 0.15s',
                  }}
                >
                  <span style={{ fontSize: 13, fontWeight: 500, color: 'var(--text-primary)' }}>{c.customer_id}</span>
                  <span className={`risk-badge ${c.risk_band}`} style={{ fontSize: 10 }}>{c.risk_band}</span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Simulator */}
        <div style={{ flex: 1 }}>
          {!selectedId ? (
            <EmptyState title="Select a customer" description="Choose a customer from the list to start a simulation." />
          ) : riskLoading ? (
            <CardSkeleton lines={10} />
          ) : (
            <div className="grid grid-2">
              {/* Current State */}
              <div className="card">
                <div className="card-header">
                  <span className="card-title">Current State</span>
                  <span className={`risk-badge ${risk?.risk_band}`}>{risk?.risk_band}</span>
                </div>
                <div style={{
                  fontSize: 32, fontWeight: 700, fontFamily: 'var(--font-mono)',
                  color: riskColor(risk?.risk_band ?? 'low'), marginBottom: 'var(--space-lg)',
                }}>
                  {formatPct(risk?.churn_probability ?? 0)}
                </div>

                {SIMULATABLE.map(f => (
                  <div key={f.key} style={{
                    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                    padding: 'var(--space-sm) 0',
                    borderBottom: '1px solid var(--border-subtle)',
                    fontSize: 13,
                  }}>
                    <span style={{ color: 'var(--text-secondary)' }}>{f.label}</span>
                    <span className="mono" style={{ fontWeight: 600 }}>
                      {currentFeatures[f.key] != null
                        ? (typeof currentFeatures[f.key] === 'number' && (currentFeatures[f.key] as number) % 1
                          ? (currentFeatures[f.key] as number).toFixed(1)
                          : currentFeatures[f.key])
                        : '—'}
                    </span>
                  </div>
                ))}

                <div style={{
                  display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                  padding: 'var(--space-sm) 0', fontSize: 13,
                }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Subscription Type</span>
                  <span className="mono" style={{ fontWeight: 600 }}>
                    {currentFeatures['subscription_type'] ?? '—'}
                  </span>
                </div>
              </div>

              {/* Scenario Builder */}
              <div className="card">
                <div className="card-header">
                  <span className="card-title">Scenario</span>
                  <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                    {Object.keys(scenario).length} changes
                  </span>
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-lg)' }}>
                  {SIMULATABLE.map(f => {
                    const current = typeof currentFeatures[f.key] === 'number' ? currentFeatures[f.key] as number : f.min;
                    const val = scenario[f.key] != null ? scenario[f.key] as number : current;
                    return (
                      <div key={f.key} className="slider-group">
                        <div className="slider-label">
                          <span className="slider-label-text">{f.label}</span>
                          <span className="slider-value">{val}</span>
                        </div>
                        <input
                          type="range"
                          min={f.min}
                          max={f.max}
                          step={f.step}
                          value={val}
                          onChange={e => {
                            const v = f.type === 'integer' ? parseInt(e.target.value) : parseFloat(e.target.value);
                            setScenario(s => ({ ...s, [f.key]: v }));
                          }}
                        />
                      </div>
                    );
                  })}

                  {/* Subscription Type */}
                  <div>
                    <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 4 }}>Subscription Type</div>
                    <select
                      className="select"
                      value={(scenario['subscription_type'] as string) ?? currentFeatures['subscription_type'] ?? ''}
                      onChange={e => setScenario(s => ({ ...s, subscription_type: e.target.value }))}
                    >
                      {SUBSCRIPTION_OPTIONS.map(o => (
                        <option key={o} value={o}>{o}</option>
                      ))}
                    </select>
                  </div>
                </div>

                <button
                  className="btn btn-primary"
                  style={{ width: '100%', marginTop: 'var(--space-xl)' }}
                  onClick={handleSimulate}
                  disabled={simulate.isPending || Object.keys(scenario).length === 0}
                >
                  {simulate.isPending ? 'Running...' : 'Run Simulation'}
                </button>
              </div>
            </div>
          )}

          {/* Result */}
          {result && (
            <div className="card mt-xl">
              <div className="card-header">
                <span className="card-title">Intervention & Sensitivity Analysis</span>
                <span className="card-subtitle">Financial exposure change under simulated customer behavior</span>
              </div>
              
              <div className="grid grid-4 mt-md" style={{ alignItems: 'center' }}>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.5px' }}>Baseline Risk</div>
                  <div style={{
                    fontSize: 26, fontWeight: 700, fontFamily: 'var(--font-mono)',
                    color: riskColor(result.risk_band_baseline),
                  }}>
                    {formatPct(result.baseline_probability)}
                  </div>
                  <span className={`risk-badge ${result.risk_band_baseline}`} style={{ marginTop: 4 }}>{result.risk_band_baseline}</span>
                </div>

                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.5px' }}>Scenario Risk</div>
                  <div style={{
                    fontSize: 26, fontWeight: 700, fontFamily: 'var(--font-mono)',
                    color: riskColor(result.risk_band_scenario),
                  }}>
                    {formatPct(result.scenario_probability)}
                  </div>
                  <span className={`risk-badge ${result.risk_band_scenario}`} style={{ marginTop: 4 }}>{result.risk_band_scenario}</span>
                </div>

                <div style={{ textAlign: 'center', borderLeft: '1px solid var(--border-subtle)', borderRight: '1px solid var(--border-subtle)' }}>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.5px' }}>Probability Shift</div>
                  <div style={{
                    fontSize: 24, fontWeight: 700, fontFamily: 'var(--font-mono)',
                    color: result.probability_change_pp < 0 ? 'var(--risk-low)' : 'var(--risk-high)',
                  }}>
                    {result.probability_change_pp > 0 ? '+' : ''}{result.probability_change_pp.toFixed(1)} pp
                  </div>
                  <div style={{ fontSize: 10, color: 'var(--text-tertiary)', marginTop: 2 }}>Sensitivity Delta</div>
                </div>

                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.5px' }}>Expected MRR Impact</div>
                  <div style={{
                    fontSize: 24, fontWeight: 700, fontFamily: 'var(--font-mono)',
                    color: (result.mrr_saved ?? (result.baseline_probability - result.scenario_probability) * 12.99) >= 0 ? 'var(--success)' : 'var(--danger)',
                  }}>
                    {(result.mrr_saved ?? (result.baseline_probability - result.scenario_probability) * 12.99) >= 0 ? '+' : ''}${((result.mrr_saved ?? (result.baseline_probability - result.scenario_probability) * 12.99)).toFixed(2)}/mo
                  </div>
                  <div style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 2 }}>
                    ${(((result.annual_value_preserved ?? ((result.baseline_probability - result.scenario_probability) * 12.99 * 12)))).toFixed(0)} Annualized ($ARR)
                  </div>
                </div>
              </div>

              <div className="mt-lg">
                <Disclaimer>
                  <strong>Causal Sensitivity Disclaimer:</strong> This simulator recalculates probability based on correlational feature associations in the production XGBoost model. It estimates model sensitivity, not a guaranteed causal treatment effect. Run live A/B holdouts to verify campaign incrementality.
                </Disclaimer>
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  );
}
