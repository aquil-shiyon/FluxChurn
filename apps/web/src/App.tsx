import { BrowserRouter, Routes, Route, NavLink, Navigate, useLocation } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import OverviewPage from './pages/OverviewPage';
import CustomersPage from './pages/CustomersPage';
import CohortsPage from './pages/CohortsPage';
import SimulatorPage from './pages/SimulatorPage';
import ModelLabPage from './pages/ModelLabPage';
import MonitoringPage from './pages/MonitoringPage';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});

function Navigation() {
  const location = useLocation();

  const getPageInfo = (path: string) => {
    switch (path) {
      case '/customers':
        return { title: 'Customer Intelligence', subtitle: 'Customer-level risk profiles and SHAP waterfall contributions' };
      case '/cohorts':
        return { title: 'Cohort Risk Explorer', subtitle: 'Multidimensional slice-and-dice segmentation analytics' };
      case '/simulator':
        return { title: 'What-If Risk Simulator', subtitle: 'Counterfactual sensitivity analysis and retention intervention sandbox' };
      case '/models':
        return { title: 'Model Performance Lab', subtitle: 'Production candidate benchmarking, PR/ROC evaluation, and calibration' };
      case '/monitoring':
        return { title: 'System & Model Monitoring', subtitle: 'Feature drift diagnostics, data contracts, and production stability' };
      case '/':
      case '/overview':
      default:
        return { title: 'Executive Risk Overview', subtitle: 'Portfolio-wide 30-day churn probability distribution and top global drivers' };
    }
  };

  const { title, subtitle } = getPageInfo(location.pathname);

  return (
    <div className="app-shell">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-brand">
          <div className="brand-icon">⚡</div>
          <span className="brand-name">FlixChurn</span>
        </div>

        <nav className="sidebar-nav">
          <div className="sidebar-section">
            <span className="sidebar-section-label">Risk Analytics</span>
          </div>

          <NavLink
            to="/"
            end
            className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
          >
            <svg className="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.75}>
              <rect x="3" y="3" width="7" height="7" rx="1.5" />
              <rect x="14" y="3" width="7" height="7" rx="1.5" />
              <rect x="14" y="14" width="7" height="7" rx="1.5" />
              <rect x="3" y="14" width="7" height="7" rx="1.5" />
            </svg>
            Overview
          </NavLink>

          <NavLink
            to="/customers"
            className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
          >
            <svg className="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.75}>
              <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
              <circle cx="9" cy="7" r="4" />
              <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
              <path d="M16 3.13a4 4 0 0 1 0 7.75" />
            </svg>
            Customer Intel
          </NavLink>

          <NavLink
            to="/cohorts"
            className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
          >
            <svg className="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.75}>
              <line x1="18" y1="20" x2="18" y2="10" />
              <line x1="12" y1="20" x2="12" y2="4" />
              <line x1="6" y1="20" x2="6" y2="14" />
            </svg>
            Cohort Explorer
          </NavLink>

          <div className="sidebar-section">
            <span className="sidebar-section-label">Decision Support</span>
          </div>

          <NavLink
            to="/simulator"
            className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
          >
            <svg className="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.75}>
              <circle cx="12" cy="12" r="3" />
              <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" />
            </svg>
            What-If Simulator
          </NavLink>

          <div className="sidebar-section">
            <span className="sidebar-section-label">MLOps & Governance</span>
          </div>

          <NavLink
            to="/models"
            className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
          >
            <svg className="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.75}>
              <polygon points="12 2 2 7 12 12 22 7 12 2" />
              <polyline points="2 17 12 22 22 17" />
              <polyline points="2 12 12 17 22 12" />
            </svg>
            Model Lab
          </NavLink>

          <NavLink
            to="/monitoring"
            className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
          >
            <svg className="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.75}>
              <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
            </svg>
            Monitoring
          </NavLink>
        </nav>

        <div className="sidebar-footer">
          <div className="sidebar-meta">
            <span>Model: <strong className="font-mono text-primary">v1.0.0-xgb</strong></span>
            <span>Horizon: <strong className="text-primary">30 Days</strong></span>
            <span className="text-muted">Batch: 7,043 scored</span>
          </div>
        </div>
      </aside>

      {/* Main Area */}
      <div className="main-area">
        {/* Top Bar */}
        <header className="topbar">
          <div className="topbar-left">
            <div>
              <span className="topbar-title">{title}</span>
              <span className="topbar-subtitle" style={{ marginLeft: 12 }}>{subtitle}</span>
            </div>
          </div>
          <div className="topbar-right">
            <span className="topbar-tag horizon">
              <span>●</span> 30-Day Forward Window
            </span>
            <span className="topbar-tag model">
              Production: XGBoost
            </span>
          </div>
        </header>

        {/* Content Area */}
        <main className="content">
          <Routes>
            <Route path="/" element={<OverviewPage />} />
            <Route path="/overview" element={<Navigate to="/" replace />} />
            <Route path="/customers" element={<CustomersPage />} />
            <Route path="/cohorts" element={<CohortsPage />} />
            <Route path="/simulator" element={<SimulatorPage />} />
            <Route path="/models" element={<ModelLabPage />} />
            <Route path="/monitoring" element={<MonitoringPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Navigation />
      </BrowserRouter>
    </QueryClientProvider>
  );
}
