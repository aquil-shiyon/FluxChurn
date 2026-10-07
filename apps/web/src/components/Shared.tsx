import type { ReactNode } from 'react';

/** Skeleton loading placeholder. */
export function Skeleton({ width, height, style }: { width?: string | number; height?: string | number; style?: React.CSSProperties }) {
  return <div className="skeleton" style={{ width: width ?? '100%', height: height ?? 14, ...style }} />;
}

/** Loading skeleton for a KPI card. */
export function KpiSkeleton() {
  return (
    <div className="card">
      <Skeleton width="60%" height={12} style={{ marginBottom: 12 }} />
      <Skeleton width="40%" height={32} />
    </div>
  );
}

/** Loading state for a card section. */
export function CardSkeleton({ lines = 5 }: { lines?: number }) {
  return (
    <div className="card">
      <Skeleton width="45%" height={12} style={{ marginBottom: 16 }} />
      {Array.from({ length: lines }).map((_, i) => (
        <Skeleton
          key={i}
          width={`${60 + Math.random() * 35}%`}
          height={14}
          style={{ marginBottom: 10 }}
        />
      ))}
    </div>
  );
}

/** Error state display. */
export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="state-message">
      <svg className="state-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5}>
        <circle cx="12" cy="12" r="10" />
        <path d="M12 8v4m0 4h.01" />
      </svg>
      <h3>Unable to load data</h3>
      <p>{message}</p>
      {onRetry && (
        <button className="btn btn-ghost btn-sm" onClick={onRetry}>
          Retry
        </button>
      )}
    </div>
  );
}

/** Empty state display. */
export function EmptyState({ title, description }: { title: string; description?: string }) {
  return (
    <div className="state-message">
      <svg className="state-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5}>
        <rect x="3" y="3" width="18" height="18" rx="2" />
        <path d="M9 9h6m-6 3h4" />
      </svg>
      <h3>{title}</h3>
      {description && <p>{description}</p>}
    </div>
  );
}

/** Disclaimer banner. */
export function Disclaimer({ children }: { children: ReactNode }) {
  return (
    <div className="disclaimer">
      <svg className="disclaimer-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
        <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
        <path d="M12 9v4m0 4h.01" />
      </svg>
      <span>{children}</span>
    </div>
  );
}
