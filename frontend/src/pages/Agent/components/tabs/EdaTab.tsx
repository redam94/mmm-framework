import { useState } from 'react';
import { AlertTriangle, ArrowRight, Check, FlaskConical, Loader2 } from 'lucide-react';
import { Badge } from '../common/Badge';
import { DashWidget } from '../common/DashWidget';
import { EmptyTabState } from '../common/EmptyTabState';
import { TableCard } from '../tables/TableCard';
import type { EdaFindings, EdaIssue, OutlierAction, TableRef } from '../../types';

// ─── EDA tab ──────────────────────────────────────────────────────────────────
// Data-quality issues + proposed outlier treatments (with one-click confirm)
// + the structured tables produced by validate_data / run_eda / detect_outliers.

const SEVERITY_ORDER = ['error', 'warning', 'info'] as const;

// Checks that are causal-identification screens, not generic data hygiene:
// each one threatens the positivity/overlap assumption (enough spend variation
// — including dark periods — to identify the zero-out counterfactual) or
// flags posteriors that will lean on priors instead of data.
const IDENTIFICATION_CHECKS = new Set(['constant_series', 'zero_inflation', 'short_history']);

const SEVERITY_BADGE: Record<string, 'red' | 'amber' | 'blue'> = {
  error: 'red',
  warning: 'amber',
  info: 'blue',
};

function severityBucket(s: string): 'error' | 'warning' | 'info' {
  return s === 'error' || s === 'warning' ? s : 'info';
}

export function IssueRow({ issue }: { issue: EdaIssue }) {
  return (
    <div className="flex items-start gap-2.5 px-3 py-2.5 bg-white rounded-lg border border-line-200">
      <span className="shrink-0 mt-0.5">
        <Badge label={severityBucket(issue.severity)} color={SEVERITY_BADGE[severityBucket(issue.severity)]} />
      </span>
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2 flex-wrap">
          <code className="bg-cream-100 px-1.5 py-0.5 rounded text-[11px] font-mono text-indigo-600">
            {issue.check}
          </code>
          {issue.variable && (
            <span className="text-[11px] font-medium text-ink-400">{issue.variable}</span>
          )}
        </div>
        <p className="text-sm text-ink-700 mt-0.5">{issue.message}</p>
      </div>
    </div>
  );
}

export function OutlierActionRow({
  action, disabled, onConfirm,
}: {
  action: OutlierAction;
  disabled: boolean;
  onConfirm: (action: OutlierAction) => Promise<string | null>;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleConfirm = async () => {
    setBusy(true);
    setError(null);
    const err = await onConfirm(action);
    setBusy(false);
    if (err) setError(err);
  };

  const applied = action.status === 'applied';

  return (
    <div className="px-3 py-2.5 bg-white rounded-lg border border-line-200">
      <div className="flex items-center gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-sm font-medium text-ink-900">{action.strategy}</span>
            {action.variable && (
              <code className="bg-cream-100 px-1.5 py-0.5 rounded text-[11px] font-mono text-indigo-600">
                {action.variable}
              </code>
            )}
          </div>
          {action.rationale && (
            <p className="text-xs text-ink-400 mt-0.5">{action.rationale}</p>
          )}
        </div>
        {applied ? (
          <span className="flex items-center gap-1 px-2.5 py-1 rounded-full bg-emerald-50 border border-emerald-200 text-emerald-700 text-xs font-medium shrink-0">
            <Check size={12} /> Applied
          </span>
        ) : action.status === 'proposed' ? (
          <button
            onClick={handleConfirm}
            disabled={disabled || busy}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed shrink-0"
          >
            {busy && <Loader2 size={12} className="animate-spin" />}
            {busy ? 'Applying…' : 'Confirm'}
          </button>
        ) : (
          <Badge label={action.status} color="gray" />
        )}
      </div>
      {error && <p className="text-xs text-red-600 mt-1.5">{error}</p>}
    </div>
  );
}

export function EdaTab({
  eda, tables, disabled, onResolveAction, onNavigate,
}: {
  eda?: EdaFindings;
  tables: TableRef[];
  disabled: boolean;
  onResolveAction: (action: OutlierAction) => Promise<string | null>;
  onNavigate: (tab: string) => void;
}) {
  const allIssues = eda?.issues ?? [];
  const identIssues = allIssues.filter(i => IDENTIFICATION_CHECKS.has(i.check));
  const issues = allIssues.filter(i => !IDENTIFICATION_CHECKS.has(i.check));
  const actions = eda?.outlier_actions ?? [];
  const damaged = eda?.normalization_damaged ?? [];

  if (allIssues.length === 0 && actions.length === 0 && damaged.length === 0 && tables.length === 0) {
    return (
      <EmptyTabState
        icon={<FlaskConical size={28} />}
        title="No EDA findings yet"
        hint="Ask the agent to validate the dataset or run EDA — it'll call `validate_data` / `run_eda` and data-quality findings will appear here."
      />
    );
  }

  // Group tables by source preserving arrival order.
  const tablesBySource: [string, TableRef[]][] = [];
  for (const t of tables) {
    const entry = tablesBySource.find(([s]) => s === t.source);
    if (entry) entry[1].push(t);
    else tablesBySource.push([t.source, [t]]);
  }

  return (
    <>
      {identIssues.length > 0 && (
        <DashWidget
          title={`Identification Risks (${identIssues.length})`}
          dotColor="bg-amber-500"
          color="amber"
        >
          <div className="space-y-1.5">
            <p className="text-xs text-ink-400 px-0.5 pb-1">
              These aren't hygiene problems — they threaten <em>positivity</em>: a channel
              needs real variation (including dark or near-dark periods) for the data to
              identify its zero-out counterfactual. Where variation is missing, the fitted ROI
              is extrapolated from the prior, not measured — a calibration experiment is the
              honest fix, not a tighter prior.
            </p>
            {identIssues.map((issue, i) => <IssueRow key={`ident-${i}`} issue={issue} />)}
          </div>
        </DashWidget>
      )}

      {issues.length > 0 && (
        <DashWidget title={`Data Quality Issues (${issues.length})`} dotColor="bg-red-500" color="red">
          <div className="space-y-3">
            {SEVERITY_ORDER.map(sev => {
              const group = issues.filter(i => severityBucket(i.severity) === sev);
              if (group.length === 0) return null;
              return (
                <div key={sev} className="space-y-1.5">
                  {group.map((issue, i) => <IssueRow key={`${sev}-${i}`} issue={issue} />)}
                </div>
              );
            })}
          </div>
        </DashWidget>
      )}

      {(actions.length > 0 || damaged.length > 0) && (
        <DashWidget title={`Outlier Actions (${actions.length})`} dotColor="bg-amber-500" color="amber">
          <div className="space-y-2">
            {damaged.length > 0 && (
              <div className="flex items-start gap-2 px-3 py-2.5 rounded-lg bg-amber-50 border border-amber-200 text-amber-800 text-sm">
                <AlertTriangle size={15} className="shrink-0 mt-0.5" />
                <span>A single point sets the saturation scale for: {damaged.join(', ')}</span>
              </div>
            )}
            {actions.map(a => (
              <OutlierActionRow
                key={a.action_id}
                action={a}
                disabled={disabled}
                onConfirm={onResolveAction}
              />
            ))}
          </div>
        </DashWidget>
      )}

      {tablesBySource.map(([source, group]) => (
        <DashWidget
          key={source}
          title={`EDA Results — ${source} (${group.length})`}
          dotColor="bg-indigo-500"
          color="indigo"
        >
          <div className="space-y-4">
            {group.map((t, idx) => <TableCard key={t.id} tableRef={t} idx={idx} />)}
          </div>
        </DashWidget>
      ))}

      {tables.length > 0 && (
        <button
          onClick={() => onNavigate('plots')}
          className="flex items-center gap-1.5 text-sm text-ink-400 hover:text-indigo-600 transition-colors self-start"
        >
          EDA charts are in the Plots tab <ArrowRight size={14} />
        </button>
      )}
    </>
  );
}
