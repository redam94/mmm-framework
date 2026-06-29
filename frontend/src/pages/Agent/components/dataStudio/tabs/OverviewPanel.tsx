import { DashWidget } from '../../common/DashWidget';
import { StudioEdaChart, StudioTable } from '../StudioEdaChart';
import { AnalysisLoading } from './AnalysisPanel';
import { useStudioAnalysis } from '../useStudioAnalysis';
import type { DataStudioState, StudioEdaResult } from '../../../types';

type RunEda = (analyses: string[], opts?: { sensitivity?: string }) => Promise<StudioEdaResult | null>;

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="bg-cream-50 p-3 rounded-xl border border-line-200">
      <p className="text-[10px] text-ink-300 uppercase tracking-wider mb-1">{label}</p>
      <p className="text-2xl font-bold text-ink-900">{value}</p>
    </div>
  );
}

export function OverviewPanel({ state, runEda, rev }: { state: DataStudioState; runEda: RunEda; rev: number }) {
  const { res, loading } = useStudioAnalysis(runEda, ['overview'], rev);
  const ov = res?.analyses?.overview;
  return (
    <div className="space-y-4">
      <DashWidget title="Dataset summary" dotColor="bg-indigo-500" color="indigo">
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <Stat label="Rows" value={state.n_rows.toLocaleString()} />
          <Stat label="Columns" value={state.n_cols} />
          <Stat label="Layout" value={state.is_long ? 'MFF (long)' : 'Wide'} />
          <Stat label="Date column" value={state.date_col ?? '—'} />
        </div>
        {state.warnings.length > 0 && (
          <ul className="mt-3 space-y-1">
            {state.warnings.map((w, i) => (
              <li key={i} className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded px-2.5 py-1">{w}</li>
            ))}
          </ul>
        )}
      </DashWidget>

      {loading ? <AnalysisLoading /> : ov && (ov.figures.length > 0 || ov.tables.length > 0) ? (
        <DashWidget title="Variables" dotColor="bg-violet-500" color="violet">
          <div className="space-y-4">
            {ov.figures.map((f, i) => <StudioEdaChart key={f.key} chart={f} idx={i} />)}
            {ov.tables.map((t, i) => <StudioTable key={`t-${i}`} table={t} />)}
          </div>
        </DashWidget>
      ) : null}
    </div>
  );
}
