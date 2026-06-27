import { Loader2 } from 'lucide-react';
import { DashWidget } from '../../common/DashWidget';
import { StudioEdaChart, StudioTable } from '../StudioEdaChart';
import { useStudioAnalysis } from '../useStudioAnalysis';
import type { StudioEdaResult } from '../../../types';

type RunEda = (analyses: string[], opts?: { sensitivity?: string }) => Promise<StudioEdaResult | null>;

export function AnalysisLoading() {
  return (
    <div className="flex items-center justify-center py-16 text-ink-300">
      <Loader2 size={22} className="animate-spin" />
    </div>
  );
}

// A blocking-EDA warning (e.g. an unparseable date column) — rendered as a
// guiding banner instead of a silent empty tab.
export function EdaWarnings({ warnings }: { warnings?: string[] }) {
  if (!warnings || warnings.length === 0) return null;
  return (
    <div className="space-y-1.5 mb-3">
      {warnings.map((w, i) => (
        <p key={i} className="text-xs text-amber-800 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2">{w}</p>
      ))}
    </div>
  );
}

// Generic panel: run one or more analyses and render their figures + tables.
export function AnalysisPanel({
  runEda, rev, analyses, title, dotColor = 'bg-indigo-500', emptyHint,
}: {
  runEda: RunEda; rev: number; analyses: string[]; title: string; dotColor?: string; emptyHint: string;
}) {
  const { res, loading } = useStudioAnalysis(runEda, analyses, rev);
  if (loading) return <AnalysisLoading />;
  const sections = analyses.map(a => res?.analyses?.[a]).filter(Boolean);
  const figures = sections.flatMap(s => s!.figures);
  const tables = sections.flatMap(s => s!.tables);
  if (figures.length === 0 && tables.length === 0) {
    return (
      <>
        <EdaWarnings warnings={res?.warnings} />
        <p className="text-sm text-ink-300 py-8 text-center">{emptyHint}</p>
      </>
    );
  }
  return (
    <DashWidget title={title} dotColor={dotColor} color="indigo">
      <EdaWarnings warnings={res?.warnings} />
      <div className="space-y-4">
        {figures.map((f, i) => <StudioEdaChart key={f.key} chart={f} idx={i} />)}
        {tables.map((t, i) => <StudioTable key={`t-${i}`} table={t} />)}
      </div>
    </DashWidget>
  );
}
