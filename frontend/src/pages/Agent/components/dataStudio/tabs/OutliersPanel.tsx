import { useState } from 'react';
import { AlertTriangle } from 'lucide-react';
import { DashWidget } from '../../common/DashWidget';
import { IssueRow, OutlierActionRow } from '../../tabs/EdaTab';
import { StudioEdaChart, StudioTable } from '../StudioEdaChart';
import { AnalysisLoading } from './AnalysisPanel';
import { useStudioAnalysis } from '../useStudioAnalysis';
import type { OutlierAction, StudioEdaResult, StudioOutlierSuggestion, TransformStep } from '../../../types';

type RunEda = (analyses: string[], opts?: { sensitivity?: string }) => Promise<StudioEdaResult | null>;

const SENSITIVITIES = ['low', 'default', 'high'] as const;

export function OutliersPanel({
  runEda, rev, onAccept,
}: {
  runEda: RunEda; rev: number;
  onAccept: (step: TransformStep) => Promise<boolean>;
}) {
  const [sensitivity, setSensitivity] = useState<string>('default');
  const { res, loading } = useStudioAnalysis(runEda, ['outliers'], rev, sensitivity);
  const [accepted, setAccepted] = useState<Set<string>>(new Set());

  const handleAccept = async (s: StudioOutlierSuggestion): Promise<string | null> => {
    if (!s.step) return 'This suggestion has no automatic fix — apply it in the Model tab.';
    const ok = await onAccept(s.step);
    if (!ok) return 'Could not add the cleaning step.';
    setAccepted(prev => new Set(prev).add(s.action_id));
    return null;
  };

  const out = res?.analyses?.outliers;
  const suggestions = res?.outlier_suggestions ?? [];
  const issues = res?.issues ?? [];
  const damaged = res?.normalization_damaged ?? [];

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <span className="text-xs text-ink-400 font-medium">Sensitivity:</span>
        {SENSITIVITIES.map(s => (
          <button
            key={s}
            onClick={() => setSensitivity(s)}
            className={`px-2.5 py-1 rounded-full text-xs font-medium border transition-colors ${
              sensitivity === s
                ? 'bg-indigo-600 text-white border-indigo-600'
                : 'bg-white text-ink-600 border-line-200 hover:border-indigo-400'
            }`}
          >{s}</button>
        ))}
      </div>

      {loading ? <AnalysisLoading /> : (
        <>
          {(suggestions.length > 0 || damaged.length > 0) && (
            <DashWidget title={`Suggested fixes (${suggestions.length})`} dotColor="bg-amber-500" color="amber">
              <div className="space-y-2">
                {damaged.length > 0 && (
                  <div className="flex items-start gap-2 px-3 py-2.5 rounded-lg bg-amber-50 border border-amber-200 text-amber-800 text-sm">
                    <AlertTriangle size={15} className="shrink-0 mt-0.5" />
                    <span>A single point sets the saturation scale for: {damaged.join(', ')}</span>
                  </div>
                )}
                {suggestions.map(s => {
                  const action: OutlierAction = {
                    action_id: s.action_id, strategy: s.strategy, variable: s.variable,
                    rationale: s.rationale,
                    status: accepted.has(s.action_id) ? 'applied' : 'proposed',
                  };
                  return (
                    <OutlierActionRow
                      key={s.action_id}
                      action={action}
                      disabled={false}
                      onConfirm={() => handleAccept(s)}
                    />
                  );
                })}
              </div>
            </DashWidget>
          )}

          {issues.length > 0 && (
            <DashWidget title={`Data-quality issues (${issues.length})`} dotColor="bg-red-500" color="red">
              <div className="space-y-1.5">
                {issues.map((i, idx) => <IssueRow key={idx} issue={i} />)}
              </div>
            </DashWidget>
          )}

          {out && (out.figures.length > 0 || out.tables.length > 0) && (
            <DashWidget title="Outlier detail" dotColor="bg-indigo-500" color="indigo">
              <div className="space-y-4">
                {out.figures.map((f, i) => <StudioEdaChart key={f.key} chart={f} idx={i} />)}
                {out.tables.map((t, i) => <StudioTable key={`t-${i}`} table={t} />)}
              </div>
            </DashWidget>
          )}

          {suggestions.length === 0 && issues.length === 0 && (out?.figures.length ?? 0) === 0 && (
            <p className="text-sm text-ink-300 py-8 text-center">
              No outliers detected at <code>{sensitivity}</code> sensitivity.
            </p>
          )}
        </>
      )}
    </div>
  );
}
