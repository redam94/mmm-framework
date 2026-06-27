import { useEffect, useState } from 'react';
import { Database, X } from 'lucide-react';
import { CommitBar } from './CommitBar';
import { StudioTabBar, type StudioTab } from './StudioTabBar';
import { UploadStep } from './UploadStep';
import { OverviewPanel } from './tabs/OverviewPanel';
import { OutliersPanel } from './tabs/OutliersPanel';
import { AnalysisPanel } from './tabs/AnalysisPanel';
import { TransformPanel } from './tabs/TransformPanel';
import { useDataStudio, type CommitPayload } from './useDataStudio';
import type { StudioRole, TransformStep } from '../../types';

// Full-screen Data Studio: stage a raw upload, explore + clean it interactively,
// then convert it into the session's working dataset. Bespoke shell (3 zones:
// header, scrollable tab body, sticky CommitBar) — the stock Modal can't host a
// sticky footer.
export function DataStudioModal({
  threadId, apiKey, modelName, chatLoading, onClose, onCommitted,
}: {
  threadId: string | null;
  apiKey: string | null;
  modelName: string | null;
  chatLoading: boolean;
  onClose: () => void;
  onCommitted: (payload: CommitPayload) => void;
}) {
  const studio = useDataStudio(threadId, apiKey, modelName);
  const { state, loading, error, rev, hydrate, upload, setSteps, runEda, commit, discard } = studio;
  const [step, setStep] = useState<'upload' | 'studio'>('upload');
  const [tab, setTab] = useState<StudioTab>('overview');
  const [busy, setBusy] = useState(false);

  // Hydrate on mount: a prior staging (uncommitted) reopens where it was left.
  useEffect(() => {
    let alive = true;
    hydrate().then(s => { if (alive && s) setStep('studio'); });
    return () => { alive = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => { if (state && step !== 'studio') setStep('studio'); }, [state, step]);

  // Escape closes (staging is kept server-side; reopening rehydrates).
  useEffect(() => {
    const h = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', h);
    return () => window.removeEventListener('keydown', h);
  }, [onClose]);

  const onUpload = async (file: File) => { const s = await upload(file); if (s) setStep('studio'); };
  const onSteps = (steps: TransformStep[]) => { void setSteps(steps); };
  const onRoles = (roles: Record<string, StudioRole>) => { if (state) void setSteps(state.steps, roles); };
  const onAccept = async (s: TransformStep) => (state ? setSteps([...state.steps, s]) : false);

  const handleCommit = async () => {
    setBusy(true);
    const payload = await commit();
    setBusy(false);
    if (payload) onCommitted(payload);
  };

  const handleDiscard = async () => {
    if (!window.confirm('Discard this staged dataset and all cleaning steps?')) return;
    await discard();
    setStep('upload');
  };

  return (
    <div className="fixed inset-0 z-50 flex flex-col bg-ink-900/40 backdrop-blur-sm p-4">
      <div className="relative flex flex-col flex-1 bg-white border border-line-200 rounded-2xl shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3.5 border-b border-line-200 shrink-0">
          <div className="flex items-center gap-2.5 min-w-0">
            <span className="w-8 h-8 rounded-lg bg-sage-100 flex items-center justify-center text-sage-700 shrink-0">
              <Database size={16} />
            </span>
            <div className="min-w-0">
              <h2 className="text-base font-bold text-ink-900 truncate">Data Studio</h2>
              <p className="text-xs text-ink-300 truncate">
                {state ? `${state.filename} · clean before it becomes the working dataset` : 'Upload, explore & clean your data'}
              </p>
            </div>
          </div>
          <button onClick={onClose} className="p-2 rounded-full hover:bg-cream-100 text-ink-300 hover:text-ink-700 transition-colors shrink-0" title="Close (Esc)">
            <X size={18} />
          </button>
        </div>

        {step === 'studio' && state && (
          <StudioTabBar active={tab} onChange={setTab} dots={{ transform: state.steps.length > 0 }} />
        )}

        {/* Body */}
        <div className="flex-1 overflow-y-auto p-5 bg-cream-50">
          {step === 'upload' || !state ? (
            <UploadStep onUpload={onUpload} uploading={loading} error={error} />
          ) : (
            <>
              {tab === 'overview' && <OverviewPanel state={state} runEda={runEda} rev={rev} />}
              {tab === 'distributions' && <AnalysisPanel runEda={runEda} rev={rev} analyses={['distributions']} title="Distributions" emptyHint="No distribution charts for the current variables." />}
              {tab === 'correlation' && <AnalysisPanel runEda={runEda} rev={rev} analyses={['correlation']} title="Correlation & VIF" emptyHint="Need ≥2 media/control variables for correlation." />}
              {tab === 'missingness' && <AnalysisPanel runEda={runEda} rev={rev} analyses={['missingness']} title="Missingness" emptyHint="No missingness map available." />}
              {tab === 'outliers' && <OutliersPanel runEda={runEda} rev={rev} onAccept={onAccept} />}
              {tab === 'transform' && <TransformPanel state={state} busy={loading || busy} onSteps={onSteps} onRoles={onRoles} />}
            </>
          )}
        </div>

        {/* Commit footer */}
        {step === 'studio' && state && (
          <CommitBar state={state} busy={busy} chatLoading={chatLoading} error={error}
            onCommit={handleCommit} onDiscard={handleDiscard} />
        )}
      </div>
    </div>
  );
}
