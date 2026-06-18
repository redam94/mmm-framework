import { useQuery } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import { useState } from 'react';
import {
  ArrowRightIcon,
  BeakerIcon,
  ChevronDownIcon,
  ChevronRightIcon,
  CircleStackIcon,
  CubeIcon,
  DocumentTextIcon,
  FingerPrintIcon,
} from '@heroicons/react/24/outline';
import { runsService } from '../../api/services/runsService';
import type { AssumptionDelta, RunInfo, SpecChange } from '../../api/services/runsService';
import { useProjectStore } from '../../stores/projectStore';

function fmtVal(v: unknown): string {
  if (v === null || v === undefined) return '—';
  if (typeof v === 'object') return JSON.stringify(v);
  return String(v);
}

function SpecChangeRow({ c }: { c: SpecChange }) {
  return (
    <li className="flex items-baseline gap-2 text-xs">
      <code className="text-ink-600 bg-cream-100 rounded px-1 py-0.5 shrink-0">{c.path}</code>
      <span className="text-ink-400 line-through truncate">{fmtVal(c.old)}</span>
      <span className="text-ink-300">→</span>
      <span className="text-ink-900 font-medium truncate">{fmtVal(c.new)}</span>
    </li>
  );
}

function AssumptionRow({ a }: { a: AssumptionDelta }) {
  return (
    <li className="text-xs text-ink-600">
      <span
        className={`px-1.5 py-0.5 rounded text-[10px] font-semibold mr-1.5 ${
          a.change === 'added' ? 'bg-sage-100 text-sage-800' : 'bg-gold-100 text-gold-700'
        }`}
      >
        {a.change}
      </span>
      <code className="text-ink-700">{a.key}</code>
      <span className="text-ink-400 num"> v{a.version}</span>
      {a.rationale && <span className="text-ink-400"> — {a.rationale}</span>}
    </li>
  );
}

function RunCard({ run, isLatest }: { run: RunInfo; isLatest: boolean }) {
  const navigate = useNavigate();
  const [open, setOpen] = useState(isLatest);
  const ch = run.changes;
  const nChanges = ch.spec_changes.length + ch.assumptions_delta.length + (ch.data_changed ? 1 : 0);

  return (
    <div className="relative pl-8">
      {/* timeline dot */}
      <div className={`absolute left-0 top-5 h-3.5 w-3.5 rounded-full border-2 ${isLatest ? 'bg-sage-700 border-sage-700' : 'bg-white border-line-300'}`} />
      <div className="bg-white rounded-lg border border-line-200 shadow-sm overflow-hidden mb-4">
        <button
          onClick={() => setOpen((v) => !v)}
          className="w-full flex items-center gap-3 px-5 py-3.5 hover:bg-cream-100/60 transition-colors text-left"
        >
          <CubeIcon className="h-5 w-5 text-sage-600 shrink-0" />
          <div className="flex-1 min-w-0">
            <p className="text-sm font-semibold text-ink-900 truncate">
              {run.run_name ?? run.run_id ?? 'run'}
              {isLatest && <span className="ml-2 px-1.5 py-0.5 rounded bg-sage-100 text-sage-800 text-[10px] font-semibold uppercase">latest</span>}
            </p>
            <p className="text-xs text-ink-400 mt-0.5 truncate">
              {run.timestamp_iso && <span className="num">{new Date(run.timestamp_iso).toLocaleString()}</span>}
              {run.kpi && <> · KPI <span className="text-ink-600">{run.kpi}</span></>}
              {run.channels?.length > 0 && <> · {run.channels.length} channels</>}
              {run.trend && <> · trend {run.trend}</>}
            </p>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            {run.data_fingerprint && (
              <span className="hidden sm:flex items-center gap-1 text-[11px] text-ink-400 num" title={`${run.data_fingerprint.n_rows.toLocaleString()} rows`}>
                <FingerPrintIcon className="h-3.5 w-3.5" />
                {run.data_fingerprint.md5}
              </span>
            )}
            {ch.baseline ? (
              <span className="px-2 py-0.5 rounded-full bg-cream-200 text-ink-600 text-[11px] font-medium">baseline</span>
            ) : (
              <span className={`px-2 py-0.5 rounded-full text-[11px] font-medium ${nChanges ? 'bg-steel-100 text-steel-700' : 'bg-cream-200 text-ink-600'}`}>
                {nChanges ? `${nChanges} change${nChanges > 1 ? 's' : ''}` : 'no changes'}
              </span>
            )}
            {open ? <ChevronDownIcon className="h-4 w-4 text-ink-300" /> : <ChevronRightIcon className="h-4 w-4 text-ink-300" />}
          </div>
        </button>

        {open && (
          <div className="px-5 pb-4 pt-1 border-t border-line-200 space-y-3">
            {run.summary && <p className="text-xs text-ink-600 leading-relaxed">{run.summary}</p>}

            {run.diagnostics?.convergence && (
              <div className="flex flex-wrap items-center gap-2 text-[11px]">
                <span className="text-ink-400 uppercase text-[10px] tracking-wide">Model health</span>
                <span
                  className={`px-2 py-0.5 rounded-full font-medium ${
                    run.diagnostics.convergence.ok
                      ? 'bg-sage-100 text-sage-800'
                      : 'bg-rust-100 text-rust-700'
                  }`}
                >
                  {run.diagnostics.convergence.ok
                    ? 'converged'
                    : `convergence: ${run.diagnostics.convergence.flags.join(', ')}`}
                </span>
                <span className="num text-ink-600">
                  {run.diagnostics.convergence.divergences ?? '—'} div · R̂{' '}
                  {run.diagnostics.convergence.rhat_max?.toFixed(3) ?? '—'} · ESS{' '}
                  {run.diagnostics.convergence.ess_bulk_min != null
                    ? Math.round(run.diagnostics.convergence.ess_bulk_min)
                    : '—'}
                </span>
                {(run.diagnostics.learning?.verdict_counts['prior-dominated'] ?? 0) > 0 && (
                  <span className="px-2 py-0.5 rounded-full bg-rust-100 text-rust-700 font-medium">
                    {run.diagnostics.learning!.verdict_counts['prior-dominated']} prior-dominated
                  </span>
                )}
                <button
                  onClick={() => navigate('/performance/health')}
                  className="text-sage-700 font-medium hover:underline"
                >
                  details
                </button>
              </div>
            )}

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
              <div>
                <p className="text-ink-400 uppercase text-[10px] tracking-wide mb-0.5">Data</p>
                {run.data_fingerprint ? (
                  <p className="text-ink-700">
                    <code className="num">{run.data_fingerprint.md5}</code> · <span className="num">{run.data_fingerprint.n_rows.toLocaleString()}</span> rows
                    {ch.data_changed && <span className="ml-1 text-gold-700 font-medium">(changed)</span>}
                  </p>
                ) : (
                  <p className="text-ink-300">not fingerprinted</p>
                )}
              </div>
              <div>
                <p className="text-ink-400 uppercase text-[10px] tracking-wide mb-0.5">Spec hash</p>
                <p className="text-ink-700"><code className="num">{run.spec_hash ?? '—'}</code></p>
              </div>
              <div>
                <p className="text-ink-400 uppercase text-[10px] tracking-wide mb-0.5">Inference</p>
                <p className="text-ink-700 num">
                  {run.inference ? `${run.inference.chains}ch × ${run.inference.draws}dr` : '—'}
                </p>
              </div>
              <div>
                <p className="text-ink-400 uppercase text-[10px] tracking-wide mb-0.5">Parent run</p>
                <p className="text-ink-700"><code className="num">{run.parent_run_id ?? '—'}</code></p>
              </div>
            </div>

            {!ch.baseline && ch.spec_changes.length > 0 && (
              <div>
                <p className="text-[10px] font-semibold text-ink-400 uppercase tracking-wide mb-1">
                  Spec changes vs previous run
                </p>
                <ul className="space-y-1">
                  {ch.spec_changes.map((c) => <SpecChangeRow key={c.path} c={c} />)}
                </ul>
              </div>
            )}

            {ch.assumptions_delta.length > 0 && (
              <div>
                <p className="text-[10px] font-semibold text-ink-400 uppercase tracking-wide mb-1">
                  Rationale ({ch.baseline ? 'assumption stack at first fit' : 'assumptions added/revised'})
                </p>
                <ul className="space-y-1">
                  {ch.assumptions_delta.map((a) => <AssumptionRow key={`${a.key}-v${a.version}`} a={a} />)}
                </ul>
              </div>
            )}

            <div className="flex items-center gap-4 pt-1">
              <button
                onClick={() => navigate(`/chat?session=${run.thread_id}`)}
                className="flex items-center gap-1 text-xs font-medium text-sage-700 hover:underline"
              >
                Open session <ArrowRightIcon className="h-3 w-3" />
              </button>
              {run.report_path && (
                <span className="flex items-center gap-1 text-xs text-ink-400" title={run.report_path}>
                  <DocumentTextIcon className="h-3.5 w-3.5" /> report generated
                </span>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export function RunsTimeline() {
  const { currentProjectId } = useProjectStore();
  const { data: runs = [], isLoading } = useQuery({
    queryKey: ['runs', currentProjectId],
    queryFn: () => runsService.listRuns(currentProjectId),
    staleTime: 30000,
  });

  return (
    <div className="space-y-6 max-w-4xl">
      <p className="text-sm text-ink-400">
        Lineage of every fit — data fingerprint, spec changes, and the rationale behind each
        revision. Ask the agent for <code className="text-xs bg-cream-200 rounded px-1">get_run_history</code> to
        fold this into a final report.
      </p>

      {isLoading ? (
        <p className="text-sm text-ink-400">Loading…</p>
      ) : runs.length === 0 ? (
        <div className="bg-white rounded-lg border border-line-200 px-6 py-10 text-center">
          <CircleStackIcon className="h-9 w-9 text-ink-300 mx-auto mb-2" />
          <p className="font-display text-sm text-ink-400">No model runs recorded yet.</p>
          <p className="text-xs text-ink-300 mt-1 flex items-center justify-center gap-1">
            <BeakerIcon className="h-3.5 w-3.5" /> Fit a model in an agent session and it will appear here.
          </p>
        </div>
      ) : (
        <div className="relative">
          {/* timeline spine */}
          <div className="absolute left-[6px] top-6 bottom-6 w-px bg-line-200" />
          {runs.map((r, i) => (
            <RunCard key={r.artifact_id} run={r} isLatest={i === 0} />
          ))}
        </div>
      )}
    </div>
  );
}

export default RunsTimeline;
