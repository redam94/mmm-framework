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
      <code className="text-gray-600 bg-gray-50 rounded px-1 py-0.5 shrink-0">{c.path}</code>
      <span className="text-gray-400 line-through truncate">{fmtVal(c.old)}</span>
      <span className="text-gray-300">→</span>
      <span className="text-gray-800 font-medium truncate">{fmtVal(c.new)}</span>
    </li>
  );
}

function AssumptionRow({ a }: { a: AssumptionDelta }) {
  return (
    <li className="text-xs text-gray-600">
      <span
        className={`px-1.5 py-0.5 rounded text-[10px] font-semibold mr-1.5 ${
          a.change === 'added' ? 'bg-emerald-50 text-emerald-700' : 'bg-amber-50 text-amber-700'
        }`}
      >
        {a.change}
      </span>
      <code className="text-gray-700">{a.key}</code>
      <span className="text-gray-400"> v{a.version}</span>
      {a.rationale && <span className="text-gray-400"> — {a.rationale}</span>}
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
      <div className={`absolute left-0 top-5 h-3.5 w-3.5 rounded-full border-2 ${isLatest ? 'bg-indigo-500 border-indigo-500' : 'bg-white border-gray-300'}`} />
      <div className="bg-white rounded-2xl border border-gray-200 shadow-sm overflow-hidden mb-4">
        <button
          onClick={() => setOpen((v) => !v)}
          className="w-full flex items-center gap-3 px-5 py-3.5 hover:bg-gray-50 transition-colors text-left"
        >
          <CubeIcon className="h-4.5 w-4.5 h-5 w-5 text-teal-500 shrink-0" />
          <div className="flex-1 min-w-0">
            <p className="text-sm font-semibold text-gray-900 truncate">
              {run.run_name ?? run.run_id ?? 'run'}
              {isLatest && <span className="ml-2 px-1.5 py-0.5 rounded bg-indigo-50 text-indigo-600 text-[10px] font-semibold uppercase">latest</span>}
            </p>
            <p className="text-xs text-gray-400 mt-0.5 truncate">
              {run.timestamp_iso ? new Date(run.timestamp_iso).toLocaleString() : ''}
              {run.kpi && <> · KPI <span className="text-gray-500">{run.kpi}</span></>}
              {run.channels?.length > 0 && <> · {run.channels.length} channels</>}
              {run.trend && <> · trend {run.trend}</>}
            </p>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            {run.data_fingerprint && (
              <span className="hidden sm:flex items-center gap-1 text-[11px] text-gray-400" title={`${run.data_fingerprint.n_rows.toLocaleString()} rows`}>
                <FingerPrintIcon className="h-3.5 w-3.5" />
                {run.data_fingerprint.md5}
              </span>
            )}
            {ch.baseline ? (
              <span className="px-2 py-0.5 rounded-full bg-gray-100 text-gray-500 text-[11px] font-medium">baseline</span>
            ) : (
              <span className={`px-2 py-0.5 rounded-full text-[11px] font-medium ${nChanges ? 'bg-violet-50 text-violet-700' : 'bg-gray-100 text-gray-500'}`}>
                {nChanges ? `${nChanges} change${nChanges > 1 ? 's' : ''}` : 'no changes'}
              </span>
            )}
            {open ? <ChevronDownIcon className="h-4 w-4 text-gray-300" /> : <ChevronRightIcon className="h-4 w-4 text-gray-300" />}
          </div>
        </button>

        {open && (
          <div className="px-5 pb-4 pt-1 border-t border-gray-50 space-y-3">
            {run.summary && <p className="text-xs text-gray-500 leading-relaxed">{run.summary}</p>}

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
              <div>
                <p className="text-gray-400 uppercase text-[10px] tracking-wide mb-0.5">Data</p>
                {run.data_fingerprint ? (
                  <p className="text-gray-700">
                    <code>{run.data_fingerprint.md5}</code> · {run.data_fingerprint.n_rows.toLocaleString()} rows
                    {ch.data_changed && <span className="ml-1 text-amber-600 font-medium">(changed)</span>}
                  </p>
                ) : (
                  <p className="text-gray-300">not fingerprinted</p>
                )}
              </div>
              <div>
                <p className="text-gray-400 uppercase text-[10px] tracking-wide mb-0.5">Spec hash</p>
                <p className="text-gray-700"><code>{run.spec_hash ?? '—'}</code></p>
              </div>
              <div>
                <p className="text-gray-400 uppercase text-[10px] tracking-wide mb-0.5">Inference</p>
                <p className="text-gray-700">
                  {run.inference ? `${run.inference.chains}ch × ${run.inference.draws}dr` : '—'}
                </p>
              </div>
              <div>
                <p className="text-gray-400 uppercase text-[10px] tracking-wide mb-0.5">Parent run</p>
                <p className="text-gray-700"><code>{run.parent_run_id ?? '—'}</code></p>
              </div>
            </div>

            {!ch.baseline && ch.spec_changes.length > 0 && (
              <div>
                <p className="text-[10px] font-semibold text-gray-400 uppercase tracking-wide mb-1">
                  Spec changes vs previous run
                </p>
                <ul className="space-y-1">
                  {ch.spec_changes.map((c) => <SpecChangeRow key={c.path} c={c} />)}
                </ul>
              </div>
            )}

            {ch.assumptions_delta.length > 0 && (
              <div>
                <p className="text-[10px] font-semibold text-gray-400 uppercase tracking-wide mb-1">
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
                className="flex items-center gap-1 text-xs font-medium text-indigo-600 hover:underline"
              >
                Open session <ArrowRightIcon className="h-3 w-3" />
              </button>
              {run.report_path && (
                <span className="flex items-center gap-1 text-xs text-gray-400" title={run.report_path}>
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

export function RunsPage() {
  const { currentProjectId } = useProjectStore();
  const { data: runs = [], isLoading } = useQuery({
    queryKey: ['runs', currentProjectId],
    queryFn: () => runsService.listRuns(currentProjectId),
    staleTime: 30000,
  });

  return (
    <div className="space-y-6 max-w-4xl">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Model Runs</h1>
        <p className="text-sm text-gray-500 mt-0.5">
          Lineage of every fit — data fingerprint, spec changes, and the rationale behind each
          revision. Ask the agent for <code className="text-xs bg-gray-100 rounded px-1">get_run_history</code> to
          fold this into a final report.
        </p>
      </div>

      {isLoading ? (
        <p className="text-sm text-gray-400">Loading…</p>
      ) : runs.length === 0 ? (
        <div className="bg-white rounded-2xl border border-gray-200 px-6 py-10 text-center">
          <CircleStackIcon className="h-9 w-9 text-gray-200 mx-auto mb-2" />
          <p className="text-sm text-gray-400">No model runs recorded yet.</p>
          <p className="text-xs text-gray-300 mt-1 flex items-center justify-center gap-1">
            <BeakerIcon className="h-3.5 w-3.5" /> Fit a model in an agent session and it will appear here.
          </p>
        </div>
      ) : (
        <div className="relative">
          {/* timeline spine */}
          <div className="absolute left-[6px] top-6 bottom-6 w-px bg-gray-200" />
          {runs.map((r, i) => (
            <RunCard key={r.artifact_id} run={r} isLatest={i === 0} />
          ))}
        </div>
      )}
    </div>
  );
}

export default RunsPage;
