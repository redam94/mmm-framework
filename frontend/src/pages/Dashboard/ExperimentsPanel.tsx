import { useState } from 'react';
import {
  BeakerIcon,
  CheckBadgeIcon,
  PlusIcon,
  TrashIcon,
} from '@heroicons/react/24/outline';
import { usePortfolio, useUpsertExperiment, useDeleteExperiment } from '../../api/hooks/usePortfolio';
import type { ExperimentInfo, ExperimentStatus } from '../../api/services/portfolioService';

const STATUS_STYLE: Record<string, string> = {
  planned: 'bg-gray-100 text-gray-600',
  running: 'bg-blue-50 text-blue-700',
  completed: 'bg-amber-50 text-amber-700',
  calibrated: 'bg-emerald-50 text-emerald-700',
  cancelled: 'bg-gray-50 text-gray-400 line-through',
};

function StatusChip({ status }: { status: string }) {
  return (
    <span className={`px-2 py-0.5 rounded-full text-[11px] font-medium ${STATUS_STYLE[status] ?? STATUS_STYLE.planned}`}>
      {status === 'completed' ? 'completed · needs calibration' : status}
    </span>
  );
}

function LogExperimentModal({
  projectId,
  onClose,
}: {
  projectId: string | null;
  onClose: () => void;
}) {
  const upsert = useUpsertExperiment();
  const [channel, setChannel] = useState('');
  const [status, setStatus] = useState<ExperimentStatus>('planned');
  const [designType, setDesignType] = useState('');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [estimand, setEstimand] = useState('roas');
  const [value, setValue] = useState('');
  const [se, setSe] = useState('');
  const [notes, setNotes] = useState('');

  const measured = status === 'completed' || status === 'calibrated';

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!channel.trim()) return;
    await upsert.mutateAsync({
      project_id: projectId,
      channel: channel.trim(),
      status,
      design_type: designType.trim() || null,
      start_date: startDate || null,
      end_date: endDate || null,
      estimand: measured ? estimand : null,
      value: measured && value !== '' ? Number(value) : null,
      se: measured && se !== '' ? Number(se) : null,
      notes: notes.trim() || null,
    });
    onClose();
  };

  const inputCls =
    'w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500';

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-gray-900/40"
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-lg">
        <div className="px-6 py-4 border-b border-gray-100">
          <h2 className="text-lg font-semibold text-gray-900">Log Experiment</h2>
          <p className="text-xs text-gray-400 mt-0.5">
            Track a lift test; completed results flag the model for a calibrated refit.
          </p>
        </div>
        <form onSubmit={handleSubmit} className="px-6 py-4 space-y-3">
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs font-medium text-gray-600 mb-1">Channel *</label>
              <input autoFocus value={channel} onChange={(e) => setChannel(e.target.value)} placeholder="e.g. TV" className={inputCls} />
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-600 mb-1">Status</label>
              <select value={status} onChange={(e) => setStatus(e.target.value as ExperimentStatus)} className={inputCls}>
                <option value="planned">planned</option>
                <option value="running">running</option>
                <option value="completed">completed (measured)</option>
                <option value="calibrated">calibrated</option>
              </select>
            </div>
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-600 mb-1">Design</label>
            <input value={designType} onChange={(e) => setDesignType(e.target.value)} placeholder="e.g. geo holdout, spend pulse" className={inputCls} />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs font-medium text-gray-600 mb-1">Start</label>
              <input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} className={inputCls} />
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-600 mb-1">End</label>
              <input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} className={inputCls} />
            </div>
          </div>
          {measured && (
            <div className="grid grid-cols-3 gap-3">
              <div>
                <label className="block text-xs font-medium text-gray-600 mb-1">Estimand</label>
                <select value={estimand} onChange={(e) => setEstimand(e.target.value)} className={inputCls}>
                  <option value="roas">ROAS</option>
                  <option value="mroas">mROAS</option>
                  <option value="contribution">Contribution</option>
                </select>
              </div>
              <div>
                <label className="block text-xs font-medium text-gray-600 mb-1">Value</label>
                <input type="number" step="any" value={value} onChange={(e) => setValue(e.target.value)} className={inputCls} />
              </div>
              <div>
                <label className="block text-xs font-medium text-gray-600 mb-1">SE</label>
                <input type="number" step="any" min="0" value={se} onChange={(e) => setSe(e.target.value)} className={inputCls} />
              </div>
            </div>
          )}
          <div>
            <label className="block text-xs font-medium text-gray-600 mb-1">Notes</label>
            <textarea value={notes} onChange={(e) => setNotes(e.target.value)} rows={2} className={`${inputCls} resize-none`} />
          </div>
          <div className="flex gap-3 justify-end pt-1">
            <button type="button" onClick={onClose} className="px-4 py-2 text-sm text-gray-600 hover:text-gray-800">Cancel</button>
            <button
              type="submit"
              disabled={!channel.trim() || upsert.isPending}
              className="px-4 py-2 bg-indigo-600 text-white text-sm font-medium rounded-lg hover:bg-indigo-700 disabled:opacity-50"
            >
              {upsert.isPending ? 'Saving…' : 'Save'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

function ExperimentRow({ exp }: { exp: ExperimentInfo }) {
  const upsert = useUpsertExperiment();
  const del = useDeleteExperiment();
  const window =
    exp.start_date || exp.end_date ? `${exp.start_date ?? '?'} → ${exp.end_date ?? '?'}` : null;

  return (
    <li className="flex items-center gap-3 px-5 py-3 hover:bg-gray-50 transition-colors group">
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <p className="text-sm font-medium text-gray-800">{exp.channel}</p>
          <StatusChip status={exp.status} />
        </div>
        <p className="text-xs text-gray-400 mt-0.5 truncate">
          {exp.design_type && <span>{exp.design_type}</span>}
          {window && <span>{exp.design_type ? ' · ' : ''}{window}</span>}
          {exp.value != null && (
            <span className="text-gray-600 font-medium">
              {' · '}
              {exp.value} ± {exp.se ?? '?'} {exp.estimand ?? ''}
            </span>
          )}
        </p>
      </div>
      <div className="flex items-center gap-1.5 opacity-0 group-hover:opacity-100 transition-opacity">
        {exp.status === 'completed' && (
          <button
            title="Mark calibrated (after refitting with this result)"
            onClick={() => upsert.mutate({ id: exp.id, status: 'calibrated' })}
            className="flex items-center gap-1 text-xs text-emerald-600 font-medium hover:underline"
          >
            <CheckBadgeIcon className="h-3.5 w-3.5" /> Mark calibrated
          </button>
        )}
        <button
          title="Delete experiment"
          onClick={() => del.mutate(exp.id)}
          className="p-1 text-gray-300 hover:text-red-500 transition-colors"
        >
          <TrashIcon className="h-3.5 w-3.5" />
        </button>
      </div>
    </li>
  );
}

export function ExperimentsPanel({ projectId }: { projectId: string | null }) {
  const { data, isLoading } = usePortfolio(projectId);
  const [showLog, setShowLog] = useState(false);
  const experiments = data?.experiments ?? [];

  return (
    <section className="bg-white rounded-2xl border border-gray-200 shadow-sm overflow-hidden">
      <div className="flex items-center justify-between px-5 py-4 border-b border-gray-100">
        <div className="flex items-center gap-2">
          <BeakerIcon className="h-5 w-5 text-violet-500" />
          <h2 className="font-semibold text-gray-900">Experiments</h2>
        </div>
        <button
          onClick={() => setShowLog(true)}
          className="flex items-center gap-1.5 text-sm text-violet-600 hover:text-violet-700 font-medium transition-colors"
        >
          <PlusIcon className="h-3.5 w-3.5" /> Log experiment
        </button>
      </div>

      {isLoading ? (
        <div className="px-5 py-8 text-center text-sm text-gray-400">Loading…</div>
      ) : experiments.length === 0 ? (
        <div className="px-5 py-8 text-center">
          <BeakerIcon className="h-8 w-8 text-gray-200 mx-auto mb-2" />
          <p className="text-sm text-gray-400">No experiments tracked yet.</p>
          <p className="text-xs text-gray-300 mt-1">
            Ask the agent to recommend lift experiments, or log one manually.
          </p>
        </div>
      ) : (
        <ul className="divide-y divide-gray-50">
          {experiments.map((e) => (
            <ExperimentRow key={e.id} exp={e} />
          ))}
        </ul>
      )}

      {showLog && <LogExperimentModal projectId={projectId} onClose={() => setShowLog(false)} />}
    </section>
  );
}
