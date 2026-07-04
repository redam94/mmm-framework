import { useState } from 'react';
import { Button } from '../../components/ui';
import { useUpsertExperiment } from '../../api/hooks/usePortfolio';
import { useProjectStore } from '../../stores/projectStore';
import { OffPanelFields, emptyOffPanel, offPanelReadoutFields } from './OffPanelFields';
import type { ExperimentStatus } from '../../api/services/portfolioService';

/** Statuses the modal can log directly (calibration happens via the loop, not by hand). */
type LoggableStatus = 'draft' | 'planned' | 'running' | 'completed';

const inputCls =
  'w-full rounded-md border border-line-300 bg-white px-3 py-2 text-sm text-ink-900 ' +
  'placeholder:text-ink-300 focus:outline-none focus:ring-2 focus:ring-sage-600';

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="mb-1 block text-xs font-medium text-ink-600">{label}</label>
      {children}
    </div>
  );
}

export function LogExperimentModal({ onClose }: { onClose: () => void }) {
  const projectId = useProjectStore((s) => s.currentProjectId);
  const upsert = useUpsertExperiment();

  const [channel, setChannel] = useState('');
  const [subchannel, setSubchannel] = useState('');
  const [hypothesis, setHypothesis] = useState('');
  const [status, setStatus] = useState<LoggableStatus>('draft');
  const [designType, setDesignType] = useState('');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [estimand, setEstimand] = useState('roas');
  const [value, setValue] = useState('');
  const [se, setSe] = useState('');
  const [method, setMethod] = useState('');
  const [offPanel, setOffPanel] = useState(emptyOffPanel);
  const [notes, setNotes] = useState('');
  const [error, setError] = useState<string | null>(null);

  const measured = status === 'completed';

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!channel.trim()) return;
    setError(null);
    try {
      // A historical import often ran outside the fitted data window — the
      // off-panel readout fields (spend Δ/period, treated units, adstock
      // state) let the calibration evaluate the response curve at the test's
      // spend level instead of requiring window overlap.
      const offPanelFields = measured ? offPanelReadoutFields(offPanel) : {};
      const readout = measured
        ? {
            ...(value !== '' ? { value: Number(value) } : {}),
            ...(se !== '' ? { se: Number(se) } : {}),
            estimand,
            ...(startDate ? { start_date: startDate } : {}),
            ...(endDate ? { end_date: endDate } : {}),
            ...(method.trim() ? { method: method.trim() } : {}),
            ...offPanelFields,
          }
        : null;
      await upsert.mutateAsync({
        project_id: projectId,
        channel: channel.trim(),
        subchannel: subchannel.trim() || null,
        status: status as ExperimentStatus,
        design_type: designType.trim() || null,
        start_date: startDate || null,
        end_date: endDate || null,
        estimand: measured ? estimand : null,
        value: measured && value !== '' ? Number(value) : null,
        se: measured && se !== '' ? Number(se) : null,
        notes: notes.trim() || null,
        design: hypothesis.trim() ? { hypothesis: hypothesis.trim() } : null,
        readout,
      });
      onClose();
    } catch (err: unknown) {
      const e = err as { response?: { data?: { detail?: unknown } }; message?: unknown } | null;
      const detail = e?.response?.data?.detail;
      const message = typeof e?.message === 'string' ? e.message : 'Save failed';
      setError(typeof detail === 'string' ? detail : message);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-ink-900/40 p-4"
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <div className="w-full max-w-lg rounded-xl bg-white shadow-xl">
        <div className="border-b border-line-200 px-6 py-4">
          <h2 className="font-display text-lg font-semibold text-ink-900">Log experiment</h2>
          <p className="mt-0.5 text-xs text-ink-400">
            Track a lift test; completed results flag the model for a calibrated refit.
          </p>
        </div>
        <form onSubmit={handleSubmit} className="space-y-3 px-6 py-4">
          <div className="grid grid-cols-2 gap-3">
            <Field label="Channel *">
              <input
                autoFocus
                value={channel}
                onChange={(e) => setChannel(e.target.value)}
                placeholder="e.g. TV"
                className={inputCls}
              />
            </Field>
            <Field label="Status">
              <select
                value={status}
                onChange={(e) => setStatus(e.target.value as LoggableStatus)}
                className={inputCls}
              >
                <option value="draft">draft</option>
                <option value="planned">planned</option>
                <option value="running">running</option>
                <option value="completed">completed (measured)</option>
              </select>
            </Field>
          </div>
          <Field label="Sub-channel (creative/keyword — optional)">
            <input
              value={subchannel}
              onChange={(e) => setSubchannel(e.target.value)}
              placeholder="e.g. Brand, NonBrand, 30s-spot"
              className={inputCls}
            />
          </Field>
          <Field label="Hypothesis">
            <input
              value={hypothesis}
              onChange={(e) => setHypothesis(e.target.value)}
              placeholder="e.g. TV ROI is overstated by the model in Q4"
              className={inputCls}
            />
          </Field>
          <Field label="Design">
            <input
              value={designType}
              onChange={(e) => setDesignType(e.target.value)}
              placeholder="e.g. geo holdout, spend pulse"
              className={inputCls}
            />
          </Field>
          <div className="grid grid-cols-2 gap-3">
            <Field label="Start">
              <input
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                className={inputCls}
              />
            </Field>
            <Field label="End">
              <input
                type="date"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                className={inputCls}
              />
            </Field>
          </div>
          {(status === 'running' || status === 'completed') && (
            <p className="rounded-md bg-gold-100 px-3 py-2 text-xs text-gold-700">
              You're logging a test that skipped pre-registration — its design and analysis plan
              weren't locked before {status === 'completed' ? 'the readout' : 'launch'}, so the
              result is open to specification shopping. It still counts as evidence, but for the
              next test use the Design studio: lock the plan first, then run.
            </p>
          )}
          {measured && (
            <>
              <div className="grid grid-cols-3 gap-3">
                <Field label="Estimand">
                  <select
                    value={estimand}
                    onChange={(e) => setEstimand(e.target.value)}
                    className={inputCls}
                  >
                    <option value="roas">ROAS</option>
                    <option value="mroas">mROAS</option>
                    <option value="contribution">Contribution</option>
                  </select>
                </Field>
                <Field label="Value">
                  <input
                    type="number"
                    step="any"
                    value={value}
                    onChange={(e) => setValue(e.target.value)}
                    className={inputCls}
                  />
                </Field>
                <Field label="SE">
                  <input
                    type="number"
                    step="any"
                    min="0"
                    value={se}
                    onChange={(e) => setSe(e.target.value)}
                    className={inputCls}
                  />
                </Field>
              </div>
              <Field label="Method (optional)">
                <input
                  value={method}
                  onChange={(e) => setMethod(e.target.value)}
                  placeholder="e.g. geo holdout DiD, synthetic control"
                  className={inputCls}
                />
              </Field>
              <OffPanelFields state={offPanel} onChange={setOffPanel} inputCls={inputCls} />
            </>
          )}
          <Field label="Notes">
            <textarea
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              rows={2}
              className={`${inputCls} resize-none`}
            />
          </Field>

          {error && <p className="text-sm text-rust-600">{error}</p>}

          <div className="flex justify-end gap-2 pt-1">
            <Button type="button" variant="ghost" onClick={onClose}>
              Cancel
            </Button>
            <Button type="submit" disabled={!channel.trim() || upsert.isPending}>
              {upsert.isPending ? 'Saving…' : 'Save'}
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
}
