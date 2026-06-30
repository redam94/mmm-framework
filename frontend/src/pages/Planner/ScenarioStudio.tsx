import { useEffect, useMemo, useState } from 'react';
import { Loader2, Save, Wand2 } from 'lucide-react';
import { Button } from '../../components/ui';
import { usePlannerScenario } from '../../api/hooks/usePlanner';
import { useSaveBudgetPlan } from '../../api/hooks/useBudgetPlans';
import { fmtInt, fmtPct, fmtSignedInt } from './format';

const inputCls =
  'w-24 rounded-md border border-line-300 bg-white px-2 py-1.5 text-sm num focus:outline-none focus:ring-2 focus:ring-sage-600';

function errorDetail(e: unknown): string {
  const detail = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
  return detail ?? String(e);
}

/** What-if studio (B1/B3): set per-channel spend changes, see the KPI delta with
 * uncertainty, and save the scenario. */
export function ScenarioStudio({
  projectId,
  channels,
  modelId,
  onSaved,
}: {
  projectId: string;
  channels: string[];
  modelId?: string | null;
  onSaved?: () => void;
}) {
  const [pct, setPct] = useState<Record<string, number>>({});
  const [planName, setPlanName] = useState('');
  const [saved, setSaved] = useState(false);
  const scenario = usePlannerScenario(projectId);
  const save = useSaveBudgetPlan();

  useEffect(() => {
    scenario.reset();
    setSaved(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pct]);

  const result = scenario.job.data?.status === 'done' ? scenario.job.data.result : null;
  const jobStatus = scenario.job.data?.status ?? null;
  const running =
    scenario.start.isPending || jobStatus === 'pending' || jobStatus === 'running';
  const jobError =
    jobStatus === 'error'
      ? scenario.job.data?.error
      : scenario.start.isError
        ? errorDetail(scenario.start.error)
        : null;

  const spendChanges = useMemo(() => {
    const out: Record<string, number> = {};
    for (const [ch, v] of Object.entries(pct)) {
      if (v) out[ch] = v / 100; // % → fractional
    }
    return out;
  }, [pct]);

  const run = () => {
    if (Object.keys(spendChanges).length === 0) return;
    scenario.start.mutate({ spend_changes: spendChanges });
  };

  const onSave = async () => {
    if (!result) return;
    try {
      await save.mutateAsync({
        name: planName.trim() || `Scenario ${new Date().toLocaleDateString()}`,
        project_id: projectId,
        model_id: modelId ?? null,
        kind: 'scenario',
        spend_changes: result.spend_changes_applied,
        baseline_outcome: result.baseline_outcome,
        scenario_outcome: result.scenario_outcome,
        outcome_change: result.outcome_change,
        outcome_change_pct: result.outcome_change_pct,
        channel_details: result.channel_details,
        plan_payload: result as unknown as Record<string, unknown>,
      });
      setSaved(true);
      setPlanName('');
      onSaved?.();
    } catch {
      /* surfaced via save.isError */
    }
  };

  if (channels.length === 0) {
    return (
      <p className="text-sm text-ink-400">
        Channels become available once a baseline model is fit for this project.
      </p>
    );
  }

  return (
    <div className="space-y-4">
      <p className="text-xs text-ink-500">
        Set a spend change per channel (e.g. +20 or −10%); leave at 0 to hold.
      </p>
      <div className="space-y-1.5">
        {channels.map((ch) => (
          <div key={ch} className="flex items-center justify-between gap-3">
            <span className="text-sm text-ink-700">{ch}</span>
            <div className="flex items-center gap-1">
              <input
                type="number"
                step={5}
                value={pct[ch] ?? 0}
                onChange={(e) => setPct((p) => ({ ...p, [ch]: Number(e.target.value) }))}
                className={inputCls}
              />
              <span className="text-xs text-ink-400">%</span>
            </div>
          </div>
        ))}
      </div>

      <Button onClick={run} disabled={running || Object.keys(spendChanges).length === 0}>
        {running ? <Loader2 className="h-4 w-4 animate-spin" /> : <Wand2 className="h-4 w-4" />}
        {running ? 'Simulating…' : 'Run what-if'}
      </Button>

      {jobError && <p className="text-sm text-rust-600">{jobError}</p>}

      {result && (
        <div className="space-y-3 border-t border-line-200 pt-4">
          <div className="grid grid-cols-3 gap-3">
            <Metric label="Baseline KPI" value={fmtInt(result.baseline_outcome)} />
            <Metric label="Scenario KPI" value={fmtInt(result.scenario_outcome)} />
            <Metric
              label="Change"
              value={`${fmtSignedInt(result.outcome_change)} (${result.outcome_change_pct >= 0 ? '+' : ''}${result.outcome_change_pct.toFixed(1)}%)`}
            />
          </div>
          {result.outcome_change_hdi && (
            <p className="text-xs text-ink-500">
              {Math.round((result.hdi_prob ?? 0.94) * 100)}% interval [
              <span className="num">{fmtInt(result.outcome_change_hdi[0])}</span>,{' '}
              <span className="num">{fmtInt(result.outcome_change_hdi[1])}</span>]; P(beats
              baseline) = <span className="num">{fmtPct(result.prob_positive)}</span> (
              {result.n_draws} draws).
            </p>
          )}

          <div className="flex flex-wrap items-end gap-2 border-t border-line-200 pt-3">
            <label className="block grow">
              <span className="mb-1 block text-xs font-medium text-ink-700">
                Save this scenario as
              </span>
              <input
                type="text"
                value={planName}
                placeholder="e.g. TV +20% test"
                onChange={(e) => setPlanName(e.target.value)}
                className="w-full rounded-md border border-line-300 bg-white px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-sage-600"
              />
            </label>
            <Button variant="secondary" onClick={onSave} disabled={save.isPending}>
              <Save className="h-4 w-4" />
              {save.isPending ? 'Saving…' : 'Save scenario'}
            </Button>
          </div>
          {saved && <p className="text-sm text-sage-800">Saved — it's in your plans below.</p>}
          {save.isError && <p className="text-sm text-rust-600">{errorDetail(save.error)}</p>}
        </div>
      )}
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-line-200 bg-white px-3 py-2.5">
      <div className="text-[10px] font-semibold uppercase tracking-wider text-ink-400">
        {label}
      </div>
      <div className="mt-0.5 font-display text-lg font-semibold text-ink-900 num">{value}</div>
    </div>
  );
}
