import { useEffect, useMemo, useState } from 'react';
import { Check, Download, Loader2 } from 'lucide-react';
import { Button, Drawer, StatusChip } from '../../components/ui';
import { useExperimentRegistry } from '../../api/hooks/useMeasurement';
import { useIngestWave } from '../../api/hooks/useLearning';
import type { ExperimentRecord } from '../../api/services/measurementService';
import { errorDetail, fmtDollars, fmtNum } from './format';

/** Only measured readouts can become summary observations. */
const IMPORTABLE = new Set(['completed', 'calibrated']);

function importable(exp: ExperimentRecord): boolean {
  return IMPORTABLE.has(exp.status);
}

/**
 * Import past experiments as evidence — the model-free bridge: completed /
 * calibrated registry readouts become summary observations on the learning
 * program's response surface (no MMM, no panel required).
 */
export function ImportExperimentsPanel({
  open,
  onClose,
  projectId,
  programId,
}: {
  open: boolean;
  onClose: () => void;
  projectId: string | null;
  programId: string;
}) {
  const registry = useExperimentRegistry(open ? projectId : null);
  const ingest = useIngestWave(projectId, programId);

  const [selected, setSelected] = useState<Set<string>>(new Set());

  useEffect(() => {
    if (open) {
      setSelected(new Set());
      ingest.reset();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, programId]);

  const candidates = useMemo(
    () => (registry.data ?? []).filter(importable),
    [registry.data],
  );

  const toggle = (id: string) =>
    setSelected((s) => {
      const next = new Set(s);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  const jobStatus = ingest.job.data?.status ?? null;
  const running =
    ingest.start.isPending || jobStatus === 'pending' || jobStatus === 'running';
  const jobResult = jobStatus === 'done' ? ingest.job.data?.result ?? null : null;
  const jobError = ingest.start.isError
    ? errorDetail(ingest.start.error)
    : jobStatus === 'error'
      ? ingest.job.data?.error ?? 'Import failed'
      : null;

  return (
    <Drawer open={open} onClose={onClose} title="Import past experiments" width="max-w-xl">
      <div className="space-y-4">
        <p className="text-sm text-ink-600">
          Completed and calibrated lift-test readouts become summary observations on the
          response surface — evidence with no model and no panel required. mROAS readouts
          are slopes, not lifts, so the server will skip them (with a reason).
        </p>

        {registry.isLoading ? (
          <p className="py-6 text-center text-sm text-ink-400">Loading experiments…</p>
        ) : candidates.length === 0 ? (
          <p className="rounded-lg border border-dashed border-line-300 bg-cream-200/60 px-4 py-6 text-center text-sm text-ink-400">
            No completed or calibrated experiments in this project's registry yet.
          </p>
        ) : (
          <ul className="divide-y divide-line-200 rounded-lg border border-line-200 bg-white">
            {candidates.map((exp) => {
              const on = selected.has(exp.id);
              return (
                <li key={exp.id} className="flex items-center gap-3 px-3 py-2.5">
                  <input
                    type="checkbox"
                    checked={on}
                    onChange={() => toggle(exp.id)}
                    disabled={running}
                    className="h-4 w-4 accent-sage-700"
                  />
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-1.5">
                      <span className="text-sm font-medium text-ink-900">{exp.channel}</span>
                      {exp.subchannel && (
                        <span className="rounded-full bg-steel-100 px-1.5 py-0.5 text-[10px] font-medium text-steel-700">
                          {exp.subchannel}
                        </span>
                      )}
                      <StatusChip status={exp.status} />
                    </div>
                    <p className="mt-0.5 truncate text-xs text-ink-400">
                      {exp.estimand && <span>{exp.estimand} </span>}
                      {exp.value != null && (
                        <span className="num">
                          {fmtNum(exp.value)} ± {exp.se != null ? fmtNum(exp.se) : '?'}
                        </span>
                      )}
                      {(exp.start_date || exp.end_date) && (
                        <span className="num">
                          {' '}· {exp.start_date ?? '?'} → {exp.end_date ?? '?'}
                        </span>
                      )}
                    </p>
                  </div>
                </li>
              );
            })}
          </ul>
        )}

        {jobError && <p className="text-sm text-rust-600">{jobError}</p>}

        {jobResult ? (
          <div className="space-y-1.5 rounded-md border border-sage-300 bg-sage-100/60 px-3 py-2.5 text-sm text-sage-800">
            <p className="flex items-center gap-1.5 font-medium">
              <Check className="h-4 w-4" />
              {jobResult.imported != null
                ? `Imported ${jobResult.imported} readout${jobResult.imported === 1 ? '' : 's'} — surface refit.`
                : 'Import complete — surface refit.'}
            </p>
            <p className="text-xs">
              E[regret] now{' '}
              <span className="num font-medium">
                {fmtDollars(jobResult.regret.e_regret_dollars)}
              </span>{' '}
              · ENBS <span className="num font-medium">{fmtDollars(jobResult.regret.enbs)}</span>.
              The program page has the full readout.
            </p>
            {jobResult.skipped != null && jobResult.skipped.length > 0 && (
              <div className="text-xs text-gold-700">
                <p className="font-medium">
                  Skipped {jobResult.skipped.length}
                  {':'}
                </p>
                <ul className="mt-0.5 list-inside list-disc">
                  {jobResult.skipped.map((s) => (
                    <li key={s.id}>
                      <span className="num">{s.id}</span> — {s.reason}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        ) : (
          <div className="flex items-center gap-3">
            <Button
              onClick={() => ingest.start.mutate({ experiment_ids: Array.from(selected) })}
              disabled={selected.size === 0 || running || !projectId}
            >
              {running ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" /> Importing & refitting…
                </>
              ) : (
                <>
                  <Download className="h-4 w-4" /> Import {selected.size || ''} experiment
                  {selected.size === 1 ? '' : 's'}
                </>
              )}
            </Button>
            {running && (
              <span className="text-xs text-ink-400">
                Converting readouts to summary observations and refitting…
              </span>
            )}
          </div>
        )}
      </div>
    </Drawer>
  );
}
