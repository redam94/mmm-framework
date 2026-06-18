import { clsx } from 'clsx';
import { LockClosedIcon } from '@heroicons/react/24/outline';
import { StatusChip } from '../../components/ui';
import type { ExperimentRecord, LifecycleStatus } from '../../api/services/measurementService';

/** Past planning without a locked design = vulnerable to specification shopping. */
function isAdHoc(exp: ExperimentRecord): boolean {
  return (
    exp.preregistered_at == null &&
    ['running', 'completed', 'calibrated'].includes(exp.status)
  );
}

const COLUMNS: { id: LifecycleStatus; label: string }[] = [
  { id: 'draft', label: 'Draft' },
  { id: 'planned', label: 'Planned' },
  { id: 'running', label: 'Running' },
  { id: 'completed', label: 'Completed' },
  { id: 'calibrated', label: 'Calibrated' },
];

function ExperimentCard({
  exp,
  onOpen,
}: {
  exp: ExperimentRecord;
  onOpen: (id: string) => void;
}) {
  const readoutValue = (exp.readout?.value ?? exp.value) as number | null | undefined;
  const readoutSe = (exp.readout?.se ?? exp.se) as number | null | undefined;
  const window =
    exp.start_date || exp.end_date
      ? `${exp.start_date ?? '?'} → ${exp.end_date ?? '?'}`
      : null;

  return (
    <button
      onClick={() => onOpen(exp.id)}
      className={clsx(
        'w-full rounded-lg border border-line-200 bg-white p-3 text-left shadow-sm',
        'transition-colors hover:border-sage-300 focus-visible:outline focus-visible:outline-2 focus-visible:outline-sage-700',
      )}
    >
      <div className="flex items-start justify-between gap-2">
        <span className="font-medium text-ink-900">{exp.channel}</span>
        <StatusChip status={exp.status} />
      </div>
      {exp.design_type && <p className="mt-1 text-xs text-ink-400">{exp.design_type}</p>}
      {exp.preregistered_at != null ? (
        <p className="mt-1 flex items-center gap-1 text-[10px] font-medium text-sage-700">
          <LockClosedIcon className="h-3 w-3" /> pre-registered
        </p>
      ) : isAdHoc(exp) ? (
        <p
          className="mt-1 inline-block rounded-full bg-gold-100 px-1.5 py-0.5 text-[10px] font-medium text-gold-700"
          title="Design was never locked before the readout — treat the result as observational evidence"
        >
          ad-hoc (no locked design)
        </p>
      ) : null}
      {window && <p className="mt-1 text-xs text-ink-600 num">{window}</p>}
      {readoutValue != null && (
        <p className="mt-1.5 text-xs text-ink-700">
          <span className="num">
            {readoutValue.toFixed(2)} ± {readoutSe != null ? readoutSe.toFixed(2) : '?'}
          </span>
          {exp.estimand && <span className="text-ink-400"> {exp.estimand}</span>}
        </p>
      )}
    </button>
  );
}

export function LifecycleBoard({
  experiments,
  onOpen,
}: {
  experiments: ExperimentRecord[];
  onOpen: (id: string) => void;
}) {
  const inactive = experiments.filter(
    (e) => e.status === 'abandoned' || e.status === 'cancelled',
  );

  return (
    <div className="space-y-3">
      <div className="-mx-1 overflow-x-auto px-1 pb-2">
        <div className="flex min-w-[900px] gap-3">
          {COLUMNS.map((col) => {
            const items = experiments.filter((e) => e.status === col.id);
            return (
              <div
                key={col.id}
                className="flex min-w-[180px] flex-1 flex-col rounded-lg border border-line-200 bg-cream-100 p-2.5"
              >
                <div className="mb-2 flex items-center justify-between px-1">
                  <span className="text-xs font-semibold uppercase tracking-wider text-ink-600">
                    {col.label}
                  </span>
                  <span className="num rounded-full bg-cream-200 px-1.5 py-0.5 text-[10px] font-semibold text-ink-600">
                    {items.length}
                  </span>
                </div>
                <div className="space-y-2">
                  {items.length === 0 ? (
                    <p className="px-1 py-4 text-center text-xs text-ink-300">—</p>
                  ) : (
                    items.map((e) => <ExperimentCard key={e.id} exp={e} onOpen={onOpen} />)
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {inactive.length > 0 && (
        <p className="text-xs text-ink-400">
          <span className="num rounded-full bg-cream-200 px-1.5 py-0.5 font-semibold text-ink-600">
            {inactive.length}
          </span>{' '}
          abandoned or cancelled — open an experiment's audit trail for details.
        </p>
      )}
    </div>
  );
}
