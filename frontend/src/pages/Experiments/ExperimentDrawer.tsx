import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { clsx } from 'clsx';
import { Lock } from 'lucide-react';
import { Button, Card, Drawer, StatusChip } from '../../components/ui';
import { useExperimentRecord, useTransitionExperiment } from '../../api/hooks/useMeasurement';
import { FlightingScheduleChart, flightingSchedule } from './FlightingSchedule';
import { designMDE, designPower, powerBasisLabel, powerTextColor } from './designSummary';
import type {
  ExperimentRecord,
  ExperimentTransition,
} from '../../api/services/measurementService';

const inputCls =
  'w-full rounded-md border border-line-300 bg-white px-2.5 py-1.5 text-sm text-ink-900 ' +
  'focus:outline-none focus:ring-2 focus:ring-sage-600';

function fmtUnix(ts: number | null | undefined): string {
  return ts ? new Date(ts * 1000).toLocaleDateString() : '—';
}

function Label({ children }: { children: React.ReactNode }) {
  return (
    <p className="mb-1.5 text-xs font-semibold uppercase tracking-wider text-ink-400">{children}</p>
  );
}

function DefRow({ term, children }: { term: string; children: React.ReactNode }) {
  return (
    <div className="flex items-baseline gap-3 py-1">
      <dt className="w-36 shrink-0 text-xs text-ink-400">{term}</dt>
      <dd className="min-w-0 text-sm text-ink-700">{children}</dd>
    </div>
  );
}

function MonoChip({ value }: { value: string }) {
  return (
    <span className="num inline-block max-w-full truncate rounded bg-cream-200 px-1.5 py-0.5 align-bottom text-xs text-ink-700">
      {value}
    </span>
  );
}

function ReadoutForm({
  onSubmit,
  pending,
  targetSe,
}: {
  onSubmit: (body: ExperimentTransition) => void;
  pending: boolean;
  /** the design's pre-registered precision target, when one was locked */
  targetSe?: number | null;
}) {
  const [value, setValue] = useState('');
  const [se, setSe] = useState('');
  const [estimand, setEstimand] = useState('roas');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');

  const seNum = se !== '' ? Number(se) : null;
  const underPowered =
    targetSe != null && seNum != null && Number.isFinite(seNum) && seNum > targetSe;

  return (
    <div className="space-y-2.5 rounded-lg border border-line-200 bg-cream-100 p-3">
      <div className="grid grid-cols-3 gap-2">
        <div>
          <label className="mb-1 block text-xs text-ink-600">Value</label>
          <input type="number" step="any" value={value} onChange={(e) => setValue(e.target.value)} className={inputCls} />
        </div>
        <div>
          <label className="mb-1 block text-xs text-ink-600">SE</label>
          <input type="number" step="any" min="0" value={se} onChange={(e) => setSe(e.target.value)} className={inputCls} />
        </div>
        <div>
          <label className="mb-1 block text-xs text-ink-600">Estimand</label>
          <select value={estimand} onChange={(e) => setEstimand(e.target.value)} className={inputCls}>
            <option value="roas">ROAS</option>
            <option value="mroas">mROAS</option>
            <option value="contribution">Contribution</option>
          </select>
        </div>
      </div>
      <div className="grid grid-cols-2 gap-2">
        <div>
          <label className="mb-1 block text-xs text-ink-600">Window start</label>
          <input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} className={inputCls} />
        </div>
        <div>
          <label className="mb-1 block text-xs text-ink-600">Window end</label>
          <input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} className={inputCls} />
        </div>
      </div>
      {underPowered && (
        <p className="rounded-md bg-gold-100 px-2.5 py-1.5 text-xs text-gold-700">
          Measured SE <span className="num">{seNum!.toFixed(2)}</span> exceeds the design's
          target SE <span className="num">{targetSe!.toFixed(2)}</span> — the test came in
          under-powered vs its pre-registered MDE, so calibration will move the posterior less
          than planned. Record it anyway (it's still evidence), but don't read a null as proof
          of no effect.
        </p>
      )}
      <div className="flex justify-end">
        <Button
          size="sm"
          disabled={value === '' || pending}
          onClick={() =>
            onSubmit({
              status: 'completed',
              value: Number(value),
              se: se !== '' ? Number(se) : undefined,
              estimand,
              start_date: startDate || undefined,
              end_date: endDate || undefined,
            })
          }
        >
          {pending ? 'Saving…' : 'Save readout'}
        </Button>
      </div>
    </div>
  );
}

function DrawerBody({ exp }: { exp: ExperimentRecord }) {
  const navigate = useNavigate();
  const transition = useTransitionExperiment();
  const [error, setError] = useState<string | null>(null);
  const [showReadoutForm, setShowReadoutForm] = useState(false);

  const doTransition = async (body: ExperimentTransition) => {
    setError(null);
    try {
      await transition.mutateAsync({ id: exp.id, body });
      setShowReadoutForm(false);
    } catch (e: unknown) {
      const err = e as { response?: { data?: { detail?: unknown } }; message?: string };
      const detail = err?.response?.data?.detail;
      setError(typeof detail === 'string' ? detail : (err?.message ?? 'Transition failed'));
    }
  };

  const design = exp.design ?? {};
  const schedule = flightingSchedule(design);
  const mde = designMDE(design);
  const power = designPower(design);
  const hasDesign = Object.keys(design).length > 0;
  const readout = exp.readout ?? null;
  const priority = exp.priority ?? null;
  const active = ['draft', 'planned', 'running'].includes(exp.status);

  const readoutValue = (readout?.value ?? exp.value) as number | null;
  const readoutSe = (readout?.se ?? exp.se) as number | null;
  const readoutEstimand = (readout?.estimand ?? exp.estimand) as string | null;
  const readoutMethod = readout?.method as string | undefined;
  const windowStart = (readout?.start_date ?? exp.start_date) as string | null;
  const windowEnd = (readout?.end_date ?? exp.end_date) as string | null;

  return (
    <div className="space-y-6">
      {/* Status header */}
      <div>
        <div className="flex items-center gap-3">
          <h3 className="font-display text-xl font-semibold text-ink-900">{exp.channel}</h3>
          <StatusChip status={exp.status} />
        </div>
        <p className="mt-1 text-xs text-ink-400">
          created <span className="num">{fmtUnix(exp.created_at)}</span> · updated{' '}
          <span className="num">{fmtUnix(exp.updated_at)}</span>
        </p>
      </div>

      {/* Hypothesis & design */}
      <div>
        <Label>Hypothesis & design</Label>
        <Card padding="sm">
          <dl className="divide-y divide-line-200">
            {design.hypothesis != null && <DefRow term="Hypothesis">{String(design.hypothesis)}</DefRow>}
            {exp.subchannel && (
              <DefRow term="Sub-channel">
                <span className="rounded-full bg-steel-100 px-2 py-0.5 text-xs font-medium text-steel-700">
                  {exp.subchannel}
                </span>
                <span className="ml-1.5 text-xs text-ink-400">creative/keyword arm</span>
              </DefRow>
            )}
            <DefRow term="Design type">{design.design_type ?? exp.design_type ?? '—'}</DefRow>
            {design.min_duration_periods != null && (
              <DefRow term="Min duration">
                <span className="num">{design.min_duration_periods}</span> periods
              </DefRow>
            )}
            {design.target_se != null && (
              <DefRow term="Target SE">
                <span className="num">{Number(design.target_se).toFixed(2)}</span>
              </DefRow>
            )}
            {mde != null && (
              <DefRow term="MDE (ROAS)">
                <span className="num">{mde.toFixed(2)}</span>
                <span className="ml-1.5 text-ink-400">@ 80% power</span>
              </DefRow>
            )}
            {power.power != null && (
              <DefRow term="Power vs expected">
                <span
                  className={clsx('num', powerTextColor(power.power, power.verdict))}
                  title={powerBasisLabel(power.basis)}
                >
                  {Math.round(power.power * 100)}%
                </span>
                {power.verdict && (
                  <span className="ml-1.5 text-ink-400">{power.verdict}</span>
                )}
                {/* The assurance fallback is a different quantity from the MDE
                    indicator — qualify it so the % isn't read as the same thing. */}
                {power.basis === 'assurance' && (
                  <span className="ml-1.5 text-ink-400">(assurance)</span>
                )}
                {power.recommendedDuration != null && (
                  <span className="ml-1.5 text-ink-400">
                    · reaches power ≈ <span className="num">{power.recommendedDuration}</span>w
                  </span>
                )}
              </DefRow>
            )}
            {design.why != null && <DefRow term="Why">{String(design.why)}</DefRow>}
          </dl>
          {mde != null && power.power == null && hasDesign && (
            <p className="mt-2 border-t border-line-200 pt-2 text-xs text-ink-400">
              MDE is the smallest ROAS effect detectable at 80% power. Power vs the model's
              expected effect wasn't captured —{' '}
              {exp.preregistered_at != null
                ? 're-run the model-backed simulation in the design studio to anchor the next test.'
                : 'run the model-backed simulation in the design studio before pre-registering to anchor it.'}
            </p>
          )}
          {exp.preregistered_at != null ? (
            <p className="mt-2 flex items-center gap-1.5 border-t border-line-200 pt-2 text-xs text-sage-700">
              <Lock className="h-3.5 w-3.5" />
              Design locked (pre-registered) <span className="num">{fmtUnix(exp.preregistered_at)}</span>
            </p>
          ) : ['running', 'completed', 'calibrated'].includes(exp.status) ? (
            <p className="mt-2 border-t border-line-200 pt-2 text-xs text-gold-700">
              No locked design — this test went live without pre-registration, so its analysis
              plan wasn't fixed before the readout. Treat the result as observational evidence
              and pre-register the next test in the design studio.
            </p>
          ) : null}
        </Card>
      </div>

      {/* Planned flighting schedule */}
      {schedule && (
        <div>
          <Label>Planned flighting pattern</Label>
          <Card padding="sm">
            <FlightingScheduleChart
              schedule={schedule}
              identification={
                (design.identification as { exogenous_share?: number } | undefined) ?? null
              }
            />
            <p className="mt-1 text-xs text-ink-400">
              Budget-neutral spend pulses locked at pre-registration —{' '}
              <span className="num">{schedule.length}</span> test weeks. Sage bars scale
              spend up, steel bars pull it down.
            </p>
          </Card>
        </div>
      )}

      {/* Priority snapshot */}
      {priority && (
        <div>
          <Label>Priority snapshot</Label>
          <div className="flex flex-wrap gap-2">
            {priority.eig != null && (
              <span className="rounded-full bg-steel-100 px-2.5 py-1 text-xs text-steel-700">
                EIG <span className="num">{Number(priority.eig).toFixed(2)}</span> nats
              </span>
            )}
            {priority.evoi != null && (
              <span className="rounded-full bg-steel-100 px-2.5 py-1 text-xs text-steel-700">
                EVOI <span className="num">{Math.round(Number(priority.evoi)).toLocaleString()}</span>
              </span>
            )}
            {priority.quadrant != null && (
              <span className="rounded-full bg-sage-100 px-2.5 py-1 text-xs text-sage-800">
                {String(priority.quadrant).replace(/_/g, ' ')}
              </span>
            )}
            {priority.priority != null && (
              <span className="rounded-full bg-cream-200 px-2.5 py-1 text-xs text-ink-600">
                priority <span className="num">{Number(priority.priority).toFixed(2)}</span>
              </span>
            )}
          </div>
        </div>
      )}

      {/* Readout */}
      <div>
        <Label>Readout</Label>
        {readoutValue != null ? (
          <Card padding="sm">
            <p className="text-lg text-ink-900">
              <span className="num">
                {readoutValue.toFixed(2)} ± {readoutSe != null ? readoutSe.toFixed(2) : '?'}
              </span>
              {readoutEstimand && <span className="ml-2 text-sm text-ink-400">{readoutEstimand}</span>}
            </p>
            {(windowStart || windowEnd) && (
              <p className="mt-1 text-xs text-ink-600">
                window <span className="num">{windowStart ?? '?'} → {windowEnd ?? '?'}</span>
              </p>
            )}
            {readoutMethod && <p className="mt-1 text-xs text-ink-400">method: {readoutMethod}</p>}
          </Card>
        ) : (
          <p className="text-sm text-ink-400">No readout yet.</p>
        )}
      </div>

      {/* Calibration linkage */}
      <div>
        <Label>Calibration linkage</Label>
        <dl>
          <DefRow term="Recommended by run">
            {exp.recommending_run_id ? <MonoChip value={exp.recommending_run_id} /> : '—'}
          </DefRow>
          <DefRow term="Calibrated into run">
            {exp.calibrated_run_id ? <MonoChip value={exp.calibrated_run_id} /> : '—'}
          </DefRow>
        </dl>
        <p className="mt-1 text-xs text-ink-400">
          The loop: a fitted run (T₀) recommends the test; its readout calibrates the next refit
          (T₁), closing the measure → learn → refit cycle.
        </p>
      </div>

      {/* Audit trail */}
      <div>
        <Label>Audit trail</Label>
        {exp.status_history.length === 0 ? (
          <p className="text-sm text-ink-400">No history recorded.</p>
        ) : (
          <ol className="space-y-3 border-l-2 border-line-300 pl-4">
            {exp.status_history.map((h, i) => (
              <li key={`${h.status}-${h.at}-${i}`} className="relative">
                <span className="absolute -left-[21.5px] top-1.5 h-2 w-2 rounded-full bg-line-400" />
                <div className="flex items-center gap-2">
                  <StatusChip status={h.status} />
                  <span className="num text-xs text-ink-400">{fmtUnix(h.at)}</span>
                </div>
                {h.note && <p className="mt-0.5 text-xs text-ink-600">{h.note}</p>}
              </li>
            ))}
          </ol>
        )}
      </div>

      {/* Actions */}
      <div className="space-y-3 border-t border-line-200 pt-4">
        {error && <p className="text-sm text-rust-600">{error}</p>}

        {exp.status === 'completed' && (
          <p className="rounded-lg bg-cream-100 px-3 py-2 text-xs text-ink-600">
            Calibrate via the Workspace: ask the agent to apply_experiment_calibration and refit.
          </p>
        )}
        {exp.status === 'completed' && (
          <Button
            onClick={() =>
              navigate('/workspace', {
                state: {
                  prefill:
                    `Calibrate experiment ${exp.id} (${exp.channel}` +
                    `${exp.subchannel ? ` / ${exp.subchannel}` : ''}) into the model: ` +
                    `run apply_experiment_calibration with experiment id "${exp.id}", ` +
                    `review the staged measurement, then refit with fit_mmm_model so ` +
                    `the readout enters the likelihood.`,
                },
              })
            }
          >
            Calibrate in Workspace
          </Button>
        )}

        {exp.status === 'running' && showReadoutForm && (
          <ReadoutForm
            onSubmit={doTransition}
            pending={transition.isPending}
            targetSe={design.target_se != null ? Number(design.target_se) : null}
          />
        )}

        <div className="flex flex-wrap items-center gap-2">
          {exp.status === 'draft' && (
            <Button
              disabled={transition.isPending}
              onClick={() => doTransition({ status: 'planned' })}
            >
              Pre-register
            </Button>
          )}
          {exp.status === 'planned' && (
            <Button
              disabled={transition.isPending}
              onClick={() => doTransition({ status: 'running' })}
            >
              Mark running
            </Button>
          )}
          {exp.status === 'running' && !showReadoutForm && (
            <Button onClick={() => setShowReadoutForm(true)}>Record readout…</Button>
          )}
          {active && (
            <Button
              variant="danger"
              disabled={transition.isPending}
              onClick={() => {
                if (window.confirm('Abandon this experiment? Its history is kept for the audit trail.')) {
                  void doTransition({ status: 'abandoned' });
                }
              }}
            >
              Abandon
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}

export function ExperimentDrawer({
  experimentId,
  onClose,
}: {
  experimentId: string | null;
  onClose: () => void;
}) {
  const { data: exp, isLoading, isError } = useExperimentRecord(experimentId);

  return (
    <Drawer open={!!experimentId} onClose={onClose} title="Experiment" width="max-w-xl">
      {isLoading ? (
        <p className="py-8 text-center text-sm text-ink-400">Loading…</p>
      ) : isError || !exp ? (
        <p className="py-8 text-center text-sm text-rust-600">
          Couldn't load this experiment — it may have been deleted.
        </p>
      ) : (
        <DrawerBody exp={exp} />
      )}
    </Drawer>
  );
}
