import { useMemo, useState } from 'react';
import { createPortal } from 'react-dom';
import { clsx } from 'clsx';
import { Button } from '../../components/ui';
import { useCreateProgram } from '../../api/hooks/useLearning';
import { ARM_SEP } from '../../api/services/learningService';
import type { CreateProgramRequest } from '../../api/services/learningService';
import { errorDetail } from './format';

const STEPS = ['Channels', 'Budget', 'Economics'] as const;

const INPUT_CLS =
  'w-full rounded-md border border-line-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-sage-600';

function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <label className="mb-1 block text-sm font-medium text-ink-700">{label}</label>
      {children}
      {hint && <p className="mt-1 text-xs text-ink-400">{hint}</p>}
    </div>
  );
}

function parseList(text: string): string[] {
  return text
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);
}

/**
 * 3-step wizard for a new continuous-learning program: channels (+ optional
 * creative/keyword arms) → budget, current spend, and the value of a KPI unit
 * → ENBS economics. POSTs the §3.1 config dict — all dollars are PER
 * GEO-PERIOD (spend for a single geo for one period).
 */
export function ProgramCreateWizard({
  projectId,
  onClose,
  onCreated,
}: {
  projectId: string | null;
  onClose: () => void;
  onCreated: (programId: string) => void;
}) {
  const create = useCreateProgram(projectId);

  const [step, setStep] = useState(0);
  const [name, setName] = useState('');
  const [channelsText, setChannelsText] = useState('');
  const [armsText, setArmsText] = useState<Record<string, string>>({});
  const [budget, setBudget] = useState('');
  const [valuePerUnit, setValuePerUnit] = useState('');
  const [center, setCenter] = useState<Record<string, string>>({});
  const [kpi, setKpi] = useState('');
  const [activation, setActivation] = useState<'hill' | 'logistic'>('hill');
  const [margin, setMargin] = useState('1.0');
  const [horizonPeriods, setHorizonPeriods] = useState('13');
  const [waveCost, setWaveCost] = useState('25000');
  const [error, setError] = useState<string | null>(null);

  const channels = useMemo(() => parseList(channelsText), [channelsText]);

  /** Arm-expanded surface dims — what budget/center apply to. */
  const dims = useMemo(() => {
    const out: string[] = [];
    for (const ch of channels) {
      const arms = parseList(armsText[ch] ?? '');
      if (arms.length >= 2) arms.forEach((a) => out.push(`${ch}${ARM_SEP}${a}`));
      else out.push(ch);
    }
    return out;
  }, [channels, armsText]);

  const budgetNum = Number(budget);
  const valueNum = Number(valuePerUnit);
  const defaultCenter =
    Number.isFinite(budgetNum) && budgetNum > 0 && dims.length > 0
      ? budgetNum / dims.length
      : null;

  const step0Valid = name.trim().length > 0 && channels.length >= 1;
  const step1Valid = Number.isFinite(budgetNum) && budgetNum > 0 && Number.isFinite(valueNum) && valueNum > 0;

  const busy = create.isPending;

  const handleCreate = async () => {
    if (!step0Valid || !step1Valid || busy) return;
    setError(null);

    const arms: Record<string, string[]> = {};
    for (const ch of channels) {
      const a = parseList(armsText[ch] ?? '');
      if (a.length >= 2) arms[ch] = a;
    }

    const centerNum: Record<string, number> = {};
    for (const dim of dims) {
      const v = Number(center[dim]);
      centerNum[dim] =
        Number.isFinite(v) && v > 0 ? v : Math.round(defaultCenter ?? 0);
    }

    const body: CreateProgramRequest = {
      name: name.trim(),
      config: {
        channels,
        ...(Object.keys(arms).length > 0 ? { arms } : {}),
        center: centerNum,
        budget: budgetNum,
        value_per_unit: valueNum,
        mode: 'fixed',
        activation,
        ...(kpi.trim() ? { kpi: kpi.trim() } : {}),
        margin: Number(margin) > 0 ? Number(margin) : 1.0,
        // horizon only — the backend computes population = n_geos × horizon
        horizon_periods: Math.max(1, Math.round(Number(horizonPeriods) || 13)),
        wave_cost: Math.max(0, Number(waveCost) || 0),
      },
    };

    try {
      const program = await create.mutateAsync(body);
      onCreated(program.id);
      onClose();
    } catch (e) {
      setError(errorDetail(e));
    }
  };

  return createPortal(
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-ink-900/40 p-4"
      onClick={(e) => e.target === e.currentTarget && !busy && onClose()}
    >
      <div className="flex max-h-[85vh] w-full max-w-lg flex-col overflow-hidden rounded-xl bg-white shadow-2xl">
        {/* Header + progress */}
        <div className="border-b border-line-200 px-6 py-4">
          <h2 className="font-display text-lg font-semibold text-ink-900">
            Start a learning program
          </h2>
          <p className="mt-0.5 text-xs text-ink-400">
            A response-surface bandit that learns how spend drives your KPI directly from
            designed geo waves — no fitted model required.
          </p>
          <div className="mt-3 flex items-center gap-2">
            {STEPS.map((label, i) => (
              <div key={label} className="flex items-center gap-2">
                <span
                  className={clsx(
                    'flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium transition-colors',
                    i === step
                      ? 'bg-sage-700 text-white'
                      : i < step
                        ? 'bg-sage-100 text-sage-800'
                        : 'bg-cream-200 text-ink-400',
                  )}
                >
                  <span className="num">{i + 1}</span>
                  <span className="hidden sm:inline">{label}</span>
                </span>
                {i < STEPS.length - 1 && <span className="h-px w-3 bg-line-300" />}
              </div>
            ))}
          </div>
        </div>

        {/* Step body */}
        <div className="flex-1 space-y-4 overflow-y-auto px-6 py-4 scrollbar-thin">
          {step === 0 && (
            <>
              <Field label="Program name">
                <input
                  autoFocus
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="e.g. FY27 Geo Learning Program"
                  className={INPUT_CLS}
                />
              </Field>
              <Field
                label="Channels"
                hint="Comma-separated — the spend dimensions the surface will learn over."
              >
                <input
                  value={channelsText}
                  onChange={(e) => setChannelsText(e.target.value)}
                  placeholder="e.g. Chatter, Pulse, Billboards"
                  className={INPUT_CLS}
                />
              </Field>
              {channels.length > 0 && (
                <div className="space-y-2">
                  <p className="text-xs text-ink-400">
                    Optional: split a channel into creative/keyword arms (comma-separated,
                    ≥2). Each extra arm adds ≈3 design cells per wave.
                  </p>
                  {channels.map((ch) => (
                    <div key={ch} className="flex items-center gap-2">
                      <span className="w-28 shrink-0 truncate text-sm text-ink-700">{ch}</span>
                      <input
                        value={armsText[ch] ?? ''}
                        onChange={(e) =>
                          setArmsText((a) => ({ ...a, [ch]: e.target.value }))
                        }
                        placeholder="arms, e.g. Brand, NonBrand"
                        className={INPUT_CLS}
                      />
                    </div>
                  ))}
                </div>
              )}
            </>
          )}

          {step === 1 && (
            <>
              <p className="text-xs text-ink-400">
                Enter spend for a single geo for one period — a $2M/week national budget
                across 50 geos is $40k per geo-week.
              </p>
              <div className="grid grid-cols-2 gap-3">
                <Field label="Budget ($/period per geo)">
                  <input
                    type="number"
                    min={0}
                    step="any"
                    value={budget}
                    onChange={(e) => setBudget(e.target.value)}
                    placeholder="e.g. 12000"
                    className={clsx(INPUT_CLS, 'num')}
                  />
                </Field>
                <Field label="Value per KPI unit ($)">
                  <input
                    type="number"
                    min={0}
                    step="any"
                    value={valuePerUnit}
                    onChange={(e) => setValuePerUnit(e.target.value)}
                    placeholder="e.g. 5.00"
                    className={clsx(INPUT_CLS, 'num')}
                  />
                </Field>
              </div>
              <div>
                <p className="mb-1 text-sm font-medium text-ink-700">
                  Current spend per channel ($/period per geo)
                </p>
                <p className="mb-2 text-xs text-ink-400">
                  The design centers waves on today's per-geo allocation. Blank = an even
                  split of the budget.
                </p>
                <div className="space-y-2">
                  {dims.map((dim) => (
                    <div key={dim} className="flex items-center gap-2">
                      <span className="w-40 shrink-0 truncate text-sm text-ink-700" title={dim}>
                        {dim}
                      </span>
                      <input
                        type="number"
                        min={0}
                        step="any"
                        value={center[dim] ?? ''}
                        onChange={(e) => setCenter((c) => ({ ...c, [dim]: e.target.value }))}
                        placeholder={
                          defaultCenter != null ? String(Math.round(defaultCenter)) : '—'
                        }
                        className={clsx(INPUT_CLS, 'num')}
                      />
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}

          {step === 2 && (
            <>
              <div className="grid grid-cols-2 gap-3">
                <Field label="Margin per KPI unit" hint="1.0 if the KPI is already profit.">
                  <input
                    type="number"
                    min={0}
                    step="any"
                    value={margin}
                    onChange={(e) => setMargin(e.target.value)}
                    className={clsx(INPUT_CLS, 'num')}
                  />
                </Field>
                <Field
                  label="Decision horizon (periods)"
                  hint="How long the learned allocation will be exploited — the value of learning scales by geos × periods."
                >
                  <input
                    type="number"
                    min={1}
                    step={1}
                    value={horizonPeriods}
                    onChange={(e) => setHorizonPeriods(e.target.value)}
                    className={clsx(INPUT_CLS, 'num')}
                  />
                </Field>
              </div>
              <Field
                label="Cost per wave ($)"
                hint="Ops + media inefficiency of running one more designed wave — the ENBS stopping rule weighs learning against this."
              >
                <input
                  type="number"
                  min={0}
                  step="any"
                  value={waveCost}
                  onChange={(e) => setWaveCost(e.target.value)}
                  className={clsx(INPUT_CLS, 'num')}
                />
              </Field>
              <div className="grid grid-cols-2 gap-3">
                <Field label="KPI name (optional)">
                  <input
                    value={kpi}
                    onChange={(e) => setKpi(e.target.value)}
                    placeholder="e.g. sales"
                    className={INPUT_CLS}
                  />
                </Field>
                <Field label="Response family">
                  <select
                    value={activation}
                    onChange={(e) => setActivation(e.target.value as 'hill' | 'logistic')}
                    className={INPUT_CLS}
                  >
                    <option value="hill">Hill (default)</option>
                    <option value="logistic">Logistic (concave)</option>
                  </select>
                </Field>
              </div>
            </>
          )}

          {error && (
            <p className="rounded-md border border-rust-600/30 bg-rust-100 px-3 py-2 text-sm text-rust-700">
              {error}
            </p>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-2 border-t border-line-200 px-6 py-4">
          <Button type="button" variant="ghost" onClick={onClose} disabled={busy}>
            Cancel
          </Button>
          {step > 0 && (
            <Button
              type="button"
              variant="secondary"
              onClick={() => setStep((s) => s - 1)}
              disabled={busy}
            >
              Back
            </Button>
          )}
          {step < STEPS.length - 1 ? (
            <Button
              type="button"
              onClick={() => setStep((s) => s + 1)}
              disabled={step === 0 ? !step0Valid : !step1Valid}
            >
              Next
            </Button>
          ) : (
            <Button
              type="button"
              onClick={handleCreate}
              disabled={!step0Valid || !step1Valid || busy}
            >
              {busy ? 'Creating…' : 'Start program'}
            </Button>
          )}
        </div>
      </div>
    </div>,
    document.body,
  );
}
