import { useState } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';

/**
 * Off-panel calibration inputs for an experiment readout.
 *
 * When a test's window falls outside the model's fitted date range, the
 * calibration likelihood can't index training rows — instead it evaluates the
 * channel's global response curve at the test's spend level. That path needs
 * three extra readout fields (the same ones the agent's
 * `record_experiment_readout` tool accepts): `spend_per_period` (SIGNED $ delta
 * vs business-as-usual, per period per treated unit), `n_treated_units`, and
 * `adstock_state` ('steady_state' for an always-on test, 'cold_start' for a
 * burst launched from dark).
 */
export interface OffPanelState {
  /** signed per-period spend delta vs BAU (string form-state; '' = not set) */
  spendPerPeriod: string;
  /** number of treated geos/units (string form-state; '' → 1) */
  treatedUnits: string;
  adstockState: 'steady_state' | 'cold_start';
}

export const emptyOffPanel: OffPanelState = {
  spendPerPeriod: '',
  treatedUnits: '',
  adstockState: 'steady_state',
};

/**
 * The readout-dict fields this state contributes. Empty when no spend level
 * was entered — treated-units/adstock-state only mean something alongside one.
 */
export function offPanelReadoutFields(s: OffPanelState): Record<string, unknown> {
  if (s.spendPerPeriod === '' || !Number.isFinite(Number(s.spendPerPeriod))) return {};
  const units = s.treatedUnits !== '' ? Math.max(1, Math.round(Number(s.treatedUnits))) : 1;
  return {
    spend_per_period: Number(s.spendPerPeriod),
    n_treated_units: Number.isFinite(units) ? units : 1,
    adstock_state: s.adstockState,
  };
}

export function OffPanelFields({
  state,
  onChange,
  inputCls,
}: {
  state: OffPanelState;
  onChange: (next: OffPanelState) => void;
  /** blend into the host form's input styling */
  inputCls: string;
}) {
  const [open, setOpen] = useState(state.spendPerPeriod !== '');
  const Chevron = open ? ChevronDown : ChevronRight;

  return (
    <div className="rounded-md border border-line-200 bg-white/60">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="flex w-full items-center gap-1.5 px-2.5 py-2 text-left text-xs font-medium text-ink-600"
      >
        <Chevron className="h-3.5 w-3.5 shrink-0 text-ink-400" />
        Off-panel calibration
        <span className="font-normal text-ink-400">— test ran outside the model's data window</span>
      </button>
      {open && (
        <div className="space-y-2 px-2.5 pb-2.5">
          <div className="grid grid-cols-3 gap-2">
            <div>
              <label className="mb-1 block text-xs text-ink-600">Spend Δ / period ($)</label>
              <input
                type="number"
                step="any"
                value={state.spendPerPeriod}
                onChange={(e) => onChange({ ...state, spendPerPeriod: e.target.value })}
                placeholder="e.g. -5000"
                className={inputCls}
              />
            </div>
            <div>
              <label className="mb-1 block text-xs text-ink-600">Treated units</label>
              <input
                type="number"
                min="1"
                step="1"
                value={state.treatedUnits}
                onChange={(e) => onChange({ ...state, treatedUnits: e.target.value })}
                placeholder="1"
                className={inputCls}
              />
            </div>
            <div>
              <label className="mb-1 block text-xs text-ink-600">Adstock state</label>
              <select
                value={state.adstockState}
                onChange={(e) =>
                  onChange({
                    ...state,
                    adstockState: e.target.value as OffPanelState['adstockState'],
                  })
                }
                className={inputCls}
              >
                <option value="steady_state">steady state (always-on)</option>
                <option value="cold_start">cold start (burst from dark)</option>
              </select>
            </div>
          </div>
          <p className="text-xs leading-relaxed text-ink-400">
            Only needed when the test window falls outside the fitted dataset — calibration then
            evaluates the channel's response curve at this spend level instead of requiring window
            overlap. Spend is <em>signed</em> vs business-as-usual: a holdout/go-dark test is
            negative, a scale-up positive; per period, per treated unit, on the dataset's spend
            scale.
          </p>
        </div>
      )}
    </div>
  );
}
