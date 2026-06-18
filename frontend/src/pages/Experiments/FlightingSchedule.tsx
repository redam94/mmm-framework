import { clsx } from 'clsx';
import Plot from 'react-plotly.js';
import { COLORS } from '../../theme/colors';
import { mmmPlotlyLayout, PLOTLY_CONFIG } from '../../theme/plotlyTheme';
import type { SchedulePoint } from '../../api/services/measurementService';

/** Multipliers within EPS of 1.0× count as "baseline" (no pulse). */
const EPS = 1e-3;

/**
 * Pull a flighting schedule out of a persisted experiment design blob, if one
 * is present. Pre-registered flighting designs store
 * `schedule: [{week_offset, multiplier}]` — the budget-neutral on/off (or
 * multi-level) spend pulses. Geo designs carry treatment/control groups instead
 * and have no schedule, so this returns null for them (and for malformed blobs).
 */
export function flightingSchedule(
  design: Record<string, any> | null | undefined,
): SchedulePoint[] | null {
  const raw = design?.schedule;
  if (!Array.isArray(raw) || raw.length === 0) return null;
  const pts = raw
    .filter(
      (p) =>
        p != null &&
        Number.isFinite(Number(p.week_offset)) &&
        Number.isFinite(Number(p.multiplier)),
    )
    .map((p) => ({
      week_offset: Number(p.week_offset),
      multiplier: Number(p.multiplier),
    }))
    .sort((a, b) => a.week_offset - b.week_offset);
  return pts.length > 0 ? pts : null;
}

/**
 * Compact at-a-glance flighting pattern — a tiny bar strip sized for kanban
 * cards. Bar height ∝ spend multiplier; sage = scaled up, steel = pulled down,
 * neutral = baseline. Reads as the planned pulse cadence without a full chart.
 */
export function FlightingStrip({ schedule }: { schedule: SchedulePoint[] }) {
  const max = Math.max(1, ...schedule.map((s) => s.multiplier));
  return (
    <div className="mt-1.5">
      <div className="flex h-6 items-end gap-px" aria-hidden="true">
        {schedule.map((s, i) => {
          const h = Math.max(10, Math.round((s.multiplier / max) * 100));
          const up = s.multiplier > 1 + EPS;
          const down = s.multiplier < 1 - EPS;
          return (
            <div
              key={i}
              className={clsx(
                'min-w-px flex-1 rounded-sm',
                up ? 'bg-sage-600' : down ? 'bg-steel-300' : 'bg-line-300',
              )}
              style={{ height: `${h}%` }}
            />
          );
        })}
      </div>
      <p className="mt-1 text-[10px] text-ink-400">
        flighting · <span className="num">{schedule.length}</span>w pulse pattern
      </p>
    </div>
  );
}

/**
 * Full planned flighting schedule for the detail drawer — a budget-neutral
 * spend-× bar chart over the test weeks, with the randomized-variance share when
 * the design recorded its identification diagnostics.
 */
export function FlightingScheduleChart({
  schedule,
  identification,
}: {
  schedule: SchedulePoint[];
  identification?: { exogenous_share?: number } | null;
}) {
  return (
    <div>
      <Plot
        data={[
          {
            x: schedule.map((s) => s.week_offset + 1),
            y: schedule.map((s) => s.multiplier),
            type: 'bar',
            marker: {
              color: schedule.map((s) =>
                s.multiplier > 1 + EPS ? COLORS.sage600 : COLORS.steel300,
              ),
            },
          } as any,
        ]}
        layout={mmmPlotlyLayout({
          height: 180,
          margin: { t: 10, l: 44, r: 16, b: 36 },
          xaxis: { title: { text: 'test week' } },
          yaxis: { title: { text: 'spend ×' } },
          showlegend: false,
        })}
        config={PLOTLY_CONFIG as any}
        style={{ width: '100%' }}
        useResizeHandler
      />
      {identification?.exogenous_share != null &&
        Number.isFinite(identification.exogenous_share) && (
          <p className="text-xs text-ink-400">
            <span className="num">
              {Math.round(identification.exogenous_share * 100)}%
            </span>{' '}
            of test-window spend variance is randomized (clean) — historical variance
            co-moves with demand and identifies nothing.
          </p>
        )}
    </div>
  );
}
