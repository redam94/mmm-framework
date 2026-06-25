import type {
  DeltaCell,
  RunComparison as RunComparisonData,
} from '../../api/services/runsService';

/**
 * Per-channel ROI/spend delta between two runs (B vs A) — the structured answer
 * to "why did this channel change since the last refresh?". Presentational: pass
 * the payload from `runsService.compareRuns`.
 */

function fmtNum(v: number | null, digits = 2): string {
  return v == null ? '—' : v.toFixed(digits);
}

function DeltaBadge({ cell }: { cell: DeltaCell }) {
  const d = cell.delta;
  if (d == null) return <span style={{ color: '#9a958a' }}>—</span>;
  const up = d > 0;
  const flat = d === 0;
  const color = flat ? '#9a958a' : up ? '#3a7d44' : '#b5654b';
  const arrow = flat ? '–' : up ? '▲' : '▼';
  return (
    <span style={{ color }}>
      {arrow} {fmtNum(Math.abs(d))}
    </span>
  );
}

export function RunComparison({ data }: { data: RunComparisonData }) {
  if (!data?.channels?.length) {
    return (
      <div className="text-sm text-muted" role="status">
        No channel metrics to compare.
      </div>
    );
  }
  return (
    <table className="w-full text-sm" aria-label="Run comparison">
      <thead>
        <tr>
          <th className="text-left">Channel</th>
          <th className="text-right">ROI (A)</th>
          <th className="text-right">ROI (B)</th>
          <th className="text-right">Δ ROI</th>
          <th className="text-right">Δ marginal ROI</th>
          <th className="text-right">Δ spend</th>
        </tr>
      </thead>
      <tbody>
        {data.channels.map((c) => (
          <tr key={c.channel}>
            <td className="text-left">
              {c.channel}
              {!c.in_a && <span title="added in B"> (new)</span>}
              {!c.in_b && <span title="removed in B"> (removed)</span>}
            </td>
            <td className="text-right">{fmtNum(c.roi_mean.a)}</td>
            <td className="text-right">{fmtNum(c.roi_mean.b)}</td>
            <td className="text-right">
              <DeltaBadge cell={c.roi_mean} />
            </td>
            <td className="text-right">
              <DeltaBadge cell={c.marginal_roi} />
            </td>
            <td className="text-right">
              <DeltaBadge cell={c.spend} />
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
