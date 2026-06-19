import { clsx } from 'clsx';
import { Card } from '../../components/ui';
import type { ChannelBenchmark as ChannelBenchmarkRow } from '../../api/services/benchmarkService';
import { fmtRoi } from './helpers';

interface Props {
  channels: ChannelBenchmarkRow[];
  /** Per-channel scatter of each brand's ROI, keyed by channel name. */
  dots: Record<string, { name: string; roi: number; percentile: number | null }[]>;
}

/** Cross-brand ROI distribution per channel: a P25–P75 box with min/max whiskers,
 *  a median tick, and one dot per brand — so a brand can be read against its peers. */
export function ChannelBenchmark({ channels, dots }: Props) {
  if (channels.length === 0) {
    return (
      <Card padding="lg" tone="cream">
        <p className="text-sm text-ink-400">
          No shared channels across brands yet — fit at least one model with run metrics to populate the benchmark.
        </p>
      </Card>
    );
  }

  const domainMin = Math.min(0, ...channels.map((c) => c.roi_min));
  const domainMax = Math.max(...channels.map((c) => c.roi_max), domainMin + 1);
  const span = domainMax - domainMin || 1;
  const pct = (v: number) => Math.max(0, Math.min(100, ((v - domainMin) / span) * 100));

  return (
    <Card padding="lg">
      {/* Domain axis */}
      <div className="mb-3 flex items-center gap-3 pl-32 pr-24 text-[11px] tabular-nums text-ink-400">
        <span>{fmtRoi(domainMin)}</span>
        <div className="relative h-px flex-1 bg-line-200">
          <span className="absolute left-1/2 top-1 -translate-x-1/2">ROI per $ spent →</span>
        </div>
        <span>{fmtRoi(domainMax)}</span>
      </div>

      <div className="space-y-3">
        {channels.map((c) => {
          const lo = c.roi_p25 ?? c.roi_min;
          const hi = c.roi_p75 ?? c.roi_max;
          const med = c.roi_median ?? c.roi_min;
          return (
            <div key={c.channel} className="flex items-center gap-3">
              {/* Channel label */}
              <div className="w-32 shrink-0 truncate">
                <p className="truncate text-sm font-medium text-ink-800">{c.channel}</p>
                <p className="text-[11px] text-ink-400">{c.n_brands} {c.n_brands === 1 ? 'brand' : 'brands'}</p>
              </div>

              {/* Distribution track */}
              <div className="relative h-9 flex-1">
                {/* baseline */}
                <div className="absolute inset-x-0 top-1/2 h-px -translate-y-1/2 bg-cream-200" />
                {/* min–max whisker */}
                <div
                  className="absolute top-1/2 h-px -translate-y-1/2 bg-ink-200"
                  style={{ left: `${pct(c.roi_min)}%`, width: `${pct(c.roi_max) - pct(c.roi_min)}%` }}
                />
                {/* IQR box */}
                <div
                  className="absolute top-1/2 h-4 -translate-y-1/2 rounded-sm bg-sage-200/70 ring-1 ring-inset ring-sage-300"
                  style={{ left: `${pct(lo)}%`, width: `${Math.max(0.5, pct(hi) - pct(lo))}%` }}
                />
                {/* median tick */}
                <div
                  className="absolute top-1/2 h-5 w-0.5 -translate-x-1/2 -translate-y-1/2 rounded bg-sage-800"
                  style={{ left: `${pct(med)}%` }}
                  title={`Median ${fmtRoi(c.roi_median)}`}
                />
                {/* brand dots */}
                {(dots[c.channel] ?? []).map((d, i) => (
                  <div
                    key={`${d.name}-${i}`}
                    className={clsx(
                      'absolute top-1/2 h-2 w-2 -translate-x-1/2 -translate-y-1/2 rounded-full ring-2 ring-white',
                      d.percentile != null && d.percentile >= 75
                        ? 'bg-sage-600'
                        : d.percentile != null && d.percentile <= 25
                          ? 'bg-rust-500'
                          : 'bg-ink-400',
                    )}
                    style={{ left: `${pct(d.roi)}%` }}
                    title={`${d.name}: ${fmtRoi(d.roi)}${d.percentile != null ? ` (P${d.percentile})` : ''}`}
                  />
                ))}
              </div>

              {/* Median + marginal readout */}
              <div className="w-24 shrink-0 text-right">
                <p className="text-sm font-semibold tabular-nums text-ink-900">{fmtRoi(c.roi_median)}</p>
                <p className="text-[11px] tabular-nums text-ink-400">m{fmtRoi(c.marginal_roi_median)}</p>
              </div>
            </div>
          );
        })}
      </div>

      {/* Legend */}
      <div className="mt-4 flex flex-wrap items-center gap-x-4 gap-y-1 border-t border-line-200 pt-3 text-[11px] text-ink-400">
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-3 w-4 rounded-sm bg-sage-200/70 ring-1 ring-inset ring-sage-300" />
          P25–P75
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-3.5 w-0.5 rounded bg-sage-800" /> Median
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-2 w-2 rounded-full bg-sage-600 ring-2 ring-white" /> Top quartile brand
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-2 w-2 rounded-full bg-rust-500 ring-2 ring-white" /> Bottom quartile brand
        </span>
        <span className="ml-auto">m = median marginal ROI</span>
      </div>
    </Card>
  );
}
