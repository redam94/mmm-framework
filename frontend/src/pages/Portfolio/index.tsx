import { useState } from 'react';
import { Building2, BarChart3 } from 'lucide-react';
import { EmptyState, SectionHeader } from '../../components/ui';
import { usePortfolioBenchmark } from '../../api/hooks/useBenchmark';
import { GovernanceTiles } from './GovernanceTiles';
import { ChannelBenchmark } from './ChannelBenchmark';
import { BrandTable } from './BrandTable';
import { brandDotsByChannel } from './helpers';

const STALE_OPTIONS = [30, 60, 90, 180];

export function PortfolioPage() {
  const [staleAfter, setStaleAfter] = useState(90);
  const { data, isLoading, isError, error } = usePortfolioBenchmark(staleAfter);

  const staleSelect = (
    <label className="flex items-center gap-2 text-sm text-ink-500">
      <span className="hidden sm:inline">Stale after</span>
      <select
        value={staleAfter}
        onChange={(e) => setStaleAfter(Number(e.target.value))}
        className="rounded-lg border border-line-200 bg-white px-2.5 py-1.5 text-sm text-ink-800 shadow-sm focus:border-sage-500 focus:outline-none focus:ring-1 focus:ring-sage-500"
      >
        {STALE_OPTIONS.map((d) => (
          <option key={d} value={d}>
            {d} days
          </option>
        ))}
      </select>
    </label>
  );

  return (
    <div className="space-y-6">
      <SectionHeader
        level={1}
        title="Portfolio"
        subtitle="Every brand benchmarked on the same yardstick — ROI, model freshness, and calibration coverage across the book."
        actions={staleSelect}
      />

      {isLoading ? (
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-4 md:grid-cols-3 xl:grid-cols-5">
            {Array.from({ length: 5 }).map((_, i) => (
              <div key={i} className="h-20 animate-pulse rounded-lg border border-line-200 bg-cream-100" />
            ))}
          </div>
          <div className="h-64 animate-pulse rounded-lg border border-line-200 bg-cream-100" />
        </div>
      ) : isError ? (
        <EmptyState
          icon={Building2}
          title="Couldn’t load the portfolio"
          description={(error as Error)?.message ?? 'The benchmark endpoint returned an error. Try again shortly.'}
        />
      ) : !data || data.governance.n_projects === 0 ? (
        <EmptyState
          icon={Building2}
          title="No brands yet"
          description="Create a project and fit a model to start building the cross-brand benchmark. Each fitted brand contributes its channel ROIs to the portfolio distribution."
        />
      ) : (
        <div className="space-y-8">
          <GovernanceTiles g={data.governance} />

          <section className="space-y-3">
            <div className="flex items-center gap-2">
              <BarChart3 size={16} className="text-ink-400" />
              <h2 className="text-xs font-bold uppercase tracking-wider text-ink-400">
                Channel ROI across the portfolio
              </h2>
            </div>
            <ChannelBenchmark channels={data.channels} dots={brandDotsByChannel(data.projects)} />
          </section>

          <section className="space-y-3">
            <div className="flex items-center gap-2">
              <Building2 size={16} className="text-ink-400" />
              <h2 className="text-xs font-bold uppercase tracking-wider text-ink-400">Brands</h2>
            </div>
            <BrandTable brands={data.projects} />
          </section>
        </div>
      )}
    </div>
  );
}
