import { clsx } from 'clsx';
import { Building2, ShieldCheck, AlertTriangle, FlaskConical, Clock } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import { Card } from '../../components/ui';
import type { PortfolioGovernance } from '../../api/services/benchmarkService';
import { fmtAge } from './helpers';

interface TileProps {
  label: string;
  value: string;
  hint?: string;
  icon: LucideIcon;
  tone?: 'neutral' | 'sage' | 'rust' | 'gold';
}

const TONE: Record<NonNullable<TileProps['tone']>, { ring: string; icon: string; value: string }> = {
  neutral: { ring: 'bg-cream-200 text-ink-500', icon: 'text-ink-400', value: 'text-ink-900' },
  sage: { ring: 'bg-sage-100 text-sage-700', icon: 'text-sage-700', value: 'text-ink-900' },
  rust: { ring: 'bg-rust-100 text-rust-700', icon: 'text-rust-700', value: 'text-rust-700' },
  gold: { ring: 'bg-gold-100 text-gold-700', icon: 'text-gold-700', value: 'text-ink-900' },
};

function Tile({ label, value, hint, icon: Icon, tone = 'neutral' }: TileProps) {
  const t = TONE[tone];
  return (
    <Card padding="md" className="flex items-start gap-3">
      <span className={clsx('grid h-9 w-9 shrink-0 place-items-center rounded-lg', t.ring)}>
        <Icon className={t.icon} size={18} />
      </span>
      <div className="min-w-0">
        <p className="text-xs font-medium uppercase tracking-wider text-ink-400">{label}</p>
        <p className={clsx('mt-0.5 font-display text-2xl font-semibold tabular-nums', t.value)}>{value}</p>
        {hint && <p className="mt-0.5 text-xs text-ink-400">{hint}</p>}
      </div>
    </Card>
  );
}

export function GovernanceTiles({ g }: { g: PortfolioGovernance }) {
  const staleTone = g.n_stale > 0 ? 'rust' : 'sage';
  return (
    <div className="grid grid-cols-2 gap-4 md:grid-cols-3 xl:grid-cols-5">
      <Tile
        label="Brands"
        value={String(g.n_projects)}
        hint={`${g.n_with_fit} with a fitted model`}
        icon={Building2}
      />
      <Tile
        label="Fresh models"
        value={String(g.n_fresh)}
        hint={`≤ ${Math.round(g.stale_after_days)}d since last fit`}
        icon={ShieldCheck}
        tone="sage"
      />
      <Tile
        label="Stale models"
        value={String(g.n_stale)}
        hint={g.n_stale > 0 ? 'need a refit' : 'all current'}
        icon={AlertTriangle}
        tone={staleTone}
      />
      <Tile
        label="Calibrated"
        value={String(g.n_calibrated_projects)}
        hint="brands with ≥1 calibrated test"
        icon={FlaskConical}
        tone="gold"
      />
      <Tile
        label="Median age"
        value={fmtAge(g.median_model_age_days)}
        hint="across fitted brands"
        icon={Clock}
      />
    </div>
  );
}
