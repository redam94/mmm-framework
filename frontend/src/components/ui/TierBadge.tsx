import { EVIDENCE_TIER, type EvidenceTier } from '../../theme/colors';

interface TierBadgeProps {
  tier: EvidenceTier;
  /** Compact: dot + no label (for dense grids) */
  compact?: boolean;
  /** Override label (e.g. "calibrated 2 cycles ago") */
  label?: string;
}

/** Evidence-tier chip: calibrated=sage · running=gold · model-only=steel · stale=rust */
export function TierBadge({ tier, compact = false, label }: TierBadgeProps) {
  const t = EVIDENCE_TIER[tier] ?? EVIDENCE_TIER.model_only;
  if (compact) {
    return (
      <span
        className="inline-block h-2.5 w-2.5 rounded-full"
        style={{ backgroundColor: t.fg }}
        title={label ?? t.label}
      />
    );
  }
  return (
    <span
      className="inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-xs font-medium"
      style={{ color: t.fg, backgroundColor: t.bg, borderColor: t.border }}
    >
      <span className="h-1.5 w-1.5 rounded-full" style={{ backgroundColor: t.fg }} />
      {label ?? t.label}
    </span>
  );
}
