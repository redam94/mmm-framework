import type { ExperimentVerdict } from '../../api/services/measurementService';

/**
 * Coerce to a finite number, mapping null / undefined / NaN / ±Infinity → null.
 * Critically NOT `Number(v)`: `Number(null) === 0` (and is finite), which would
 * turn a persisted `null` (an absent posterior quantity — e.g. an inconclusive
 * anchor's `prob_detectable`) into a fabricated `0`. Returning null keeps "no
 * value" distinct from "the value is zero".
 */
function fin(v: unknown): number | null {
  const n = typeof v === 'number' ? v : v == null ? NaN : Number(v);
  return Number.isFinite(n) ? n : null;
}

/**
 * MDE (ROAS) at 80% power — the smallest true ROAS effect the design can
 * detect. Always recorded on a studio/agent design payload (`mde_roas`).
 */
export function designMDE(design: Record<string, any> | null | undefined): number | null {
  return fin(design?.mde_roas);
}

export interface DesignPower {
  /** P(detect the model's expected effect) ∈ [0,1], or null when no model anchor. */
  power: number | null;
  verdict: ExperimentVerdict | null;
  /** which posterior quantity backs `power` */
  basis: 'prob_detectable' | 'assurance' | null;
  /** test weeks at which the design would reach power, if the curve says so */
  recommendedDuration: number | null;
}

const EMPTY: DesignPower = {
  power: null,
  verdict: null,
  basis: null,
  recommendedDuration: null,
};

/**
 * Power to detect the model's *expected* effect, read from a design's model
 * anchor. Tolerant of both persisted shapes:
 *   - agent / economics op:  design.model_anchor.verdict.{prob_detectable,assurance,verdict}
 *   - flat ExperimentAnchor: design.model_anchor.{prob_detectable,assurance,verdict}
 * Returns null power when no anchor was attached (e.g. a design pre-registered
 * without a fitted model / before running the model-backed simulation), AND
 * when the anchor exists but reported a null power (an inconclusive/degenerate
 * anchor) — "no number" must not collapse to "0%".
 *
 * prob_detectable (the MDE indicator — P(|effect| > MDE)) leads; the signed
 * two-sided assurance is the fallback. Both are genuine "will this test detect
 * what we expect" numbers, not posterior width — `basis` records which one.
 */
export function designPower(design: Record<string, any> | null | undefined): DesignPower {
  const ma = design?.model_anchor;
  if (ma == null || typeof ma !== 'object') return EMPTY;
  // nested anchor stores a `verdict` sub-object; a flat ExperimentAnchor keeps
  // the verdict as a string and the numbers at the top level.
  const nested = ma.verdict != null && typeof ma.verdict === 'object';
  const src = nested ? ma.verdict : ma;
  const verdictLabel =
    typeof ma.verdict === 'string'
      ? ma.verdict
      : typeof src?.verdict === 'string'
        ? src.verdict
        : null;

  const pd = fin(src?.prob_detectable);
  const asr = fin(src?.assurance);
  let power: number | null = null;
  let basis: DesignPower['basis'] = null;
  if (pd != null) {
    power = pd;
    basis = 'prob_detectable';
  } else if (asr != null) {
    power = asr;
    basis = 'assurance';
  }

  return {
    power,
    verdict: (verdictLabel as ExperimentVerdict | null) ?? null,
    basis,
    recommendedDuration: fin(src?.recommended_duration),
  };
}

const VERDICT_TEXT: Record<string, string> = {
  powered: 'text-sage-800',
  overpowered: 'text-gold-700',
  underpowered: 'text-rust-600',
  inconclusive: 'text-rust-600',
};

/** Text colour for a power figure: verdict-driven, else a hard 80% threshold. */
export function powerTextColor(power: number | null, verdict: string | null): string {
  if (verdict && VERDICT_TEXT[verdict]) return VERDICT_TEXT[verdict];
  if (power == null) return 'text-ink-400';
  return power >= 0.8 ? 'text-sage-800' : 'text-rust-600';
}

/**
 * Human label for what the power figure measures — keeps the assurance fallback
 * honest (it is a different quantity from the MDE-indicator prob_detectable, so
 * it must not be shown under an identical, unqualified "power" label).
 */
export function powerBasisLabel(basis: DesignPower['basis']): string {
  if (basis === 'assurance') return 'signed two-sided assurance';
  if (basis === 'prob_detectable') return 'P(detect expected effect): P(|effect| > MDE)';
  return 'power to detect the model’s expected effect';
}
