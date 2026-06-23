import { useEffect, useState } from 'react';
import { Compass } from 'lucide-react';
import {
  MODELING_MODES,
  sessionService,
  type ModelingMode,
} from '../../../../api/services/sessionService';

/**
 * Compact switcher for the session's modeling mode. The mode selects the oracle's
 * prompt framing + available tools (MMM keeps the full ROI / experiment surface;
 * the other modes foreground general Bayesian / causal / measurement work). The
 * change persists via PATCH /sessions/{id}/mode and the next chat turn applies it.
 */
export function ModeSwitcher({
  threadId,
  value,
  onChange,
}: {
  threadId: string | null;
  value: ModelingMode;
  onChange: (mode: ModelingMode) => void;
}) {
  const [saving, setSaving] = useState(false);

  // Hydrate the current mode from the session on mount / thread change.
  useEffect(() => {
    if (!threadId) return;
    let cancelled = false;
    sessionService
      .getSession(threadId)
      .then((s) => {
        if (!cancelled && s.modeling_mode) onChange(s.modeling_mode);
      })
      .catch(() => {/* keep the default */});
    return () => {
      cancelled = true;
    };
    // onChange is stable (setState); intentionally only react to threadId.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [threadId]);

  const handleSelect = async (next: ModelingMode) => {
    if (!threadId || next === value) return;
    const prev = value;
    onChange(next); // optimistic
    setSaving(true);
    try {
      await sessionService.setSessionMode(threadId, next);
    } catch {
      onChange(prev); // revert on failure
    } finally {
      setSaving(false);
    }
  };

  const current = MODELING_MODES.find((m) => m.value === value) ?? MODELING_MODES[0];

  return (
    <label
      title={`Modeling mode — ${current.blurb}`}
      className={`relative flex items-center gap-1.5 pl-2 pr-1 py-1 rounded-lg border border-line-200 bg-white text-ink-600 hover:text-ink-900 transition-colors ${
        saving ? 'opacity-60' : ''
      }`}
    >
      <Compass size={14} className="text-sage-600 shrink-0" />
      <select
        value={value}
        disabled={!threadId || saving}
        onChange={(e) => handleSelect(e.target.value as ModelingMode)}
        className="appearance-none bg-transparent text-xs font-medium text-ink-700 pr-4 focus:outline-none cursor-pointer"
      >
        {MODELING_MODES.map((m) => (
          <option key={m.value} value={m.value}>
            {m.label}
          </option>
        ))}
      </select>
    </label>
  );
}
