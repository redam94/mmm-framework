import React from 'react';

// ─── Shared form primitives ───────────────────────────────────────────────────

export const iCls = 'w-full bg-cream-50 border border-line-200 rounded-lg px-2.5 py-1.5 text-xs text-ink-900 focus:outline-none focus:ring-2 focus:ring-indigo-400 focus:border-transparent transition-all';
export const sCls = iCls + ' cursor-pointer';

export function FLabel({ children }: { children: React.ReactNode }) {
  return <p className="text-[10px] font-semibold text-ink-400 uppercase tracking-wider mb-0.5">{children}</p>;
}
