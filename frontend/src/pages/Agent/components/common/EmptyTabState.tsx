import React from 'react';

export function EmptyTabState({ icon, title, hint }: { icon: React.ReactNode; title: string; hint: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-16 text-ink-300 space-y-3">
      <div className="w-14 h-14 rounded-2xl bg-white flex items-center justify-center border border-line-200 shadow-sm text-ink-300">
        {icon}
      </div>
      <p className="text-base text-ink-400 font-medium">{title}</p>
      <p className="text-sm text-ink-300 max-w-sm text-center">{hint}</p>
    </div>
  );
}
