import React, { useState } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';

// Small collapsible shell (mirrors CausalWidgets PanelShell) for new widgets.
export function PanelShellLite({ title, icon, color = 'gray', children }: {
  title: React.ReactNode; icon: React.ReactNode; color?: string; children: React.ReactNode;
}) {
  const [open, setOpen] = useState(true);
  return (
    <div className="bg-white rounded-2xl border border-gray-200 shadow-sm overflow-hidden">
      <button onClick={() => setOpen(v => !v)} className="w-full flex items-center gap-3 px-5 py-4 text-left">
        {icon}
        <span className={`font-semibold text-sm text-${color}-600 flex-1`}>{title}</span>
        {open ? <ChevronDown size={15} className="text-gray-400" /> : <ChevronRight size={15} className="text-gray-400" />}
      </button>
      {open && <div className="px-5 pb-5">{children}</div>}
    </div>
  );
}
