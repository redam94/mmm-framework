import React, { useState } from 'react';
import { ChevronDown, ChevronRight, Maximize2 } from 'lucide-react';
import { Modal } from './Modal';
import { titleColorClass } from '../../../../theme/uiMaps';

export function DashWidget({
  title, icon, color = 'indigo', dotColor = 'bg-sage-600',
  defaultOpen = true, expandTitle, expandContent, children,
}: {
  title: string; icon?: React.ReactNode; color?: string; dotColor?: string;
  defaultOpen?: boolean; expandTitle?: string; expandContent?: React.ReactNode; children: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  const [modal, setModal] = useState(false);

  return (
    <>
      <div className="bg-white rounded-2xl border border-line-200 shadow-sm hover:shadow-md transition-all overflow-hidden">
        <div className="flex items-center gap-3 px-5 py-4">
          <button onClick={() => setOpen(v => !v)} className="flex items-center gap-3 flex-1 text-left">
            {icon || <span className={`w-2 h-2 rounded-full shrink-0 ${dotColor}`} />}
            <span className={`font-semibold text-sm flex-1 ${titleColorClass(color)}`}>{title}</span>
            {open ? <ChevronDown size={15} className="text-ink-300 shrink-0" /> : <ChevronRight size={15} className="text-ink-300 shrink-0" />}
          </button>
          <button onClick={() => setModal(true)} className="p-1.5 rounded-lg hover:bg-cream-100 text-ink-300 hover:text-ink-700 transition-colors shrink-0" title="Expand">
            <Maximize2 size={14} />
          </button>
        </div>
        {open && <div className="px-5 pb-5">{children}</div>}
      </div>
      {modal && <Modal title={expandTitle || title} onClose={() => setModal(false)}>{expandContent || children}</Modal>}
    </>
  );
}
