import React, { useEffect } from 'react';
import { X } from 'lucide-react';

export function Modal({ title, onClose, fullWidth = false, children }: {
  title: string; onClose: () => void; fullWidth?: boolean; children: React.ReactNode;
}) {
  useEffect(() => {
    const h = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', h);
    return () => window.removeEventListener('keydown', h);
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-ink-900/40 backdrop-blur-sm"
      onClick={e => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div className={`relative flex flex-col bg-white border border-line-200 rounded-2xl shadow-2xl overflow-hidden ${fullWidth ? 'w-full h-full max-w-none' : 'w-full max-w-4xl max-h-[90vh]'}`}>
        <div className="flex items-center justify-between px-6 py-4 border-b border-line-200 shrink-0">
          <h2 className="text-lg font-bold text-ink-900 truncate pr-4">{title}</h2>
          <button onClick={onClose} className="p-2 rounded-full hover:bg-cream-100 text-ink-300 hover:text-ink-700 transition-colors shrink-0" title="Close (Esc)">
            <X size={18} />
          </button>
        </div>
        <div className="overflow-y-auto flex-1 p-6">{children}</div>
      </div>
    </div>
  );
}
