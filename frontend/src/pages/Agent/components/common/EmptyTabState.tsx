import React from 'react';

export function EmptyTabState({ icon, title, hint }: { icon: React.ReactNode; title: string; hint: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-16 text-gray-400 space-y-3">
      <div className="w-14 h-14 rounded-2xl bg-white flex items-center justify-center border border-gray-200 shadow-sm text-gray-300">
        {icon}
      </div>
      <p className="text-base text-gray-500 font-medium">{title}</p>
      <p className="text-sm text-gray-400 max-w-sm text-center">{hint}</p>
    </div>
  );
}
