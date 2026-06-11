export function Badge({ label, color = 'gray' }: { label: string; color?: 'blue' | 'indigo' | 'gray' | 'green' | 'amber' | 'red' }) {
  const cls = {
    blue: 'bg-blue-50 text-blue-700 border-blue-200',
    indigo: 'bg-indigo-50 text-indigo-700 border-indigo-200',
    gray: 'bg-gray-100 text-gray-600 border-gray-200',
    green: 'bg-emerald-50 text-emerald-700 border-emerald-200',
    amber: 'bg-amber-50 text-amber-700 border-amber-200',
    red: 'bg-red-50 text-red-700 border-red-200',
  }[color];
  return <span className={`inline-block px-2.5 py-0.5 text-xs rounded-full border font-medium ${cls}`}>{label}</span>;
}
