import { parseTextTable, pyOutputKind } from '../../utils/python';

export function PythonOutputBlock({ output, hasError }: { output: string; hasError: boolean }) {
  const kind = hasError ? 'error' : pyOutputKind(output);

  if (kind === 'error') {
    const lines = output.split('\n');
    return (
      <div className="rounded-b-lg border border-t-0 border-red-200 bg-red-50 overflow-hidden">
        <div className="flex items-center gap-2 px-3 py-1.5 bg-red-100 border-b border-red-200">
          <span className="w-1.5 h-1.5 rounded-full bg-red-500" />
          <span className="text-[10px] font-semibold text-red-700 uppercase tracking-widest">Error</span>
        </div>
        <div className="overflow-x-auto max-h-48 p-3">
          <pre className="text-[11px] font-mono text-red-800 whitespace-pre leading-5">
            {lines.map((l, i) => (
              <span key={i} className={/^\w+Error:|^\w+Exception:/.test(l) ? 'font-bold text-red-900' : ''}>
                {l}{'\n'}
              </span>
            ))}
          </pre>
        </div>
      </div>
    );
  }

  if (kind === 'table') {
    const parsed = parseTextTable(output);
    if (parsed) {
      return (
        <div className="rounded-b-lg border border-t-0 border-gray-200 bg-white overflow-hidden">
          <div className="flex items-center gap-2 px-3 py-1.5 bg-gray-50 border-b border-gray-200">
            <span className="w-1.5 h-1.5 rounded-full bg-blue-400" />
            <span className="text-[10px] font-semibold text-gray-500 uppercase tracking-widest">DataFrame</span>
            <span className="ml-auto text-[10px] text-gray-400">{parsed.rows.length} rows × {parsed.headers.length} cols</span>
          </div>
          <div className="overflow-x-auto max-h-56">
            <table className="w-full text-[11px] font-mono">
              <thead className="sticky top-0 bg-gray-50 border-b border-gray-200">
                <tr>
                  {parsed.headers.map((h, i) => (
                    <th key={i} className="px-3 py-1.5 text-right font-semibold text-gray-600 whitespace-nowrap first:text-left">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {parsed.rows.map((row, i) => (
                  <tr key={i} className="hover:bg-blue-50/40">
                    {row.map((cell, j) => (
                      <td key={j} className="px-3 py-1 text-right text-gray-700 whitespace-nowrap first:text-left first:font-medium first:text-gray-500">{cell}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      );
    }
  }

  // Plain terminal output
  return (
    <div className="rounded-b-lg border border-t-0 border-gray-700 bg-gray-950 overflow-hidden">
      <div className="flex items-center gap-2 px-3 py-1.5 bg-gray-800 border-b border-gray-700">
        <span className="w-1.5 h-1.5 rounded-full bg-green-400" />
        <span className="text-[10px] font-semibold text-gray-400 uppercase tracking-widest">Output</span>
      </div>
      <div className="overflow-x-auto max-h-56 p-3">
        <pre className="text-[11px] font-mono text-green-300 whitespace-pre leading-5">{output}</pre>
      </div>
    </div>
  );
}
