import { useState } from 'react';
import { Check } from 'lucide-react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';

export function PythonCodeBlock({ code }: { code: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <div className="rounded-t-lg overflow-hidden border border-gray-700">
      <div className="flex items-center justify-between px-3 py-1.5 bg-gray-800">
        <div className="flex gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-red-500/60" />
          <span className="w-2.5 h-2.5 rounded-full bg-yellow-500/60" />
          <span className="w-2.5 h-2.5 rounded-full bg-green-500/60" />
        </div>
        <span className="text-[10px] text-gray-400 font-medium">Python</span>
        <button
          onClick={() => { navigator.clipboard.writeText(code); setCopied(true); setTimeout(() => setCopied(false), 1500); }}
          className="text-[10px] text-gray-500 hover:text-gray-200 transition-colors flex items-center gap-1"
        >
          {copied ? <><Check size={10} />Copied</> : 'Copy'}
        </button>
      </div>
      <div className="overflow-x-auto max-h-64 bg-[#fafafa]">
        <SyntaxHighlighter
          language="python"
          style={oneLight}
          showLineNumbers
          PreTag="div"
          customStyle={{ margin: 0, padding: '0.5rem 0', fontSize: '0.6875rem', background: '#fafafa', lineHeight: '1.25rem' }}
          lineNumberStyle={{ minWidth: '2.25em', paddingRight: '0.75em', color: '#9ca3af', userSelect: 'none' }}
          codeTagProps={{ style: { fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace' } }}
        >
          {code.replace(/\n$/, '')}
        </SyntaxHighlighter>
      </div>
    </div>
  );
}
