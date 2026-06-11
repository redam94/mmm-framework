import React from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';

// ─── mdComponents ─────────────────────────────────────────────────────────────
// Factory for the ReactMarkdown `components` map. Built per `onNavigate`
// identity (call sites memoize on it) so chat deep-links of the form
// `#tab:<id>` can intercept clicks and switch workspace tabs in-app.

const TAB_LINK_RE = /^#tab:([\w-]+)$/;

export function mdComponents(onNavigate?: (tab: string) => void): any {
  return {
    a: ({ href, children, title }: { href?: string; children?: React.ReactNode; title?: string }) => {
      const m = TAB_LINK_RE.exec(href || '');
      if (m && onNavigate) {
        const tab = m[1];
        return (
          <a
            href={href}
            title={title}
            onClick={(e) => { e.preventDefault(); onNavigate(tab); }}
            className="text-indigo-600 underline decoration-indigo-300 hover:decoration-indigo-500 cursor-pointer"
          >
            {children}
          </a>
        );
      }
      // Non-tab links: keep the default <a> behavior (same as before this factory).
      return <a href={href} title={title}>{children}</a>;
    },
    table: ({ children }: any) => (
      <div className="overflow-x-auto my-2">
        <table className="min-w-full text-sm border-collapse">{children}</table>
      </div>
    ),
    thead: ({ children }: any) => <thead className="bg-gray-100">{children}</thead>,
    th: ({ children }: any) => <th className="px-3 py-2 text-left font-semibold text-indigo-600 border border-gray-200">{children}</th>,
    td: ({ children }: any) => <td className="px-3 py-2 text-gray-700 border border-gray-200">{children}</td>,
    tr: ({ children }: any) => <tr className="even:bg-gray-50">{children}</tr>,
    code: ({ inline, className, children }: any) => {
      const raw = String(children ?? '').replace(/\n$/, '');
      const langMatch = /language-(\w+)/.exec(className || '');
      // react-markdown v10 no longer passes `inline`; fall back to detecting a
      // fenced block via the language className or a multi-line body.
      const isBlock = inline === false || !!langMatch || raw.includes('\n');
      if (!isBlock) {
        return <code className="bg-gray-100 px-1 py-0.5 rounded text-indigo-600 text-xs font-mono">{children}</code>;
      }
      return (
        <SyntaxHighlighter
          language={langMatch ? langMatch[1] : 'text'}
          style={oneLight}
          PreTag="div"
          customStyle={{
            margin: '0.5rem 0', borderRadius: '0.5rem', fontSize: '0.75rem',
            border: '1px solid #e5e7eb', background: '#f9fafb',
          }}
          codeTagProps={{ style: { fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace' } }}
        >
          {raw}
        </SyntaxHighlighter>
      );
    },
  };
}
