/**
 * Shared markdown renderer + code-extraction helper for the Atelier copilots
 * (editor `CopilotPanel` and `NotebookCopilotPanel`). Kept in its own module so
 * the component files only export components (React Fast Refresh).
 */
import type { Components } from 'react-markdown';

export const MD_COMPONENTS: Components = {
  p: ({ children }) => <p className="mb-2 last:mb-0 leading-relaxed">{children}</p>,
  ul: ({ children }) => <ul className="mb-2 ml-4 list-disc space-y-1">{children}</ul>,
  ol: ({ children }) => <ol className="mb-2 ml-4 list-decimal space-y-1">{children}</ol>,
  li: ({ children }) => <li className="leading-relaxed">{children}</li>,
  a: ({ children, href }) => (
    <a href={href} target="_blank" rel="noreferrer" className="text-sage-700 underline">
      {children}
    </a>
  ),
  code: ({ className, children }) => {
    const isBlock = (className ?? '').includes('language-');
    if (isBlock) {
      return (
        <code className="block overflow-x-auto rounded-md bg-ink-900/95 p-3 font-mono text-xs leading-relaxed text-cream-100">
          {children}
        </code>
      );
    }
    return (
      <code className="rounded bg-cream-200 px-1 py-0.5 font-mono text-[0.85em] text-ink-800">
        {children}
      </code>
    );
  },
  pre: ({ children }) => <pre className="mb-2">{children}</pre>,
  h1: ({ children }) => <h3 className="mb-1 mt-2 font-display text-sm font-semibold text-ink-900">{children}</h3>,
  h2: ({ children }) => <h3 className="mb-1 mt-2 font-display text-sm font-semibold text-ink-900">{children}</h3>,
  h3: ({ children }) => <h4 className="mb-1 mt-2 text-sm font-semibold text-ink-800">{children}</h4>,
};

/** Pull the LAST fenced code block from a markdown string (python-ish or bare). */
export function lastCodeBlock(md: string): string | null {
  const re = /```(?:python|py)?\s*\n([\s\S]*?)```/g;
  let match: RegExpExecArray | null;
  let last: string | null = null;
  while ((match = re.exec(md)) !== null) last = match[1];
  return last ? last.replace(/\s+$/, '') : null;
}
