/**
 * Shared math-aware markdown configuration for every `react-markdown` surface
 * (agent chat, copilots, guide, notebook cells, garden docs).
 *
 * Two pieces:
 *  - `remarkPlugins` / `rehypePlugins`: GFM + math parsing + KaTeX rendering.
 *  - `normalizeMath`: a pre-processor that makes LLM-emitted LaTeX render
 *    reliably WITHOUT mangling currency — the constant hazard in an MMM app
 *    where messages are full of `$5,000` / `$1.2M ROI`.
 *
 * Why a pre-processor? `remark-math` (with single-`$` math, which we keep so
 * inline `$\beta$` works) parses `"$5,000 and $3,000"` as the inline math
 * `"5,000 and "` — corrupting currency. And it does not understand the
 * `\( … \)` / `\[ … \]` delimiters that some models emit at all. `normalizeMath`
 * fixes both, operating only on non-code text so code blocks stay verbatim:
 *   1. `\[ … \]`  → `$$ … $$`   (display)
 *   2. `\( … \)`  → `$ … $`      (inline)
 *   3. a lone `$` immediately followed by a digit → `\$` (escaped currency), so
 *      it can never open a math span. Real math like `$\beta$` / `$x=5$` (no
 *      leading digit) is untouched; only the rare digit-leading inline math
 *      written with `$` is sacrificed.
 */
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import type { PluggableList } from 'unified';

export const remarkPlugins: PluggableList = [remarkGfm, remarkMath];

// `throwOnError: false` renders invalid LaTeX as an inline error string instead
// of throwing (which would blank the whole message); `errorColor` matches the
// app's muted danger tone.
export const rehypePlugins: PluggableList = [
  [rehypeKatex, { throwOnError: false, errorColor: '#b5654b' }],
];

// Fenced (``` / ~~~) blocks and inline `code` spans — preserved verbatim so we
// never rewrite `$` or `\(` that is really shell/code text.
const CODE_RE = /(```[\s\S]*?```|~~~[\s\S]*?~~~|`[^`\n]*`)/g;

export function normalizeMath(src: string): string {
  if (!src || (src.indexOf('\\') === -1 && src.indexOf('$') === -1)) return src;
  return src
    .split(CODE_RE)
    .map((seg, i) => {
      if (i % 2 === 1) return seg; // code segment — leave untouched
      return seg
        .replace(/\\\[([\s\S]+?)\\\]/g, (_m, body) => `$$${body}$$`)
        .replace(/\\\(([\s\S]+?)\\\)/g, (_m, body) => `$${body}$`)
        .replace(/(?<!\$)\$(?!\$)(?=\d)/g, '\\$');
    })
    .join('');
}
