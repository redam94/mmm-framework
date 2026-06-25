import { describe, it, expect } from 'vitest';
import { normalizeMath } from './markdownMath';

// The high-value, tricky pure function: make LLM LaTeX render WITHOUT mangling
// the currency that pervades an MMM app ($5,000 / $1.2M ROI).
describe('normalizeMath', () => {
  it('escapes a $ that directly precedes a digit (currency, not math)', () => {
    expect(normalizeMath('$5,000')).toBe('\\$5,000');
    expect(normalizeMath('spend $5,000 and $3,000')).toBe(
      'spend \\$5,000 and \\$3,000',
    );
  });

  it('leaves real inline math ($\\beta$, $x=5$) untouched', () => {
    expect(normalizeMath('the coefficient $\\beta$ is')).toBe(
      'the coefficient $\\beta$ is',
    );
    expect(normalizeMath('$x=5$')).toBe('$x=5$'); // no leading digit after $
  });

  it('converts \\[ … \\] to $$ … $$ (display math)', () => {
    expect(normalizeMath('\\[ a + b \\]')).toBe('$$ a + b $$');
  });

  it('converts \\( … \\) to $ … $ (inline math)', () => {
    expect(normalizeMath('\\( \\alpha \\)')).toBe('$ \\alpha $');
  });

  it('leaves code spans/fences verbatim (no $ or \\ rewriting)', () => {
    expect(normalizeMath('`$5`')).toBe('`$5`');
    const fenced = '```\nprice = $5\n```';
    expect(normalizeMath(fenced)).toBe(fenced);
  });

  it('is a no-op for plain text with no $ or backslash', () => {
    expect(normalizeMath('just words')).toBe('just words');
  });

  it('handles mixed currency + math in one string', () => {
    const out = normalizeMath('ROI $\\beta$ on $5,000');
    expect(out).toContain('$\\beta$'); // math preserved
    expect(out).toContain('\\$5,000'); // currency escaped
  });
});
