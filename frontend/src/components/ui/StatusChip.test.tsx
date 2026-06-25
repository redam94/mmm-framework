import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { StatusChip } from './StatusChip';

// Proves the harness is component-ready (jsdom + Testing Library + jest-dom),
// not just for pure functions.
describe('StatusChip', () => {
  it('renders a custom label when provided', () => {
    render(<StatusChip status="running" label="In flight" />);
    expect(screen.getByText('In flight')).toBeInTheDocument();
  });

  it('falls back to the raw status for an unknown status', () => {
    render(<StatusChip status="zzz_unknown" />);
    expect(screen.getByText('zzz_unknown')).toBeInTheDocument();
  });

  it('renders a non-empty label for a known status', () => {
    const { container } = render(<StatusChip status="running" />);
    expect((container.textContent ?? '').trim().length).toBeGreaterThan(0);
  });
});
