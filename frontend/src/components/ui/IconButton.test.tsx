import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { IconButton } from './IconButton';

describe('IconButton (a11y / H2)', () => {
  it('exposes the label as the accessible name', () => {
    render(
      <IconButton label="Delete session">
        <svg aria-hidden="true" />
      </IconButton>,
    );
    // Found BY its accessible name — what a screen reader announces.
    const btn = screen.getByRole('button', { name: 'Delete session' });
    expect(btn).toBeInTheDocument();
    expect(btn).toHaveAttribute('title', 'Delete session');
  });

  it('fires onClick', async () => {
    const onClick = vi.fn();
    render(
      <IconButton label="Refresh" onClick={onClick}>
        <svg aria-hidden="true" />
      </IconButton>,
    );
    await userEvent.click(screen.getByRole('button', { name: 'Refresh' }));
    expect(onClick).toHaveBeenCalledTimes(1);
  });
});
