import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ReportViewer } from './ReportViewer';

describe('ReportViewer (U4)', () => {
  it('scopes the iframe + download to the session via ?thread_id=', () => {
    const { container } = render(<ReportViewer threadId="abc123" kind="mmm" />);
    const iframe = container.querySelector('iframe');
    expect(iframe?.getAttribute('src')).toContain('/report?thread_id=abc123');
    const dl = screen.getByText('Download').closest('a');
    expect(dl?.getAttribute('href')).toContain('/report/download?thread_id=abc123');
  });

  it('uses the project-report endpoint for the project kind', () => {
    const { container } = render(<ReportViewer threadId="t1" kind="project" />);
    expect(container.querySelector('iframe')?.getAttribute('src')).toContain(
      '/project-report?thread_id=t1',
    );
  });

  it('shows an empty hint with no active session', () => {
    const { container } = render(<ReportViewer threadId={null} />);
    expect(container.querySelector('iframe')).toBeNull();
    expect(screen.getByText(/No active session/i)).toBeInTheDocument();
  });
});
