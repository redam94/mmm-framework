import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';

// The sections fetch account/model/connection state — stub them so this test
// pins the Settings SHELL (tabs render + switch), not the section internals.
vi.mock('./ProfileSection', () => ({
  ProfileSection: () => <div data-testid="profile-section" />,
}));
vi.mock('./SecuritySection', () => ({
  SecuritySection: () => <div data-testid="security-section" />,
}));
vi.mock('./ModelSection', () => ({
  ModelSection: () => <div data-testid="model-section" />,
}));
vi.mock('./DataConnectionsSection', () => ({
  DataConnectionsSection: () => <div data-testid="connections-section" />,
}));

const { SettingsPage } = await import('./index');

describe('SettingsPage', () => {
  it('renders the profile tab by default', () => {
    render(<SettingsPage />);
    expect(screen.getByTestId('profile-section')).toBeInTheDocument();
  });

  it('switches tabs', () => {
    render(<SettingsPage />);
    fireEvent.click(screen.getByText('Model & API'));
    expect(screen.getByTestId('model-section')).toBeInTheDocument();
    fireEvent.click(screen.getByText('Data connections'));
    expect(screen.getByTestId('connections-section')).toBeInTheDocument();
  });
});
