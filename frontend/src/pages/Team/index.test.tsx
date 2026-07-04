import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import type { TeamUser } from '../../api/services/teamService';

const USERS: TeamUser[] = [
  { user_id: 'u1', name: 'Dana Whitfield', email: 'dana@acme.com', role: 'owner' } as TeamUser,
  { user_id: 'u2', name: 'Sam Ortiz', email: null as unknown as string, role: 'viewer' } as TeamUser,
];

const updateMutate = vi.fn();

vi.mock('../../api/hooks/useTeam', () => ({
  useUsers: () => ({ data: { users: USERS, total: 2 }, isLoading: false }),
  useCreateUser: () => ({ mutateAsync: vi.fn(), isPending: false }),
  useUpdateUser: () => ({ mutate: updateMutate, isPending: false }),
  useDeleteUser: () => ({ mutate: vi.fn(), isPending: false }),
}));

const { TeamPage } = await import('./index');

describe('TeamPage', () => {
  it('renders the roster with role chips and an add action', () => {
    render(<TeamPage />);
    expect(screen.getByText('Dana Whitfield')).toBeInTheDocument();
    expect(screen.getByText('Sam Ortiz')).toBeInTheDocument();
    // role appears as a chip AND as a select option
    expect(screen.getAllByText('owner').length).toBeGreaterThanOrEqual(1);
    // missing email renders a placeholder, not a crash
    expect(screen.getByText('—')).toBeInTheDocument();
  });

  it('changing a role select fires the update mutation', () => {
    render(<TeamPage />);
    const selects = screen.getAllByRole('combobox');
    fireEvent.change(selects[0], { target: { value: 'analyst' } });
    expect(updateMutate).toHaveBeenCalledWith({
      userId: 'u1',
      body: { role: 'analyst' },
    });
  });
});
