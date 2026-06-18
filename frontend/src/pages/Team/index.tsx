import { useState } from 'react';
import { Trash2, UserPlus, Users } from 'lucide-react';
import { clsx } from 'clsx';
import {
  useCreateUser,
  useDeleteUser,
  useUpdateUser,
  useUsers,
} from '../../api/hooks/useTeam';
import type { TeamRole, TeamUser } from '../../api/services/teamService';
import { Button, DataTable, EmptyState, SectionHeader } from '../../components/ui';
import type { Column } from '../../components/ui';

const ROLE_OPTIONS: TeamRole[] = ['owner', 'analyst', 'viewer'];

const ROLE_CHIP: Record<TeamRole, string> = {
  owner: 'bg-sage-100 text-sage-800',
  analyst: 'bg-steel-100 text-steel-700',
  viewer: 'bg-cream-200 text-ink-600',
};

function RoleChip({ role }: { role: TeamRole }) {
  return (
    <span
      className={clsx(
        'inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium',
        ROLE_CHIP[role] ?? ROLE_CHIP.viewer,
      )}
    >
      {role}
    </span>
  );
}

function AddMemberModal({ onClose }: { onClose: () => void }) {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [role, setRole] = useState<TeamRole>('analyst');
  const [error, setError] = useState<string | null>(null);
  const create = useCreateUser();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim() || create.isPending) return;
    setError(null);
    try {
      await create.mutateAsync({
        name: name.trim(),
        email: email.trim() || undefined,
        role,
      });
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not add the member.');
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-ink-900/40 p-4"
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <div className="w-full max-w-md rounded-xl bg-white shadow-2xl">
        <div className="border-b border-line-200 px-6 py-4">
          <h2 className="font-display text-lg font-semibold text-ink-900">Add member</h2>
        </div>
        <form onSubmit={handleSubmit} className="space-y-4 px-6 py-4">
          <div>
            <label className="mb-1 block text-sm font-medium text-ink-700">Name</label>
            <input
              autoFocus
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. Dana Whitfield"
              className="w-full rounded-md border border-line-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-sage-600"
            />
          </div>
          <div>
            <label className="mb-1 block text-sm font-medium text-ink-700">Email (optional)</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="dana@example.com"
              className="w-full rounded-md border border-line-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-sage-600"
            />
          </div>
          <div>
            <label className="mb-1 block text-sm font-medium text-ink-700">Role</label>
            <select
              value={role}
              onChange={(e) => setRole(e.target.value as TeamRole)}
              className="w-full rounded-md border border-line-300 px-3 py-2 text-sm text-ink-700 focus:outline-none focus:ring-2 focus:ring-sage-600"
            >
              {ROLE_OPTIONS.map((r) => (
                <option key={r} value={r}>
                  {r}
                </option>
              ))}
            </select>
          </div>
          {error && (
            <p className="rounded-md border border-rust-600/30 bg-rust-100 px-3 py-2 text-sm text-rust-700">
              {error}
            </p>
          )}
          <div className="flex justify-end gap-3 pt-2">
            <Button type="button" variant="ghost" onClick={onClose}>
              Cancel
            </Button>
            <Button type="submit" disabled={!name.trim() || create.isPending}>
              Add member
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
}

export function TeamPage() {
  const [showAdd, setShowAdd] = useState(false);
  const { data, isLoading } = useUsers();
  const updateUser = useUpdateUser();
  const deleteUser = useDeleteUser();
  const users = data?.users ?? [];

  const handleDelete = (user: TeamUser) => {
    if (window.confirm(`Remove ${user.name} from the roster?`)) {
      deleteUser.mutate(user.user_id);
    }
  };

  const columns: Column<TeamUser>[] = [
    {
      key: 'name',
      header: 'Name',
      render: (u) => <span className="font-medium text-ink-900">{u.name}</span>,
    },
    {
      key: 'email',
      header: 'Email',
      render: (u) => <span className="text-ink-400">{u.email || '—'}</span>,
    },
    {
      key: 'role',
      header: 'Role',
      render: (u) => (
        <span className="flex items-center gap-2">
          <RoleChip role={u.role} />
          <select
            value={u.role}
            onChange={(e) =>
              updateUser.mutate({ userId: u.user_id, body: { role: e.target.value as TeamRole } })
            }
            className="rounded-md border border-line-300 px-1.5 py-1 text-xs text-ink-600 focus:outline-none focus:ring-2 focus:ring-sage-600"
          >
            {ROLE_OPTIONS.map((r) => (
              <option key={r} value={r}>
                {r}
              </option>
            ))}
          </select>
        </span>
      ),
    },
    {
      key: 'actions',
      header: '',
      className: 'w-12',
      render: (u) => (
        <button
          onClick={() => handleDelete(u)}
          title={`Remove ${u.name}`}
          className="rounded-md p-1.5 text-ink-300 transition-colors hover:bg-rust-100 hover:text-rust-600"
        >
          <Trash2 size={15} />
        </button>
      ),
    },
  ];

  return (
    <div className="space-y-6">
      <SectionHeader
        level={1}
        title="Team"
        subtitle="Who's on the measurement program — roles are used for attribution and sign-off (not authentication)."
        actions={
          <Button onClick={() => setShowAdd(true)}>
            <UserPlus size={15} />
            Add member
          </Button>
        }
      />

      {isLoading ? (
        <div className="rounded-lg border border-line-200 bg-white px-6 py-10 text-center text-sm text-ink-400 shadow-sm">
          Loading roster…
        </div>
      ) : (
        <DataTable
          columns={columns}
          rows={users}
          rowKey={(u) => u.user_id}
          empty={
            <EmptyState
              icon={Users}
              title="No teammates yet"
              description="No teammates yet — add the people who plan, run, and sign off on experiments"
              action={
                <Button onClick={() => setShowAdd(true)}>
                  <UserPlus size={15} />
                  Add member
                </Button>
              }
            />
          }
        />
      )}

      {showAdd && <AddMemberModal onClose={() => setShowAdd(false)} />}
    </div>
  );
}
