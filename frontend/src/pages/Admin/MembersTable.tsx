import { useState } from 'react';
import { clsx } from 'clsx';
import { Trash2 } from 'lucide-react';
import { DataTable, type Column } from '../../components/ui';
import { useMembers, useRemoveMember, useSetMemberRole } from '../../api/hooks/useAdmin';
import { useMe } from '../../api/hooks/useAccount';
import { apiErrorMessage } from '../../api/client';
import type { OrgMember } from '../../api/services/adminService';
import type { OrgRole } from '../../api/services/accountService';

const ROLE_CHIP: Record<OrgRole, string> = {
  owner: 'bg-sage-100 text-sage-800',
  admin: 'bg-gold-100 text-gold-700',
  analyst: 'bg-steel-100 text-steel-700',
  viewer: 'bg-cream-200 text-ink-600',
};

export function MembersTable() {
  const { data: members = [], isLoading } = useMembers();
  const { data: me } = useMe();
  const setRole = useSetMemberRole();
  const removeMember = useRemoveMember();
  const [error, setError] = useState<string | null>(null);

  const ownerCount = members.filter((m) => m.role === 'owner').length;
  // Only owners may grant/touch the owner role (matches the backend guard).
  const canManageOwner = me?.org_role === 'owner';
  const baseOptions: OrgRole[] = canManageOwner
    ? ['owner', 'admin', 'analyst', 'viewer']
    : ['admin', 'analyst', 'viewer'];

  const onMutationError = (fallback: string) => ({
    onSuccess: () => setError(null),
    onError: (e: unknown) => setError(apiErrorMessage(e, fallback)),
  });

  const handleRemove = (m: OrgMember) => {
    const who = m.name || m.email || m.user_id;
    if (window.confirm(`Remove ${who} from the organization? Their sessions end immediately.`)) {
      removeMember.mutate(m.user_id, onMutationError('Could not remove the member.'));
    }
  };

  const columns: Column<OrgMember>[] = [
    {
      key: 'name',
      header: 'Member',
      render: (m) => (
        <div className="min-w-0">
          <p className="truncate font-medium text-ink-900">
            {m.name || m.email || <span className="font-mono text-xs">{m.user_id}</span>}
            {m.user_id === me?.user_id && <span className="ml-1.5 text-xs text-ink-400">(you)</span>}
          </p>
          {m.name && m.email && <p className="truncate text-[11px] text-ink-400">{m.email}</p>}
        </div>
      ),
    },
    {
      key: 'role',
      header: 'Role',
      render: (m) => {
        const isLastOwner = m.role === 'owner' && ownerCount <= 1;
        const lockedOwner = m.role === 'owner' && !canManageOwner;
        const disabled = isLastOwner || lockedOwner || setRole.isPending;
        // Always include the current role so the <select> value stays valid.
        const opts = baseOptions.includes(m.role) ? baseOptions : [m.role, ...baseOptions];
        return (
          <span className="flex items-center gap-2">
            <span className={clsx('inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium', ROLE_CHIP[m.role])}>
              {m.role}
            </span>
            <select
              value={m.role}
              disabled={disabled}
              onChange={(e) =>
                setRole.mutate(
                  { userId: m.user_id, role: e.target.value as OrgRole },
                  onMutationError('Could not change the role.'),
                )
              }
              className="rounded-md border border-line-300 px-1.5 py-1 text-xs text-ink-600 focus:outline-none focus:ring-2 focus:ring-sage-600 disabled:opacity-40"
              title={
                isLastOwner
                  ? 'The last owner cannot be demoted'
                  : lockedOwner
                    ? 'Only an owner can change an owner'
                    : 'Change role'
              }
            >
              {opts.map((r) => (
                <option key={r} value={r}>{r}</option>
              ))}
            </select>
          </span>
        );
      },
    },
    {
      key: 'actions',
      header: '',
      className: 'w-12',
      render: (m) => {
        const blocked =
          m.user_id === me?.user_id ||
          (m.role === 'owner' && (ownerCount <= 1 || !canManageOwner));
        return (
          <button
            onClick={() => handleRemove(m)}
            disabled={blocked}
            title={blocked ? 'You cannot remove yourself, the last owner, or (as an admin) an owner' : 'Remove from org'}
            className="rounded-md p-1.5 text-ink-300 transition-colors hover:bg-rust-100 hover:text-rust-600 disabled:cursor-not-allowed disabled:opacity-30 disabled:hover:bg-transparent"
          >
            <Trash2 size={15} />
          </button>
        );
      },
    },
  ];

  if (isLoading) {
    return <div className="h-32 animate-pulse rounded-lg border border-line-200 bg-cream-100" />;
  }

  return (
    <div className="space-y-3">
      {error && (
        <p className="rounded-md border border-rust-600/30 bg-rust-100 px-3 py-2 text-sm text-rust-700">{error}</p>
      )}
      <DataTable columns={columns} rows={members} rowKey={(m) => m.user_id} />
    </div>
  );
}
