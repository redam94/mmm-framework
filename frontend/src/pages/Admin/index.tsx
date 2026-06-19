import { ShieldAlert, Users } from 'lucide-react';
import { EmptyState, SectionHeader } from '../../components/ui';
import { useMe } from '../../api/hooks/useAccount';
import { SeatUsage } from './SeatUsage';
import { MembersTable } from './MembersTable';
import { InvitePanel } from './InvitePanel';

const ADMIN_ROLES = new Set(['admin', 'owner']);

export function AdminPage() {
  const { data: me, isLoading } = useMe();
  const isAdmin = me ? ADMIN_ROLES.has(me.org_role) : false;

  return (
    <div className="space-y-6">
      <SectionHeader
        level={1}
        title="Admin"
        subtitle="Onboard teammates, manage roles, and track seat usage across the organization."
      />

      {isLoading ? (
        <div className="h-24 animate-pulse rounded-lg border border-line-200 bg-cream-100" />
      ) : !isAdmin ? (
        <EmptyState
          icon={ShieldAlert}
          title="Admin access required"
          description="Only organization owners and admins can manage members and invites. Ask an owner to grant you the admin role."
        />
      ) : (
        <div className="space-y-8">
          <SeatUsage />

          <section className="space-y-3">
            <h2 className="text-xs font-bold uppercase tracking-wider text-ink-400">Invitations</h2>
            <InvitePanel />
          </section>

          <section className="space-y-3">
            <div className="flex items-center gap-2">
              <Users size={16} className="text-ink-400" />
              <h2 className="text-xs font-bold uppercase tracking-wider text-ink-400">Members</h2>
            </div>
            <MembersTable />
          </section>
        </div>
      )}
    </div>
  );
}
