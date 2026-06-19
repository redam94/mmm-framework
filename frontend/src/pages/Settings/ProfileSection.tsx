import { clsx } from 'clsx';
import { Building2, Mail, Shield, User } from 'lucide-react';
import { Card } from '../../components/ui';
import { useMe } from '../../api/hooks/useAccount';
import type { OrgRole } from '../../api/services/accountService';

const ROLE_CHIP: Record<OrgRole, string> = {
  owner: 'bg-sage-100 text-sage-800',
  admin: 'bg-gold-100 text-gold-700',
  analyst: 'bg-steel-100 text-steel-700',
  viewer: 'bg-cream-200 text-ink-600',
};

function Field({ icon: Icon, label, value }: { icon: typeof User; label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-start gap-3 py-3">
      <Icon size={16} className="mt-0.5 shrink-0 text-ink-400" />
      <div className="min-w-0">
        <p className="text-xs font-medium uppercase tracking-wider text-ink-400">{label}</p>
        <div className="mt-0.5 text-sm text-ink-900">{value}</div>
      </div>
    </div>
  );
}

export function ProfileSection() {
  const { data: me, isLoading, isError } = useMe();

  if (isLoading) {
    return <Card padding="lg"><p className="text-sm text-ink-400">Loading your profile…</p></Card>;
  }
  if (isError || !me) {
    return (
      <Card padding="lg" tone="cream">
        <p className="text-sm text-ink-600">
          You're signed in with an API key rather than an account, so there's no profile to show.
          Sign in with email + password to manage an organization and team.
        </p>
      </Card>
    );
  }

  return (
    <Card padding="lg" className="max-w-2xl divide-y divide-line-200">
      <Field icon={User} label="Name" value={me.name || <span className="text-ink-400">—</span>} />
      <Field icon={Mail} label="Email" value={me.email} />
      <Field icon={Building2} label="Organization" value={<span className="font-mono text-xs">{me.org_id}</span>} />
      <Field
        icon={Shield}
        label="Role"
        value={
          <span
            className={clsx(
              'inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium',
              ROLE_CHIP[me.org_role] ?? ROLE_CHIP.viewer,
            )}
          >
            {me.org_role}
          </span>
        }
      />
    </Card>
  );
}
