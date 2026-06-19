import { useState } from 'react';
import { Copy, Mail, Trash2, UserPlus } from 'lucide-react';
import { Button, Card } from '../../components/ui';
import { useCreateInvite, useInvites, useRevokeInvite } from '../../api/hooks/useAdmin';
import type { OrgRole } from '../../api/services/accountService';
import type { InviteResult } from '../../api/services/adminService';

const ROLE_OPTIONS: OrgRole[] = ['admin', 'analyst', 'viewer'];
const inputCls =
  'w-full rounded-md border border-line-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-sage-600';

function InviteResultBox({ result }: { result: InviteResult }) {
  const [copied, setCopied] = useState(false);
  const copy = () => {
    navigator.clipboard?.writeText(result.token).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  };
  return (
    <div className="mt-3 rounded-md border border-sage-300 bg-sage-100/60 px-3 py-2.5">
      <p className="text-sm text-sage-800">
        Invite created for <span className="font-medium">{result.email}</span> ({result.role}). Share this token —
        they set their password to join.
      </p>
      <div className="mt-2 flex items-center gap-2">
        <input readOnly value={result.token} className="w-full rounded border border-line-300 bg-white px-2 py-1 font-mono text-xs text-ink-700" />
        <Button type="button" variant="secondary" size="sm" onClick={copy}>
          <Copy size={13} />
          {copied ? 'Copied' : 'Copy'}
        </Button>
      </div>
    </div>
  );
}

export function InvitePanel() {
  const [email, setEmail] = useState('');
  const [role, setRole] = useState<OrgRole>('analyst');
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<InviteResult | null>(null);

  const create = useCreateInvite();
  const { data: invites = [] } = useInvites();
  const revoke = useRevokeInvite();

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email.trim() || create.isPending) return;
    setError(null);
    try {
      const res = await create.mutateAsync({ email: email.trim(), role });
      setResult(res);
      setEmail('');
    } catch (err: unknown) {
      const detail =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ??
        (err instanceof Error ? err.message : 'Could not create the invite.');
      setError(detail);
    }
  };

  return (
    <div className="space-y-4">
      <Card padding="lg">
        <h3 className="flex items-center gap-2 font-display text-base font-semibold text-ink-900">
          <UserPlus size={16} className="text-ink-400" />
          Invite a teammate
        </h3>
        <form onSubmit={submit} className="mt-3 flex flex-wrap items-end gap-3">
          <div className="min-w-[16rem] flex-1">
            <label className="mb-1 block text-sm font-medium text-ink-700">Email</label>
            <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="teammate@company.com" className={inputCls} />
          </div>
          <div>
            <label className="mb-1 block text-sm font-medium text-ink-700">Role</label>
            <select value={role} onChange={(e) => setRole(e.target.value as OrgRole)} className={inputCls}>
              {ROLE_OPTIONS.map((r) => <option key={r} value={r}>{r}</option>)}
            </select>
          </div>
          <Button type="submit" disabled={!email.trim() || create.isPending}>
            {create.isPending ? 'Creating…' : 'Create invite'}
          </Button>
        </form>
        {error && <p className="mt-3 rounded-md border border-rust-600/30 bg-rust-100 px-3 py-2 text-sm text-rust-700">{error}</p>}
        {result && <InviteResultBox result={result} />}
      </Card>

      {invites.length > 0 && (
        <Card padding="md">
          <h4 className="mb-2 text-xs font-bold uppercase tracking-wider text-ink-400">Pending invites</h4>
          <ul className="divide-y divide-line-200">
            {invites.map((inv) => (
              <li key={inv.token} className="flex items-center justify-between gap-3 py-2">
                <span className="flex items-center gap-2 text-sm text-ink-800">
                  <Mail size={14} className="text-ink-400" />
                  {inv.email}
                  <span className="rounded-full bg-cream-200 px-2 py-0.5 text-xs text-ink-500">{inv.role}</span>
                </span>
                <button
                  onClick={() => revoke.mutate(inv.token)}
                  title="Revoke invite"
                  className="rounded-md p-1.5 text-ink-300 transition-colors hover:bg-rust-100 hover:text-rust-600"
                >
                  <Trash2 size={15} />
                </button>
              </li>
            ))}
          </ul>
        </Card>
      )}
    </div>
  );
}
