import { useState } from 'react';
import { Check, LogOut } from 'lucide-react';
import { Button, Card } from '../../components/ui';
import { useChangePassword } from '../../api/hooks/useAccount';
import { apiErrorMessage } from '../../api/client';
import { useAuthStore } from '../../stores/authStore';

const inputCls =
  'w-full rounded-md border border-line-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-sage-600';

export function SecuritySection() {
  const [current, setCurrent] = useState('');
  const [next, setNext] = useState('');
  const [confirm, setConfirm] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState(false);
  const change = useChangePassword();
  const logout = useAuthStore((s) => s.logout);

  const mismatch = confirm.length > 0 && next !== confirm;
  const canSubmit = current.length > 0 && next.length >= 8 && next === confirm && !change.isPending;

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!canSubmit) return;
    setError(null);
    setDone(false);
    try {
      await change.mutateAsync({ current, next });
      setDone(true);
      setCurrent('');
      setNext('');
      setConfirm('');
    } catch (err: unknown) {
      setError(apiErrorMessage(err, 'Could not change the password.'));
    }
  };

  return (
    <div className="max-w-2xl space-y-6">
      <Card padding="lg">
        <h3 className="font-display text-base font-semibold text-ink-900">Change password</h3>
        <p className="mt-1 text-sm text-ink-400">
          Changing your password signs out your other sessions; this one stays active.
        </p>
        <form onSubmit={submit} className="mt-4 space-y-4">
          <div>
            <label className="mb-1 block text-sm font-medium text-ink-700">Current password</label>
            <input type="password" value={current} onChange={(e) => setCurrent(e.target.value)} className={inputCls} autoComplete="current-password" />
          </div>
          <div>
            <label className="mb-1 block text-sm font-medium text-ink-700">New password</label>
            <input type="password" value={next} onChange={(e) => setNext(e.target.value)} className={inputCls} autoComplete="new-password" />
            <p className="mt-1 text-xs text-ink-400">At least 8 characters.</p>
          </div>
          <div>
            <label className="mb-1 block text-sm font-medium text-ink-700">Confirm new password</label>
            <input type="password" value={confirm} onChange={(e) => setConfirm(e.target.value)} className={inputCls} autoComplete="new-password" />
            {mismatch && <p className="mt-1 text-xs text-rust-600">Passwords don't match.</p>}
          </div>
          {error && (
            <p className="rounded-md border border-rust-600/30 bg-rust-100 px-3 py-2 text-sm text-rust-700">{error}</p>
          )}
          {done && (
            <p className="flex items-center gap-1.5 rounded-md border border-sage-300 bg-sage-100 px-3 py-2 text-sm text-sage-800">
              <Check size={15} /> Password updated.
            </p>
          )}
          <Button type="submit" disabled={!canSubmit}>
            {change.isPending ? 'Updating…' : 'Update password'}
          </Button>
        </form>
      </Card>

      <Card padding="lg" tone="cream" className="flex items-center justify-between">
        <div>
          <h3 className="font-display text-base font-semibold text-ink-900">Sign out</h3>
          <p className="mt-1 text-sm text-ink-400">End this session on this device.</p>
        </div>
        <Button variant="secondary" onClick={logout}>
          <LogOut size={15} />
          Sign out
        </Button>
      </Card>
    </div>
  );
}
