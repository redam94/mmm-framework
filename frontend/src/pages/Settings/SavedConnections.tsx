import { useState } from 'react';
import { clsx } from 'clsx';
import { CheckCircle2, Database, Eye, Cloud, Plus, Trash2, XCircle } from 'lucide-react';
import { Button, Card } from '../../components/ui';
import { apiErrorMessage } from '../../api/client';
import { useProjectStore } from '../../stores/projectStore';
import {
  useConnections,
  useCreateConnection,
  useDeleteConnection,
  usePreviewConnection,
  useTestConnection,
} from '../../api/hooks/useConnections';
import type { DataConnection } from '../../api/services/connectionsService';

const inputCls =
  'w-full rounded-md border border-line-300 px-2.5 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-sage-600';

function summarize(c: DataConnection): string {
  const cfg = c.config as Record<string, string>;
  if (c.kind === 'gcs') return `gs://${cfg.bucket ?? '?'}/${cfg.object ?? ''}`;
  if (cfg.query) return cfg.query.length > 60 ? cfg.query.slice(0, 60) + '…' : cfg.query;
  return [cfg.dataset, cfg.table].filter(Boolean).join('.') || '(no table/query)';
}

function ConnectionRow({ projectId, conn }: { projectId: string; conn: DataConnection }) {
  const test = useTestConnection(projectId);
  const preview = usePreviewConnection(projectId);
  const del = useDeleteConnection(projectId);
  const [open, setOpen] = useState(false);

  const onDelete = () => {
    if (window.confirm(`Delete the connection “${conn.name}”?`)) del.mutate(conn.id);
  };

  const Icon = conn.kind === 'gcs' ? Cloud : Database;
  const pv = preview.data;

  return (
    <li className="py-3">
      <div className="flex items-start justify-between gap-3">
        <div className="flex min-w-0 items-start gap-2">
          <Icon size={16} className="mt-0.5 shrink-0 text-ink-400" />
          <div className="min-w-0">
            <p className="truncate text-sm font-medium text-ink-900">
              {conn.name}
              <span className="ml-2 rounded-full bg-cream-200 px-1.5 py-0.5 text-[11px] font-medium text-ink-500">{conn.kind}</span>
            </p>
            <p className="truncate font-mono text-[11px] text-ink-400">{summarize(conn)}</p>
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-1">
          <Button variant="ghost" size="sm" onClick={() => test.mutate(conn.id)} disabled={test.isPending}>
            {test.isPending ? 'Testing…' : 'Test'}
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => {
              setOpen(true);
              preview.mutate(conn.id);
            }}
            disabled={preview.isPending}
          >
            <Eye size={13} />
            Preview
          </Button>
          <button
            onClick={onDelete}
            title="Delete connection"
            className="rounded-md p-1.5 text-ink-300 transition-colors hover:bg-rust-100 hover:text-rust-600"
          >
            <Trash2 size={15} />
          </button>
        </div>
      </div>

      {(test.data || test.isError) && (
        <p
          className={clsx(
            'mt-2 flex items-center gap-1.5 rounded-md px-2 py-1 text-xs',
            test.data?.ok ? 'bg-sage-100 text-sage-800' : 'bg-rust-100 text-rust-700',
          )}
        >
          {test.data?.ok ? <CheckCircle2 size={13} /> : <XCircle size={13} />}
          {test.isError ? apiErrorMessage(test.error, 'Test failed.') : test.data?.detail}
        </p>
      )}

      {open && (preview.isPending || pv || preview.isError) && (
        <div className="mt-2 rounded-md border border-line-200 bg-cream-50 p-2 text-xs">
          {preview.isPending ? (
            <span className="text-ink-400">Reading a preview…</span>
          ) : preview.isError ? (
            <span className="text-rust-700">{apiErrorMessage(preview.error, 'Preview failed.')}</span>
          ) : pv ? (
            <div className="overflow-x-auto">
              <p className="mb-1 text-ink-400">{pv.n_preview} rows · {pv.columns.length} columns</p>
              <table className="min-w-full">
                <thead>
                  <tr className="text-left text-ink-500">
                    {pv.columns.slice(0, 8).map((c) => (
                      <th key={c} className="px-2 py-1 font-medium">{c}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {pv.rows.slice(0, 5).map((r, i) => (
                    <tr key={i} className="border-t border-line-200">
                      {pv.columns.slice(0, 8).map((c) => (
                        <td key={c} className="px-2 py-1 font-mono text-ink-700">{String(r[c] ?? '')}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : null}
        </div>
      )}
    </li>
  );
}

function AddConnectionForm({ projectId }: { projectId: string }) {
  const create = useCreateConnection(projectId);
  const [name, setName] = useState('');
  const [kind, setKind] = useState<'gcs' | 'bigquery'>('bigquery');
  const [f, setF] = useState<Record<string, string>>({});
  const [error, setError] = useState<string | null>(null);

  const set = (k: string) => (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) =>
    setF((prev) => ({ ...prev, [k]: e.target.value }));
  const clean = (obj: Record<string, string>) =>
    Object.fromEntries(Object.entries(obj).filter(([, v]) => v && v.trim()));

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim() || create.isPending) return;
    setError(null);
    const config =
      kind === 'gcs'
        ? clean({ bucket: f.bucket, object: f.object, prefix: f.prefix, project: f.project })
        : clean({ query: f.query, table: f.table, dataset: f.dataset, project: f.project, location: f.location });
    try {
      await create.mutateAsync({ name: name.trim(), kind, config });
      setName('');
      setF({});
    } catch (err) {
      setError(apiErrorMessage(err, 'Could not save the connection.'));
    }
  };

  return (
    <Card padding="md" tone="cream">
      <form onSubmit={submit} className="space-y-3">
        <div className="flex flex-wrap items-end gap-3">
          <div className="min-w-[12rem] flex-1">
            <label className="mb-1 block text-xs font-medium text-ink-600">Name</label>
            <input value={name} onChange={(e) => setName(e.target.value)} placeholder="weekly-spend" className={inputCls} />
          </div>
          <div>
            <label className="mb-1 block text-xs font-medium text-ink-600">Type</label>
            <select value={kind} onChange={(e) => setKind(e.target.value as 'gcs' | 'bigquery')} className={inputCls}>
              <option value="bigquery">BigQuery</option>
              <option value="gcs">Google Cloud Storage</option>
            </select>
          </div>
        </div>

        {kind === 'gcs' ? (
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <input value={f.bucket ?? ''} onChange={set('bucket')} placeholder="Bucket *" className={inputCls} />
            <input value={f.object ?? ''} onChange={set('object')} placeholder="Object path * (exports/mmm.csv)" className={inputCls} />
            <input value={f.prefix ?? ''} onChange={set('prefix')} placeholder="Prefix (optional)" className={inputCls} />
            <input value={f.project ?? ''} onChange={set('project')} placeholder="GCP project (optional, ADC default)" className={inputCls} />
          </div>
        ) : (
          <div className="space-y-3">
            <textarea
              value={f.query ?? ''}
              onChange={set('query')}
              placeholder="SQL query (or fill dataset + table below)"
              rows={2}
              className={inputCls}
            />
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              <input value={f.dataset ?? ''} onChange={set('dataset')} placeholder="Dataset (for a table read)" className={inputCls} />
              <input value={f.table ?? ''} onChange={set('table')} placeholder="Table (without a query)" className={inputCls} />
              <input value={f.project ?? ''} onChange={set('project')} placeholder="GCP project (optional)" className={inputCls} />
              <input value={f.location ?? ''} onChange={set('location')} placeholder="Location (optional, US)" className={inputCls} />
            </div>
          </div>
        )}

        {error && <p className="rounded-md border border-rust-600/30 bg-rust-100 px-2.5 py-1.5 text-xs text-rust-700">{error}</p>}
        <Button type="submit" size="sm" disabled={!name.trim() || create.isPending}>
          <Plus size={14} />
          {create.isPending ? 'Saving…' : 'Save connection'}
        </Button>
      </form>
    </Card>
  );
}

export function SavedConnections() {
  const projectId = useProjectStore((s) => s.currentProjectId);
  const { data: connections = [], isLoading } = useConnections(projectId);

  if (!projectId) {
    return (
      <Card padding="md" tone="cream">
        <p className="text-sm text-ink-500">Select a project to save and manage its data connections.</p>
      </Card>
    );
  }

  return (
    <div className="space-y-3">
      <AddConnectionForm projectId={projectId} />
      {isLoading ? (
        <div className="h-16 animate-pulse rounded-lg border border-line-200 bg-cream-100" />
      ) : connections.length === 0 ? (
        <p className="px-1 text-sm text-ink-400">No saved connections yet. Add one above — then in chat say “sync my <i>name</i> connection”.</p>
      ) : (
        <Card padding="md">
          <ul className="divide-y divide-line-200">
            {connections.map((c) => (
              <ConnectionRow key={c.id} projectId={projectId} conn={c} />
            ))}
          </ul>
        </Card>
      )}
    </div>
  );
}
