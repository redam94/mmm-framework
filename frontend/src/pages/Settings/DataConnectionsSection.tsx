import { clsx } from 'clsx';
import { CheckCircle2, Cloud, Database, Link2, PlugZap, XCircle } from 'lucide-react';
import { Card } from '../../components/ui';
import { useIntegrationsCatalog } from '../../api/hooks/useAccount';
import { SavedConnections } from './SavedConnections';
import type {
  AdPlatformCatalogEntry,
  DataSourceCatalogEntry,
} from '../../api/services/accountService';

const EASE_CHIP: Record<AdPlatformCatalogEntry['ease'], string> = {
  easy: 'bg-sage-100 text-sage-700',
  moderate: 'bg-gold-100 text-gold-700',
  hard: 'bg-rust-100 text-rust-700',
};

function InstalledBadge({ ok, label }: { ok: boolean; label?: string }) {
  return (
    <span
      className={clsx(
        'inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium',
        ok ? 'bg-sage-100 text-sage-700' : 'bg-cream-200 text-ink-500',
      )}
    >
      {ok ? <CheckCircle2 size={12} /> : <XCircle size={12} />}
      {label ?? (ok ? 'Installed' : 'Not installed')}
    </span>
  );
}

function DataSourceCard({ ds }: { ds: DataSourceCatalogEntry }) {
  const Icon = ds.kind === 'gcs' ? Cloud : Database;
  return (
    <Card padding="md">
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-2">
          <Icon size={18} className="text-ink-400" />
          <h4 className="font-display text-base font-semibold text-ink-900">{ds.label}</h4>
        </div>
        <InstalledBadge ok={ds.installed} />
      </div>
      <p className="mt-1.5 text-sm text-ink-600">{ds.description}</p>
      <p className="mt-2 text-xs text-ink-400">
        <span className="font-medium text-ink-500">Auth:</span> {ds.auth}
      </p>
      {!ds.installed && (
        <p className="mt-2 rounded bg-cream-100 px-2 py-1 font-mono text-[11px] text-ink-500">
          uv sync --extra {ds.install_extra}
        </p>
      )}
    </Card>
  );
}

function AdPlatformCard({ ap }: { ap: AdPlatformCatalogEntry }) {
  return (
    <Card padding="md">
      <div className="flex items-start justify-between gap-3">
        <h4 className="font-display text-base font-semibold text-ink-900">{ap.label}</h4>
        <div className="flex items-center gap-1.5">
          <span className={clsx('rounded-full px-2 py-0.5 text-xs font-medium', EASE_CHIP[ap.ease])}>{ap.ease}</span>
          <span className="rounded-full bg-cream-200 px-2 py-0.5 text-xs font-medium text-ink-500">{ap.status}</span>
        </div>
      </div>
      <p className="mt-1.5 text-sm text-ink-600">
        <span className="font-medium text-ink-500">Recommended:</span> {ap.recommended_path}
      </p>
      <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-ink-400">
        <span><span className="font-medium text-ink-500">Auth:</span> {ap.auth}</span>
        {ap.official_sdk && <InstalledBadge ok={ap.sdk_installed} label={`${ap.official_sdk}${ap.sdk_installed ? ' ✓' : ''}`} />}
      </div>
    </Card>
  );
}

export function DataConnectionsSection() {
  const { data, isLoading, isError } = useIntegrationsCatalog();

  if (isLoading) return <Card padding="lg"><p className="text-sm text-ink-400">Loading connectors…</p></Card>;
  if (isError || !data) {
    return (
      <Card padding="lg" tone="cream">
        <p className="text-sm text-ink-600">Couldn't load the integrations catalog.</p>
      </Card>
    );
  }

  return (
    <div className="space-y-8">
      <section className="space-y-3">
        <div className="flex items-center gap-2">
          <Link2 size={16} className="text-ink-400" />
          <h3 className="text-xs font-bold uppercase tracking-wider text-ink-400">Saved connections</h3>
        </div>
        <SavedConnections />
      </section>

      <section className="space-y-3">
        <div className="flex items-center gap-2">
          <Database size={16} className="text-ink-400" />
          <h3 className="text-xs font-bold uppercase tracking-wider text-ink-400">Data warehouses & storage</h3>
        </div>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          {data.data_sources.map((ds) => <DataSourceCard key={ds.kind} ds={ds} />)}
        </div>
        <p className="text-xs text-ink-400">
          Connectors authenticate with Application Default Credentials (ADC) — the same identity used
          for Vertex AI. From chat, the agent can pull data with <code className="font-mono">load_from_bigquery</code> /
          <code className="font-mono"> load_from_gcs</code>.
        </p>
      </section>

      <section className="space-y-3">
        <div className="flex items-center gap-2">
          <PlugZap size={16} className="text-ink-400" />
          <h3 className="text-xs font-bold uppercase tracking-wider text-ink-400">Ad platforms (ranked easiest-first)</h3>
        </div>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
          {data.ad_platforms.map((ap) => <AdPlatformCard key={ap.platform} ap={ap} />)}
        </div>
        <p className="text-xs text-ink-400">
          The lowest-effort path lands spend in BigQuery via a managed transfer, then reads it through
          the BigQuery connector — no API client to maintain. See
          <code className="font-mono"> technical-docs/ad-platform-integrations.md</code>.
        </p>
      </section>
    </div>
  );
}
