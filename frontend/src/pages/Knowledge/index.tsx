import { useRef, useState } from 'react';
import {
  BookOpen,
  FileText,
  Loader2,
  Search,
  Trash2,
  UploadCloud,
} from 'lucide-react';
import { clsx } from 'clsx';
import {
  useDeleteKbDocument,
  useKbDocuments,
  useKbSearch,
  useUploadKbDocument,
} from '../../api/hooks/useKb';
import type { KbDocument, KbSearchResult } from '../../api/services/kbService';
import { useProjectStore } from '../../stores/projectStore';
import { Card, EmptyState, SectionHeader } from '../../components/ui';

const ACCEPT = '.txt,.md,.markdown,.csv,.pdf,.docx,.xlsx';

const STATUS_CHIP: Record<string, string> = {
  ready: 'bg-sage-100 text-sage-800',
  pending: 'bg-gold-100 text-gold-700',
  error: 'bg-rust-100 text-rust-700',
};

function fmtBytes(n: number | null): string {
  if (n == null) return '—';
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

function sourceLabel(doc: KbDocument): string | null {
  if (doc.meta?.source === 'onboarding') return 'project brief';
  if (doc.meta?.template) return 'template';
  return null;
}

function DocumentRow({
  doc,
  onDelete,
}: {
  doc: KbDocument;
  onDelete: (doc: KbDocument) => void;
}) {
  const source = sourceLabel(doc);
  return (
    <li className="flex items-start gap-3 px-4 py-3">
      <FileText className="mt-0.5 h-4 w-4 shrink-0 text-sage-700" strokeWidth={1.75} />
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
          <span className="truncate text-sm font-medium text-ink-900">{doc.name}</span>
          <span className="text-xs uppercase tracking-wider text-ink-400">{doc.kind}</span>
          {source && (
            <span className="rounded-full bg-steel-100 px-2 py-0.5 text-xs font-medium text-steel-700">
              {source}
            </span>
          )}
          <span
            className={clsx(
              'inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium',
              STATUS_CHIP[doc.status] ?? STATUS_CHIP.error,
            )}
          >
            {doc.status === 'pending' && <Loader2 className="h-3 w-3 animate-spin" />}
            {doc.status}
          </span>
        </div>
        <div className="mt-0.5 text-xs text-ink-400">
          <span className="num">{fmtBytes(doc.size_bytes)}</span>
          {doc.n_chunks > 0 && (
            <span>
              {' · '}
              <span className="num">{doc.n_chunks}</span> chunk{doc.n_chunks !== 1 ? 's' : ''}
            </span>
          )}
          {doc.status === 'error' && doc.error && (
            <span className="text-rust-700"> · {doc.error}</span>
          )}
        </div>
      </div>
      <button
        onClick={() => onDelete(doc)}
        title="Remove from knowledge base"
        className="rounded p-1.5 text-ink-300 transition-colors hover:bg-rust-100 hover:text-rust-700"
      >
        <Trash2 className="h-4 w-4" />
      </button>
    </li>
  );
}

export function KnowledgePage() {
  const { currentProjectId } = useProjectStore();
  const { data, isLoading } = useKbDocuments(currentProjectId);
  const upload = useUploadKbDocument(currentProjectId);
  const remove = useDeleteKbDocument(currentProjectId);
  const search = useKbSearch(currentProjectId);

  const fileRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const [asTemplate, setAsTemplate] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<KbSearchResult[] | null>(null);

  if (!currentProjectId) {
    return (
      <div className="mx-auto max-w-3xl py-16">
        <EmptyState
          icon={BookOpen}
          title="Pick a project to manage its knowledge base"
          description="Each project keeps its own library of briefs, reports, and reference data that the guide and workspace copilot ground their answers in. Use the project switcher in the header."
        />
      </div>
    );
  }

  const docs = data?.documents ?? [];

  const uploadFiles = async (files: FileList | File[]) => {
    setUploadError(null);
    for (const file of Array.from(files)) {
      try {
        await upload.mutateAsync({ file, template: asTemplate });
      } catch (e) {
        setUploadError(
          e instanceof Error && e.message
            ? `${file.name}: ${e.message}`
            : `Could not upload ${file.name}.`,
        );
      }
    }
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files?.length) uploadFiles(e.dataTransfer.files);
  };

  const deleteDoc = (doc: KbDocument) => {
    if (!confirm(`Remove "${doc.name}" from the knowledge base?`)) return;
    remove.mutate(doc.id);
  };

  const runSearch = async () => {
    const q = query.trim();
    if (!q) return;
    try {
      setResults(await search.mutateAsync({ query: q }));
    } catch {
      setResults([]);
    }
  };

  return (
    <div className="mx-auto max-w-5xl space-y-6">
      <SectionHeader
        level={1}
        title="Knowledge base"
        subtitle="Reports, briefs, and reference data for this project. Everything here is chunked and embedded so the project guide and workspace copilot can cite it."
      />

      {/* Upload */}
      <Card>
        <input
          ref={fileRef}
          type="file"
          multiple
          accept={ACCEPT}
          className="hidden"
          onChange={(e) => {
            if (e.target.files?.length) uploadFiles(e.target.files);
            e.target.value = '';
          }}
        />
        <div
          onClick={() => fileRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
          className={clsx(
            'flex cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed px-4 py-10 transition-colors',
            dragOver
              ? 'border-sage-600 bg-sage-100/60'
              : 'border-line-300 hover:border-sage-600/60 hover:bg-cream-100',
          )}
        >
          {upload.isPending ? (
            <>
              <Loader2 className="h-6 w-6 animate-spin text-sage-700" />
              <p className="text-sm text-ink-400">Uploading &amp; indexing…</p>
            </>
          ) : (
            <>
              <UploadCloud className="h-6 w-6 text-sage-700" strokeWidth={1.5} />
              <p className="text-sm font-medium text-ink-700">
                Drop files or click to upload
              </p>
              <p className="text-xs text-ink-400">txt · md · csv · pdf · docx · xlsx</p>
            </>
          )}
        </div>
        <label className="mt-3 flex items-center gap-2 text-sm text-ink-600">
          <input
            type="checkbox"
            checked={asTemplate}
            onChange={(e) => setAsTemplate(e.target.checked)}
            className="h-4 w-4 accent-sage-700"
          />
          Mark uploads as templates (report/deliverable formats the copilot can reuse)
        </label>
        {uploadError && (
          <p className="mt-3 rounded-md border border-rust-600/30 bg-rust-100 px-3 py-2 text-sm text-rust-700">
            {uploadError}
          </p>
        )}
      </Card>

      {/* Documents */}
      <div>
        <SectionHeader
          title="Documents"
          subtitle={`${docs.length} document${docs.length !== 1 ? 's' : ''} in this project's library`}
        />
        <Card padding="none" className="mt-3">
          {isLoading ? (
            <div className="flex items-center justify-center py-10 text-ink-300">
              <Loader2 className="h-5 w-5 animate-spin" />
            </div>
          ) : docs.length === 0 ? (
            <p className="px-4 py-8 text-center text-sm text-ink-400">
              No documents yet. Upload past reports, data dictionaries, media plans — anything
              the copilot should know about this client.
            </p>
          ) : (
            <ul className="divide-y divide-line-200">
              {docs.map((d) => (
                <DocumentRow key={d.id} doc={d} onDelete={deleteDoc} />
              ))}
            </ul>
          )}
        </Card>
      </div>

      {/* Search */}
      <div>
        <SectionHeader
          title="Search"
          subtitle="Check what the copilot would retrieve for a question"
        />
        <Card className="mt-3">
          <div className="flex items-center gap-2">
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && runSearch()}
              placeholder="Search the knowledge base…"
              className="flex-1 rounded-md border border-line-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-sage-600"
            />
            <button
              onClick={runSearch}
              disabled={search.isPending || !query.trim()}
              className="flex items-center gap-1.5 rounded-md bg-sage-700 px-3 py-2 text-sm font-medium text-white transition-colors hover:bg-sage-800 disabled:opacity-40"
            >
              {search.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Search className="h-4 w-4" />
              )}
              Search
            </button>
          </div>
          {results != null && (
            <div className="mt-4 space-y-2">
              {results.length === 0 ? (
                <p className="text-sm italic text-ink-400">No matches found.</p>
              ) : (
                results.map((r, i) => (
                  <div key={i} className="rounded-lg border border-line-200 bg-cream-100 px-3 py-2">
                    <div className="mb-1 flex items-baseline gap-2">
                      <FileText className="h-3.5 w-3.5 shrink-0 self-center text-sage-700" />
                      <span className="flex-1 truncate text-xs font-semibold text-ink-700">
                        {r.document}
                      </span>
                      <span className="text-xs text-ink-300 num">#{r.chunk_index}</span>
                      <span className="rounded bg-sage-100 px-1.5 py-0.5 text-xs text-sage-800 num">
                        {r.score.toFixed(3)}
                      </span>
                    </div>
                    <p className="line-clamp-6 whitespace-pre-wrap text-xs leading-relaxed text-ink-600">
                      {r.text}
                    </p>
                  </div>
                ))
              )}
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}

export default KnowledgePage;
