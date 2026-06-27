import { useRef, useState } from 'react';
import { Loader2, UploadCloud } from 'lucide-react';
import { fmtBytes } from '../../constants';

// Dropzone (pattern from KnowledgeTab) — stages a raw upload into the studio.
export function UploadStep({
  onUpload, uploading, error,
}: {
  onUpload: (file: File) => void;
  uploading: boolean;
  error: string | null;
}) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const [picked, setPicked] = useState<File | null>(null);

  const handle = (f: File | undefined) => {
    if (!f) return;
    setPicked(f);
    onUpload(f);
    if (fileRef.current) fileRef.current.value = '';
  };

  return (
    <div className="max-w-2xl mx-auto pt-8">
      <input ref={fileRef} type="file" className="hidden" accept=".csv,.tsv,.xlsx,.xls,.parquet"
        onChange={e => handle(e.target.files?.[0])} />
      <div
        onClick={() => fileRef.current?.click()}
        onDragOver={e => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={e => { e.preventDefault(); setDragOver(false); handle(e.dataTransfer.files?.[0]); }}
        className={`flex flex-col items-center justify-center gap-3 py-16 px-6 rounded-2xl border-2 border-dashed cursor-pointer transition-colors ${
          dragOver ? 'border-indigo-400 bg-indigo-50' : 'border-line-300 hover:border-indigo-300 hover:bg-cream-100'
        }`}
      >
        {uploading ? (
          <><Loader2 size={30} className="text-indigo-500 animate-spin" />
            <p className="text-sm text-ink-500">Staging {picked?.name ?? 'your file'}…</p></>
        ) : (
          <><UploadCloud size={30} className="text-indigo-400" />
            <p className="text-base text-ink-700 font-semibold">Drop a data file or click to upload</p>
            <p className="text-xs text-ink-300">csv · tsv · xlsx · parquet</p>
            {picked && <p className="text-xs text-ink-400">{picked.name} · {fmtBytes(picked.size)}</p>}</>
        )}
      </div>
      <p className="text-xs text-ink-300 text-center mt-4 max-w-md mx-auto">
        Your upload is <strong>staged</strong> — explore and clean it here first. It only becomes
        the model's working dataset when you choose <strong>Use as dataset</strong>.
      </p>
      {error && <p className="mt-3 text-xs text-red-600 bg-red-50 border border-red-200 rounded px-3 py-2 text-center">{error}</p>}
    </div>
  );
}
