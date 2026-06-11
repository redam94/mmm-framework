import { useState } from 'react';
import { Check, Globe, Loader2, X } from 'lucide-react';
import { API_BASE, authHeaders } from '../../constants';
import { BrandPreview } from './BrandPreview';
import type { Branding } from './lib';

/** "Extract from website" flow: URL → POST /branding/extract → proposal
 * preview with Confirm / Discard. The server already saved the proposal as
 * unconfirmed (unconfirmed never styles output), so Discard needs no call;
 * Confirm copies the proposal into the form for the user to Save. */
export function ExtractFromWebsite({ projectId, apiKey, modelName, onConfirm }: {
  projectId: string;
  apiKey: string | null;
  modelName: string | null;
  onConfirm: (proposal: Branding) => void;
}) {
  const [url, setUrl] = useState('');
  const [extracting, setExtracting] = useState(false);
  const [proposal, setProposal] = useState<Branding | null>(null);
  const [error, setError] = useState<string | null>(null);

  const runExtract = async () => {
    const u = url.trim();
    if (!u || extracting) return;
    setExtracting(true);
    setError(null);
    setProposal(null);
    try {
      const res = await fetch(`${API_BASE}/projects/${encodeURIComponent(projectId)}/branding/extract`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...authHeaders(apiKey, modelName) },
        body: JSON.stringify({ url: u, save: true }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        setError(typeof data?.detail === 'string' ? data.detail : `Extraction failed (HTTP ${res.status})`);
        return;
      }
      setProposal(data as Branding);
    } catch {
      setError('Extraction failed — is the API running?');
    } finally {
      setExtracting(false);
    }
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <input
          type="url"
          value={url}
          onChange={e => setUrl(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && runExtract()}
          placeholder="https://client-website.com"
          className="flex-1 text-sm border border-gray-200 rounded-lg px-3 py-2 bg-white focus:outline-none focus:ring-2 focus:ring-indigo-400"
        />
        <button
          onClick={runExtract}
          disabled={extracting || !url.trim()}
          className="flex items-center gap-1.5 px-3 py-2 rounded-lg bg-indigo-600 text-white text-sm font-medium hover:bg-indigo-500 disabled:opacity-40 transition-colors"
        >
          {extracting ? <Loader2 size={14} className="animate-spin" /> : <Globe size={14} />} Extract
        </button>
      </div>
      {error && (
        <p className="text-xs text-red-600 bg-red-50 border border-red-200 rounded px-3 py-1.5">{error}</p>
      )}

      {proposal && (
        <div className="rounded-xl border border-amber-200 bg-amber-50/40 p-3 space-y-3">
          <div className="flex items-center gap-2">
            <span className="inline-block px-2.5 py-0.5 text-xs rounded-full border font-medium bg-amber-50 text-amber-700 border-amber-200">
              Proposed (unconfirmed)
            </span>
            {proposal.source_url && (
              <span className="text-[10px] text-gray-400 truncate">{proposal.source_url}</span>
            )}
          </div>
          <BrandPreview branding={proposal} />
          <div className="flex items-center gap-2">
            <button
              onClick={() => { onConfirm(proposal); setProposal(null); }}
              className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-indigo-600 text-white text-sm font-medium hover:bg-indigo-500 transition-colors"
            >
              <Check size={14} /> Use this branding
            </button>
            <button
              onClick={() => setProposal(null)}
              className="flex items-center gap-1.5 px-3 py-2 rounded-lg bg-white border border-gray-200 text-gray-600 text-sm font-medium hover:bg-gray-50 transition-colors"
            >
              <X size={14} /> Discard
            </button>
            <span className="text-[10px] text-gray-400">Confirm copies it into the form — hit Save to apply.</span>
          </div>
        </div>
      )}
    </div>
  );
}
