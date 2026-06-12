import { useState, useEffect } from 'react';
import { createPortal } from 'react-dom';
import {
  ChevronDownIcon,
  XMarkIcon,
  CloudIcon,
  ComputerDesktopIcon,
  KeyIcon,
} from '@heroicons/react/24/outline';
import { useAuthStore } from '../../stores/authStore';
import {
  getStoredKeyForProvider,
  getModelConfig,
  getVertexModels,
  getLmStudioModels,
  SERVER_MANAGED_KEY,
  type ServerModelConfig,
  type VertexModel,
  type LmStudioModel,
} from '../../api/client';
import { MODELS, getModelLabel } from '../../constants/models';

interface ModelSwitcherProps {
  /** 'light' for white backgrounds, 'dark' for dark headers */
  theme?: 'light' | 'dark';
}

// Backend provider catalog (these strings are what the server's `provider`
// enum + the X-Provider header expect). `needs` drives which credential field
// the UI shows; `ui` maps to the model catalog in constants/models.ts.
type Needs = 'key' | 'adc' | 'baseUrl';
const PROVIDERS: { id: string; label: string; needs: Needs; ui?: 'anthropic' | 'openai' | 'google' }[] = [
  { id: 'anthropic', label: 'Anthropic (API key)', needs: 'key', ui: 'anthropic' },
  { id: 'openai', label: 'OpenAI (API key)', needs: 'key', ui: 'openai' },
  { id: 'google_genai', label: 'Google Gemini (API key)', needs: 'key', ui: 'google' },
  { id: 'vertex_anthropic', label: 'Vertex AI · Claude', needs: 'adc' },
  { id: 'vertex_gemini', label: 'Vertex AI · Gemini', needs: 'adc' },
  { id: 'lmstudio', label: 'LM Studio (local)', needs: 'baseUrl' },
];

function metaFor(provider: string) {
  return PROVIDERS.find((p) => p.id === provider);
}

function dotFor(provider: string | undefined): string {
  if (!provider) return 'bg-gray-300';
  if (provider.startsWith('vertex')) return 'bg-blue-500';
  if (provider === 'lmstudio') return 'bg-teal-500';
  if (provider === 'openai') return 'bg-emerald-500';
  if (provider === 'google_genai') return 'bg-blue-500';
  return 'bg-amber-500'; // anthropic
}

export function ModelSwitcher({ theme = 'light' }: ModelSwitcherProps) {
  const { provider, modelName, baseUrl, setApiKey, setBaseUrl, setProvider } = useAuthStore();
  const [open, setOpen] = useState(false);
  const [serverConfig, setServerConfig] = useState<ServerModelConfig | null>(null);
  const [vertexModels, setVertexModels] = useState<VertexModel[]>([]);
  const [lmStudioModels, setLmStudioModels] = useState<LmStudioModel[]>([]);

  // Modal form state
  const [selProvider, setSelProvider] = useState('anthropic');
  const [selModel, setSelModel] = useState('');
  const [customModel, setCustomModel] = useState('');
  const [keyInput, setKeyInput] = useState('');
  const [baseUrlInput, setBaseUrlInput] = useState('http://localhost:1234/v1');
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const effectiveProvider = provider || serverConfig?.provider || 'anthropic';
  const serverVertexLocked = !!serverConfig?.uses_vertex;
  const currentLabel = modelName
    ? effectiveProvider.startsWith('vertex') || effectiveProvider === 'lmstudio'
      ? modelName
      : getModelLabel(modelName)
    : 'No model';

  useEffect(() => {
    getModelConfig().then(setServerConfig);
  }, []);

  // Initialise the form whenever the modal opens.
  useEffect(() => {
    if (!open) return;
    const p = effectiveProvider;
    setSelProvider(p);
    setSelModel(modelName || serverConfig?.model || '');
    setCustomModel('');
    setBaseUrlInput(baseUrl || serverConfig?.base_url || 'http://localhost:1234/v1');
    const m = metaFor(p);
    setKeyInput(m?.ui ? getStoredKeyForProvider(m.ui) || '' : '');
    setError(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  // Discover models for the selected provider (in the modal).
  useEffect(() => {
    if (!open) return;
    if (selProvider.startsWith('vertex')) {
      getVertexModels().then((res) => setVertexModels(res?.models ?? []));
    }
  }, [open, selProvider]);

  useEffect(() => {
    if (!open || selProvider !== 'lmstudio') return;
    const t = setTimeout(() => {
      getLmStudioModels(baseUrlInput || undefined).then((res) => setLmStudioModels(res?.models ?? []));
    }, 300);
    return () => clearTimeout(t);
  }, [open, selProvider, baseUrlInput]);

  const needs = metaFor(selProvider)?.needs ?? 'key';

  // Model options for the selected provider.
  const modelOptions: { id: string; label: string }[] = (() => {
    const out: { id: string; label: string }[] = [];
    const ids = new Set<string>();
    const push = (id: string, label: string) => {
      if (!id || ids.has(id)) return;
      ids.add(id);
      out.push({ id, label });
    };
    if (modelName && selProvider === effectiveProvider) push(modelName, `${modelName} (current)`);
    if (needs === 'key') {
      const ui = metaFor(selProvider)?.ui;
      MODELS.filter((m) => m.provider === ui).forEach((m) => push(m.id, m.label));
    } else if (needs === 'adc') {
      const fam = selProvider === 'vertex_anthropic' ? 'claude' : 'gemini';
      vertexModels.filter((m) => m.family === fam).forEach((m) => push(m.id, m.display_name));
    } else {
      lmStudioModels.forEach((m) => push(m.id, m.display_name));
    }
    return out;
  })();

  async function handleSave() {
    const model = (customModel.trim() || selModel).trim();
    if (!model) {
      setError('Pick or enter a model id.');
      return;
    }
    setSaving(true);
    setError(null);

    // Provider override is ignored by a Vertex-locked server, so don't set it there.
    setProvider(serverVertexLocked ? null : selProvider);
    setBaseUrl(needs === 'baseUrl' ? baseUrlInput.trim() || null : null);

    let key = SERVER_MANAGED_KEY;
    if (needs === 'key') {
      const ui = metaFor(selProvider)?.ui;
      key = keyInput.trim() || (ui ? getStoredKeyForProvider(ui) || '' : '');
      if (!key) {
        setSaving(false);
        setError('This provider needs an API key.');
        return;
      }
    }

    const ok = await setApiKey(key, model);
    setSaving(false);
    if (ok) {
      setOpen(false);
    } else {
      setError('Could not reach the server. Check it is running (and the URL/key).');
    }
  }

  const triggerClass =
    theme === 'dark'
      ? 'flex items-center gap-1.5 text-xs text-gray-400 hover:text-white transition-colors cursor-pointer select-none'
      : 'flex items-center gap-1.5 text-xs text-gray-500 hover:text-gray-800 transition-colors cursor-pointer select-none';

  return (
    <>
      <button className={triggerClass} onClick={() => setOpen(true)} title="Model settings">
        <span className={`h-1.5 w-1.5 rounded-full ${dotFor(effectiveProvider)}`} />
        <span className="truncate max-w-40">{currentLabel}</span>
        <ChevronDownIcon className="h-3 w-3 shrink-0" />
      </button>

      {open && createPortal(
        <div
          // Portaled to <body>: an ancestor with backdrop-filter (the sticky
          // Header) would otherwise become this fixed overlay's containing
          // block, clipping the dialog to the header strip.
          className="fixed inset-0 z-[100] flex items-center justify-center bg-ink-900/40 p-4"
          onClick={() => !saving && setOpen(false)}
        >
          <div
            className="w-full max-w-md bg-white rounded-2xl shadow-2xl overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between px-5 py-3 border-b border-gray-100">
              <h3 className="text-sm font-semibold text-gray-900">Model settings</h3>
              <button onClick={() => setOpen(false)} className="text-gray-400 hover:text-gray-700">
                <XMarkIcon className="h-5 w-5" />
              </button>
            </div>

            <div className="px-5 py-4 space-y-4 max-h-[70vh] overflow-y-auto">
              {/* Provider */}
              <div>
                <label className="block text-xs font-medium text-gray-600 mb-1">Provider</label>
                <select
                  value={selProvider}
                  disabled={serverVertexLocked}
                  onChange={(e) => {
                    setSelProvider(e.target.value);
                    setSelModel('');
                    setCustomModel('');
                    const m = metaFor(e.target.value);
                    setKeyInput(m?.ui ? getStoredKeyForProvider(m.ui) || '' : '');
                  }}
                  className="w-full text-sm border border-gray-200 rounded-lg px-3 py-2 bg-white focus:outline-none focus:ring-2 focus:ring-sage-600 disabled:bg-gray-100 disabled:text-gray-500"
                >
                  {PROVIDERS.map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.label}
                    </option>
                  ))}
                </select>
                {serverVertexLocked && (
                  <p className="mt-1 flex items-center gap-1 text-[11px] text-gray-400">
                    <CloudIcon className="h-3 w-3" />
                    Provider is locked to Vertex AI / ADC on this server; you can still change the model.
                  </p>
                )}
              </div>

              {/* LM Studio server URL */}
              {needs === 'baseUrl' && (
                <div>
                  <label className="block text-xs font-medium text-gray-600 mb-1">
                    <ComputerDesktopIcon className="inline h-3.5 w-3.5 mr-1 -mt-0.5" />
                    LM Studio server URL
                  </label>
                  <input
                    type="text"
                    placeholder="http://localhost:1234/v1"
                    value={baseUrlInput}
                    onChange={(e) => setBaseUrlInput(e.target.value)}
                    className="w-full text-sm border border-gray-200 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-teal-500"
                  />
                  <p className="mt-1 text-[11px] text-gray-400">
                    {lmStudioModels.length > 0
                      ? `${lmStudioModels.length} model(s) loaded at this URL.`
                      : 'Start LM Studio’s server (Developer → Start Server) and load a model.'}
                  </p>
                </div>
              )}

              {/* Model */}
              <div>
                <label className="block text-xs font-medium text-gray-600 mb-1">Model</label>
                <select
                  value={customModel ? '' : selModel}
                  onChange={(e) => {
                    setSelModel(e.target.value);
                    setCustomModel('');
                  }}
                  className="w-full text-sm border border-gray-200 rounded-lg px-3 py-2 bg-white focus:outline-none focus:ring-2 focus:ring-sage-600"
                >
                  <option value="" disabled>
                    {modelOptions.length ? 'Select a model…' : 'No models discovered — type one below'}
                  </option>
                  {modelOptions.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.label}
                    </option>
                  ))}
                </select>
                <input
                  type="text"
                  placeholder="…or type a model id"
                  value={customModel}
                  onChange={(e) => setCustomModel(e.target.value)}
                  className="mt-2 w-full text-sm border border-gray-200 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-sage-600"
                />
              </div>

              {/* API key (direct providers) */}
              {needs === 'key' && (
                <div>
                  <label className="block text-xs font-medium text-gray-600 mb-1">
                    <KeyIcon className="inline h-3.5 w-3.5 mr-1 -mt-0.5" />
                    API key
                  </label>
                  <input
                    type="password"
                    placeholder="sk-…"
                    value={keyInput}
                    onChange={(e) => setKeyInput(e.target.value)}
                    className="w-full text-sm border border-gray-200 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-sage-600"
                  />
                  <p className="mt-1 text-[11px] text-gray-400">
                    Stored locally in your browser and sent as the X-API-Key header.
                  </p>
                </div>
              )}

              {needs === 'adc' && (
                <p className="text-[11px] text-gray-400">
                  Vertex AI authenticates with the server’s Application Default Credentials — no key needed.
                </p>
              )}

              {error && <p className="text-xs text-red-600">{error}</p>}
            </div>

            <div className="flex justify-end gap-2 px-5 py-3 border-t border-gray-100 bg-gray-50">
              <button
                onClick={() => setOpen(false)}
                className="px-3 py-1.5 text-sm text-gray-600 hover:text-gray-900"
              >
                Cancel
              </button>
              <button
                onClick={handleSave}
                disabled={saving}
                className="px-4 py-1.5 bg-sage-700 text-white text-sm font-medium rounded-lg hover:bg-sage-800 disabled:opacity-50 transition-colors"
              >
                {saving ? 'Saving…' : 'Save'}
              </button>
            </div>
          </div>
        </div>,
        document.body,
      )}
    </>
  );
}

export default ModelSwitcher;
