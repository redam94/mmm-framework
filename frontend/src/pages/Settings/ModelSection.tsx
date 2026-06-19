import { useState } from 'react';
import { Check } from 'lucide-react';
import { Button, Card } from '../../components/ui';
import { useAuthStore } from '../../stores/authStore';

const inputCls =
  'w-full rounded-md border border-line-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-sage-600';

export function ModelSection() {
  const { provider, modelName, baseUrl, isValidating, validationError } = useAuthStore();
  const setApiKey = useAuthStore((s) => s.setApiKey);
  const setProvider = useAuthStore((s) => s.setProvider);
  const setBaseUrl = useAuthStore((s) => s.setBaseUrl);

  const [providerVal, setProviderVal] = useState(provider ?? '');
  const [modelVal, setModelVal] = useState(modelName ?? '');
  const [baseUrlVal, setBaseUrlVal] = useState(baseUrl ?? '');
  const [keyVal, setKeyVal] = useState('');
  const [saved, setSaved] = useState(false);

  const save = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaved(false);
    setProvider(providerVal.trim() || null);
    setBaseUrl(baseUrlVal.trim() || null);
    // A key change is validated against the server; model name is required for that.
    if (keyVal.trim() && modelVal.trim()) {
      const ok = await setApiKey(keyVal.trim(), modelVal.trim());
      if (ok) {
        setKeyVal('');
        setSaved(true);
      }
    } else {
      setSaved(true);
    }
  };

  return (
    <Card padding="lg" className="max-w-2xl">
      <h3 className="font-display text-base font-semibold text-ink-900">Model &amp; API</h3>
      <p className="mt-1 text-sm text-ink-400">
        The LLM the agent uses. Leave the API key blank when the server authenticates via Vertex AI /
        ADC or a local model — no key is needed then.
      </p>
      <div className="mt-3 flex flex-wrap gap-x-6 gap-y-1 text-sm">
        <span className="text-ink-400">Active provider: <span className="font-medium text-ink-800">{provider || 'server default'}</span></span>
        <span className="text-ink-400">Model: <span className="font-medium text-ink-800">{modelName || '—'}</span></span>
      </div>

      <form onSubmit={save} className="mt-4 space-y-4">
        <div>
          <label className="mb-1 block text-sm font-medium text-ink-700">Provider</label>
          <input value={providerVal} onChange={(e) => setProviderVal(e.target.value)} placeholder="anthropic / openai / google_genai / vertex_anthropic / lmstudio" className={inputCls} />
        </div>
        <div>
          <label className="mb-1 block text-sm font-medium text-ink-700">Model name</label>
          <input value={modelVal} onChange={(e) => setModelVal(e.target.value)} placeholder="e.g. claude-opus-4-8" className={inputCls} />
        </div>
        <div>
          <label className="mb-1 block text-sm font-medium text-ink-700">Base URL (optional)</label>
          <input value={baseUrlVal} onChange={(e) => setBaseUrlVal(e.target.value)} placeholder="e.g. http://localhost:1234/v1 (LM Studio)" className={inputCls} />
        </div>
        <div>
          <label className="mb-1 block text-sm font-medium text-ink-700">API key</label>
          <input type="password" value={keyVal} onChange={(e) => setKeyVal(e.target.value)} placeholder="(unchanged — leave blank to keep current / use ADC)" className={inputCls} autoComplete="off" />
        </div>
        {validationError && (
          <p className="rounded-md border border-rust-600/30 bg-rust-100 px-3 py-2 text-sm text-rust-700">{validationError}</p>
        )}
        {saved && !validationError && (
          <p className="flex items-center gap-1.5 rounded-md border border-sage-300 bg-sage-100 px-3 py-2 text-sm text-sage-800">
            <Check size={15} /> Settings saved.
          </p>
        )}
        <Button type="submit" disabled={isValidating}>
          {isValidating ? 'Validating…' : 'Save'}
        </Button>
      </form>
    </Card>
  );
}
