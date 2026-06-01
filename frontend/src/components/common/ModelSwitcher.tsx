import { useState, useRef, useEffect } from 'react';
import { CheckIcon, ChevronDownIcon, KeyIcon } from '@heroicons/react/24/outline';
import { useAuthStore } from '../../stores/authStore';
import { getStoredKeyForProvider } from '../../api/client';
import {
  MODELS,
  PROVIDER_LABELS,
  getModelsByProvider,
  getModelLabel,
  type Provider,
} from '../../constants/models';

interface ModelSwitcherProps {
  /** 'light' for white backgrounds, 'dark' for the sidebar */
  theme?: 'light' | 'dark';
}

const PROVIDER_COLORS: Record<Provider, string> = {
  anthropic: 'bg-amber-500',
  openai: 'bg-emerald-500',
  google: 'bg-blue-500',
};

const PROVIDER_TEXT_COLORS: Record<Provider, string> = {
  anthropic: 'text-amber-600',
  openai: 'text-emerald-600',
  google: 'text-blue-600',
};

export function ModelSwitcher({ theme = 'light' }: ModelSwitcherProps) {
  const { modelName, setApiKey, switchModel } = useAuthStore();
  const [open, setOpen] = useState(false);
  const [pendingModel, setPendingModel] = useState<string | null>(null);
  const [keyInput, setKeyInput] = useState('');
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const byProvider = getModelsByProvider();
  const currentLabel = modelName ? getModelLabel(modelName) : 'No model';

  // Close on outside click
  useEffect(() => {
    if (!open) return;
    function handler(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false);
        setPendingModel(null);
        setKeyInput('');
        setSaveError(null);
      }
    }
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [open]);

  function handleSelectModel(id: string) {
    if (id === modelName) {
      setOpen(false);
      return;
    }
    const switched = switchModel(id);
    if (switched) {
      setOpen(false);
      setPendingModel(null);
    } else {
      setPendingModel(id);
      setKeyInput('');
      setSaveError(null);
    }
  }

  async function handleSaveKey() {
    if (!pendingModel || !keyInput.trim()) return;
    setSaving(true);
    setSaveError(null);
    const success = await setApiKey(keyInput.trim(), pendingModel);
    setSaving(false);
    if (success) {
      setOpen(false);
      setPendingModel(null);
      setKeyInput('');
    } else {
      setSaveError('Invalid API key for this provider');
    }
  }

  const triggerClass =
    theme === 'dark'
      ? 'flex items-center gap-1.5 text-xs text-gray-400 hover:text-white transition-colors cursor-pointer select-none'
      : 'flex items-center gap-1.5 text-xs text-gray-500 hover:text-gray-800 transition-colors cursor-pointer select-none';

  return (
    <div ref={containerRef} className="relative">
      <button className={triggerClass} onClick={() => setOpen((v) => !v)}>
        {modelName && (
          <span
            className={`h-1.5 w-1.5 rounded-full ${PROVIDER_COLORS[
              MODELS.find((m) => m.id === modelName)?.provider ?? 'anthropic'
            ]}`}
          />
        )}
        <span className="truncate max-w-32">{currentLabel}</span>
        <ChevronDownIcon className="h-3 w-3 shrink-0" />
      </button>

      {open && (
        <div className="absolute bottom-full mb-2 left-0 z-50 w-72 bg-white rounded-xl border border-gray-200 shadow-xl overflow-hidden">
          <div className="px-3 py-2 border-b border-gray-100">
            <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider">
              Switch Model
            </p>
          </div>

          {(Object.entries(byProvider) as [Provider, typeof MODELS[number][]][]).map(
            ([provider, models]) => (
              <div key={provider}>
                <p
                  className={`px-3 pt-2 pb-1 text-xs font-medium ${PROVIDER_TEXT_COLORS[provider]}`}
                >
                  {PROVIDER_LABELS[provider]}
                </p>
                {models.map((m) => {
                  const hasKey = !!getStoredKeyForProvider(provider);
                  const isCurrent = m.id === modelName;
                  const isPending = m.id === pendingModel;

                  return (
                    <div key={m.id}>
                      <button
                        onClick={() => handleSelectModel(m.id)}
                        className={`w-full flex items-center gap-2 px-3 py-2 text-left hover:bg-gray-50 transition-colors ${
                          isCurrent ? 'bg-gray-50' : ''
                        }`}
                      >
                        <span
                          className={`h-1.5 w-1.5 rounded-full shrink-0 ${
                            hasKey ? PROVIDER_COLORS[provider] : 'bg-gray-300'
                          }`}
                          title={hasKey ? 'API key stored' : 'No API key stored'}
                        />
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium text-gray-800">{m.label}</p>
                          <p className="text-xs text-gray-400">{m.description}</p>
                        </div>
                        {isCurrent && <CheckIcon className="h-4 w-4 text-indigo-600 shrink-0" />}
                        {!hasKey && !isCurrent && (
                          <KeyIcon className="h-3.5 w-3.5 text-gray-300 shrink-0" />
                        )}
                      </button>

                      {/* Inline key input when no key stored */}
                      {isPending && (
                        <div className="px-3 pb-3 space-y-2 bg-gray-50 border-t border-gray-100">
                          <p className="text-xs text-gray-500 pt-2">
                            Enter your{' '}
                            <span className={`font-medium ${PROVIDER_TEXT_COLORS[provider]}`}>
                              {PROVIDER_LABELS[provider]}
                            </span>{' '}
                            API key to use {m.label}:
                          </p>
                          <input
                            autoFocus
                            type="password"
                            placeholder="sk-..."
                            value={keyInput}
                            onChange={(e) => setKeyInput(e.target.value)}
                            onKeyDown={(e) => e.key === 'Enter' && handleSaveKey()}
                            className="w-full text-sm border border-gray-200 rounded-lg px-3 py-1.5 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                          />
                          {saveError && (
                            <p className="text-xs text-red-500">{saveError}</p>
                          )}
                          <div className="flex gap-2">
                            <button
                              onClick={handleSaveKey}
                              disabled={!keyInput.trim() || saving}
                              className="flex-1 py-1.5 bg-indigo-600 text-white text-xs font-medium rounded-lg hover:bg-indigo-700 disabled:opacity-50 transition-colors"
                            >
                              {saving ? 'Saving…' : 'Save & Switch'}
                            </button>
                            <button
                              onClick={() => {
                                setPendingModel(null);
                                setKeyInput('');
                                setSaveError(null);
                              }}
                              className="px-3 py-1.5 text-xs text-gray-500 hover:text-gray-700 transition-colors"
                            >
                              Cancel
                            </button>
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )
          )}

          <div className="border-t border-gray-100 px-3 py-2">
            <p className="text-xs text-gray-400">
              <span className="inline-block h-1.5 w-1.5 rounded-full bg-gray-300 mr-1.5 align-middle" />
              No key stored &nbsp;·&nbsp;
              <span className="inline-block h-1.5 w-1.5 rounded-full bg-green-400 mr-1.5 align-middle" />
              Key saved
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

export default ModelSwitcher;
