import { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useForm } from 'react-hook-form';
import { Card, Title, Text, TextInput, Button, Select, SelectItem } from '@tremor/react';
import { KeyIcon, ExclamationCircleIcon, ChartBarIcon, CloudIcon } from '@heroicons/react/24/outline';
import { useAuthStore } from '../../stores/authStore';
import { checkApiHealth, getModelConfig, getVertexModels, getLmStudioModels, SERVER_MANAGED_KEY, type ServerModelConfig, type VertexModel, type LmStudioModel } from '../../api/client';
import { MODELS, PROVIDER_LABELS, getModelsByProvider, type Provider } from '../../constants/models';

interface LoginFormData {
  apiKey: string;
  modelName: string;
}

export function LoginPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const { isAuthenticated, setApiKey, setBaseUrl, isValidating, validationError } = useAuthStore();
  const [apiStatus, setApiStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  const [serverConfig, setServerConfig] = useState<ServerModelConfig | null>(null);
  const [vertexModels, setVertexModels] = useState<VertexModel[]>([]);
  const [lmStudioModels, setLmStudioModels] = useState<LmStudioModel[]>([]);
  // Editable LM Studio base URL (defaults to the server's configured value).
  const [localBaseUrl, setLocalBaseUrl] = useState<string>('');
  // The model chosen in server-managed mode (Vertex or a local LM Studio
  // endpoint): either a discovered id or a free-text id.
  const [vertexModel, setVertexModel] = useState<string>('');
  const [customModel, setCustomModel] = useState<string>('');

  const {
    register,
    handleSubmit,
    setValue,
    watch,
    formState: { errors },
  } = useForm<LoginFormData>({
    defaultValues: {
      modelName: 'claude-sonnet-4-6'
    }
  });

  const selectedModel = watch('modelName');

  // Get the intended destination from location state
  const from = (location.state as { from?: { pathname: string } })?.from?.pathname || '/dashboard';

  // Check API health and active model configuration on mount
  useEffect(() => {
    checkApiHealth().then((healthy) => {
      setApiStatus(healthy ? 'online' : 'offline');
    });
    getModelConfig().then((cfg) => {
      setServerConfig(cfg);
      if (cfg) setVertexModel(cfg.model);
      if (cfg?.is_local_endpoint) setLocalBaseUrl(cfg.base_url || 'http://localhost:1234/v1');
    });
  }, []);

  // When the server uses Vertex AI, discover the selectable models so the user
  // can pick one instead of being locked to the server's default.
  useEffect(() => {
    if (serverConfig?.uses_vertex) {
      getVertexModels().then((res) => {
        if (res?.models?.length) setVertexModels(res.models);
      });
    }
  }, [serverConfig?.uses_vertex]);

  // When the server points at a local LM Studio endpoint, discover the loaded
  // models (at the chosen base URL) so the user can pick which one to chat with.
  useEffect(() => {
    if (!serverConfig?.is_local_endpoint) return;
    const t = setTimeout(() => {
      getLmStudioModels(localBaseUrl || undefined).then((res) => {
        setLmStudioModels(res?.models ?? []);
      });
    }, 300);
    return () => clearTimeout(t);
  }, [serverConfig?.is_local_endpoint, localBaseUrl]);

  // Redirect if already authenticated
  useEffect(() => {
    if (isAuthenticated) {
      navigate(from, { replace: true });
    }
  }, [isAuthenticated, navigate, from]);

  // The server manages its own credentials (Vertex AI / ADC, or a server-side
  // env key), so no user-entered API key is required.
  const serverManaged = serverConfig != null && !serverConfig.requires_api_key;

  // Options for the Vertex model dropdown: the server default first, then any
  // discovered models (deduped). Free-text entry is always available alongside.
  const vertexOptions = (() => {
    const ids = new Set<string>();
    const opts: { id: string; label: string }[] = [];
    if (serverConfig?.model) {
      ids.add(serverConfig.model);
      opts.push({ id: serverConfig.model, label: `${serverConfig.model} (default)` });
    }
    for (const m of vertexModels) {
      if (ids.has(m.id)) continue;
      ids.add(m.id);
      const tag = m.family === 'claude' ? 'Claude' : m.family === 'gemini' ? 'Gemini' : m.family;
      const suffix = m.source === 'catalog' ? ' (catalog)' : '';
      opts.push({ id: m.id, label: `${tag} — ${m.display_name}${suffix}` });
    }
    return opts;
  })();

  // Options for the LM Studio model dropdown: the server default first, then the
  // models LM Studio currently has loaded.
  const lmStudioOptions = (() => {
    const ids = new Set<string>();
    const opts: { id: string; label: string }[] = [];
    if (serverConfig?.model) {
      ids.add(serverConfig.model);
      opts.push({ id: serverConfig.model, label: `${serverConfig.model} (default)` });
    }
    for (const m of lmStudioModels) {
      if (ids.has(m.id)) continue;
      ids.add(m.id);
      opts.push({ id: m.id, label: m.display_name });
    }
    return opts;
  })();

  const onSubmit = async (data: LoginFormData) => {
    const success = await setApiKey(data.apiKey.trim(), data.modelName);
    if (success) {
      navigate(from, { replace: true });
    }
  };

  const onServerManagedSignIn = async () => {
    // Free-text id (pasted from the console) wins, then the dropdown selection,
    // then the server default.
    const model = customModel.trim() || vertexModel || serverConfig?.model || selectedModel;
    // For a local endpoint, persist the chosen base URL so chat requests target it.
    if (serverConfig?.is_local_endpoint) {
      setBaseUrl(localBaseUrl.trim() || null);
    }
    const success = await setApiKey(SERVER_MANAGED_KEY, model);
    if (success) {
      navigate(from, { replace: true });
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full space-y-8">
        {/* Logo and title */}
        <div className="text-center">
          <div className="flex justify-center">
            <ChartBarIcon className="h-16 w-16 text-blue-600" />
          </div>
          <Title className="mt-4">MMM Studio</Title>
          <Text className="mt-2">
            Marketing Mix Modeling with Bayesian Inference
          </Text>
        </div>

        {/* Login card */}
        <Card className="mt-8">
          {serverManaged ? (
            // ── Server-managed credentials (Vertex AI / ADC) ──────────────
            <div className="space-y-6">
              <div className="flex items-start gap-3 p-3 bg-blue-50 rounded-md">
                <CloudIcon className="h-6 w-6 text-blue-500 flex-shrink-0" />
                <div>
                  <h3 className="text-sm font-medium text-blue-800">
                    {serverConfig?.is_local_endpoint ? 'Local model (LM Studio)' : 'Server-managed credentials'}
                  </h3>
                  <p className="mt-1 text-sm text-blue-700">
                    {serverConfig?.is_local_endpoint
                      ? `Using a local OpenAI-compatible server at ${serverConfig?.base_url ?? 'http://localhost:1234/v1'}. No API key needed.`
                      : serverConfig?.uses_adc
                      ? 'Authenticating to Vertex AI with Application Default Credentials. No API key needed.'
                      : 'The server provides the model credentials. No API key needed.'}
                  </p>
                </div>
              </div>

              <dl className="text-sm divide-y divide-gray-100">
                <div className="flex justify-between py-2">
                  <dt className="text-gray-500">Provider</dt>
                  <dd className="font-medium text-gray-900">{serverConfig?.provider}</dd>
                </div>
                <div className="flex justify-between py-2">
                  <dt className="text-gray-500">Model</dt>
                  <dd className="font-medium text-gray-900 text-right break-all">{serverConfig?.model}</dd>
                </div>
                {serverConfig?.location && (
                  <div className="flex justify-between py-2">
                    <dt className="text-gray-500">Region</dt>
                    <dd className="font-medium text-gray-900">{serverConfig.location}</dd>
                  </div>
                )}
                {serverConfig?.project && (
                  <div className="flex justify-between py-2">
                    <dt className="text-gray-500">GCP Project</dt>
                    <dd className="font-medium text-gray-900 text-right break-all">{serverConfig.project}</dd>
                  </div>
                )}
              </dl>

              {serverConfig?.uses_vertex && (
                <div className="space-y-3">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Model
                    </label>
                    <Select
                      value={customModel ? '' : vertexModel}
                      onValueChange={(v) => { setVertexModel(v); setCustomModel(''); }}
                    >
                      {vertexOptions.map((o) => (
                        <SelectItem key={o.id} value={o.id}>{o.label}</SelectItem>
                      ))}
                    </Select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Or enter a model id
                    </label>
                    <TextInput
                      placeholder="e.g. claude-sonnet-4-5@20250929"
                      value={customModel}
                      onChange={(e) => setCustomModel(e.target.value)}
                    />
                    <p className="mt-1 text-xs text-gray-500">
                      Paste an exact id from your Vertex Model Garden console (overrides the dropdown).
                    </p>
                  </div>
                </div>
              )}

              {serverConfig?.is_local_endpoint && (
                <div className="space-y-3">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      LM Studio server URL
                    </label>
                    <TextInput
                      placeholder="http://localhost:1234/v1"
                      value={localBaseUrl}
                      onChange={(e) => setLocalBaseUrl(e.target.value)}
                    />
                    <p className="mt-1 text-xs text-gray-500">
                      The base URL of LM Studio's local server (Developer tab → Start Server).
                    </p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Loaded model
                    </label>
                    <Select
                      value={customModel ? '' : vertexModel}
                      onValueChange={(v) => { setVertexModel(v); setCustomModel(''); }}
                    >
                      {lmStudioOptions.map((o) => (
                        <SelectItem key={o.id} value={o.id}>{o.label}</SelectItem>
                      ))}
                    </Select>
                    <p className="mt-1 text-xs text-gray-500">
                      {lmStudioModels.length > 0
                        ? 'Models currently loaded in LM Studio.'
                        : 'LM Studio not detected — start it and load a model, or type an id below.'}
                    </p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Or enter a model id
                    </label>
                    <TextInput
                      placeholder="e.g. qwen2.5-7b-instruct"
                      value={customModel}
                      onChange={(e) => setCustomModel(e.target.value)}
                    />
                    <p className="mt-1 text-xs text-gray-500">
                      Use the model identifier shown in LM Studio (overrides the dropdown).
                    </p>
                  </div>
                </div>
              )}

              {validationError && (
                <p className="text-sm text-red-600">{validationError}</p>
              )}

              <Button
                type="button"
                className="w-full"
                loading={isValidating}
                disabled={apiStatus === 'offline'}
                onClick={onServerManagedSignIn}
              >
                {isValidating ? 'Connecting...' : 'Continue'}
              </Button>
            </div>
          ) : (
            // ── API-key entry (direct providers) ──────────────────────────
            <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
              <div>
                <label htmlFor="modelName" className="block text-sm font-medium text-gray-700 mb-1">
                  LLM Provider
                </label>
                <Select
                  id="modelName"
                  value={selectedModel}
                  onValueChange={(value) => setValue('modelName', value)}
                >
                  {(Object.entries(getModelsByProvider()) as [Provider, typeof MODELS[number][]][]).map(
                    ([provider, models]) =>
                      models.map((m) => (
                        <SelectItem key={m.id} value={m.id}>
                          {PROVIDER_LABELS[provider]} — {m.label}
                        </SelectItem>
                      ))
                  )}
                </Select>
              </div>

              <div>
                <label htmlFor="apiKey" className="block text-sm font-medium text-gray-700">
                  API Key
                </label>
                <div className="mt-1 relative">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                    <KeyIcon className="h-5 w-5 text-gray-400" />
                  </div>
                  <TextInput
                    id="apiKey"
                    type="password"
                    placeholder="Enter your API key"
                    className="pl-10"
                    error={!!errors.apiKey || !!validationError}
                    {...register('apiKey', {
                      required: 'API key is required',
                      minLength: {
                        value: 8,
                        message: 'API key must be at least 8 characters',
                      },
                    })}
                  />
                </div>
                {errors.apiKey && (
                  <p className="mt-2 text-sm text-red-600">{errors.apiKey.message}</p>
                )}
                {validationError && (
                  <p className="mt-2 text-sm text-red-600">{validationError}</p>
                )}
              </div>

              <Button
                type="submit"
                className="w-full"
                loading={isValidating}
                disabled={apiStatus === 'offline'}
              >
                {isValidating ? 'Validating...' : 'Sign In'}
              </Button>
            </form>
          )}

          {/* API status indicator */}
          <div className="mt-6 pt-4 border-t border-gray-200">
            <div className="flex items-center justify-between text-sm">
              <span className="text-gray-500">API Status</span>
              <div className="flex items-center">
                {apiStatus === 'checking' && (
                  <>
                    <span className="h-2 w-2 rounded-full bg-yellow-400 animate-pulse mr-2" />
                    <span className="text-yellow-600">Checking...</span>
                  </>
                )}
                {apiStatus === 'online' && (
                  <>
                    <span className="h-2 w-2 rounded-full bg-green-400 mr-2" />
                    <span className="text-green-600">Online</span>
                  </>
                )}
                {apiStatus === 'offline' && (
                  <>
                    <span className="h-2 w-2 rounded-full bg-red-400 mr-2" />
                    <span className="text-red-600">Offline</span>
                  </>
                )}
              </div>
            </div>
          </div>

          {apiStatus === 'offline' && (
            <div className="mt-4 p-3 bg-red-50 rounded-md">
              <div className="flex">
                <ExclamationCircleIcon className="h-5 w-5 text-red-400" />
                <div className="ml-3">
                  <h3 className="text-sm font-medium text-red-800">
                    API server is not reachable
                  </h3>
                  <p className="mt-1 text-sm text-red-700">
                    Make sure the backend server is running at http://localhost:8000
                  </p>
                </div>
              </div>
            </div>
          )}
        </Card>

        {/* Help text */}
        <p className="text-center text-sm text-gray-500">
          {serverManaged
            ? 'Credentials are configured on the server.'
            : 'Need an API key? Contact your administrator or check the documentation.'}
        </p>
      </div>
    </div>
  );
}

export default LoginPage;
