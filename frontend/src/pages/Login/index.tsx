import { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useForm } from 'react-hook-form';
import { AlertCircle, Cloud, KeyRound } from 'lucide-react';
import { APP_NAME } from '../../appIdentity';
import { useAuthStore } from '../../stores/authStore';
import { checkApiHealth, getModelConfig, getVertexModels, getLmStudioModels, SERVER_MANAGED_KEY, type ServerModelConfig, type VertexModel, type LmStudioModel } from '../../api/client';
import { MODELS, PROVIDER_LABELS, getModelsByProvider, type Provider } from '../../constants/models';
import { Button } from '../../components/ui';

interface LoginFormData {
  apiKey: string;
  modelName: string;
}

const inputClass =
  'w-full rounded-md border border-line-300 bg-white px-3 py-2 text-sm text-ink-900 placeholder:text-ink-300 focus:outline-none focus:ring-2 focus:ring-sage-600';

function FieldLabel({ children, htmlFor }: { children: React.ReactNode; htmlFor?: string }) {
  return (
    <label htmlFor={htmlFor} className="mb-1 block text-sm font-medium text-ink-700">
      {children}
    </label>
  );
}

export function LoginPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const { isAuthenticated, setApiKey, setBaseUrl, login, isValidating, validationError } = useAuthStore();
  // Optional JWT email+password sign-in (coexists with the LLM-key form).
  const authEnabled = import.meta.env.VITE_AUTH_ENABLED === 'true';
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loginError, setLoginError] = useState<string | null>(null);
  const [loggingIn, setLoggingIn] = useState(false);

  const onPasswordSignIn = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoginError(null);
    setLoggingIn(true);
    try {
      const ok = await login(email.trim(), password);
      if (ok) {
        navigate(from, { replace: true });
      } else {
        setLoginError('Invalid email or password.');
      }
    } finally {
      setLoggingIn(false);
    }
  };
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

  useEffect(() => {
    document.title = `Sign in · ${APP_NAME}`;
  }, []);

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
  const from = (location.state as { from?: { pathname: string } })?.from?.pathname || '/program';

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
    <div className="flex min-h-screen items-center justify-center bg-cream-50 px-4 py-12 sm:px-6 lg:px-8">
      <div className="w-full max-w-md space-y-8">
        {/* Wordmark */}
        <div className="text-center">
          <img
            src="/augur.svg"
            alt=""
            className="mx-auto h-14 w-14 rounded-xl ring-1 ring-sage-300"
          />
          <h1 className="mt-5 font-display text-3xl font-semibold tracking-tight text-ink-900">
            {APP_NAME}
          </h1>
          <p className="mt-2 text-sm text-ink-400">
            Causality-centered marketing mix modeling — measure, experiment, calibrate.
          </p>
        </div>

        {/* Optional JWT email+password sign-in (additive; enabled via VITE_AUTH_ENABLED) */}
        {authEnabled && (
          <div className="rounded-xl border border-line-200 bg-white p-6 shadow-sm">
            <form onSubmit={onPasswordSignIn} className="space-y-4">
              <div>
                <FieldLabel htmlFor="email">Email</FieldLabel>
                <input
                  id="email"
                  type="email"
                  autoComplete="username"
                  placeholder="you@example.com"
                  className={inputClass}
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                />
              </div>
              <div>
                <FieldLabel htmlFor="password">Password</FieldLabel>
                <input
                  id="password"
                  type="password"
                  autoComplete="current-password"
                  placeholder="Enter your password"
                  className={`${inputClass} num`}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                />
              </div>
              {loginError && <p className="text-sm text-rust-600">{loginError}</p>}
              <Button
                type="submit"
                className="w-full"
                disabled={apiStatus === 'offline' || loggingIn || !email || !password}
              >
                {loggingIn ? 'Signing in...' : 'Sign in'}
              </Button>
            </form>
          </div>
        )}

        {/* Login card */}
        <div className="rounded-xl border border-line-200 bg-white p-6 shadow-sm">
          {serverManaged ? (
            // ── Server-managed credentials (Vertex AI / ADC) ──────────────
            <div className="space-y-6">
              <div className="flex items-start gap-3 rounded-md bg-steel-100 p-3">
                <Cloud className="h-5 w-5 flex-shrink-0 text-steel-600" />
                <div>
                  <h3 className="text-sm font-medium text-steel-700">
                    {serverConfig?.is_local_endpoint ? 'Local model (LM Studio)' : 'Server-managed credentials'}
                  </h3>
                  <p className="mt-1 text-sm text-steel-600">
                    {serverConfig?.is_local_endpoint
                      ? `Using a local OpenAI-compatible server at ${serverConfig?.base_url ?? 'http://localhost:1234/v1'}. No API key needed.`
                      : serverConfig?.uses_adc
                      ? 'Authenticating to Vertex AI with Application Default Credentials. No API key needed.'
                      : 'The server provides the model credentials. No API key needed.'}
                  </p>
                </div>
              </div>

              <dl className="divide-y divide-line-200 text-sm">
                <div className="flex justify-between py-2">
                  <dt className="text-ink-400">Provider</dt>
                  <dd className="font-medium text-ink-900">{serverConfig?.provider}</dd>
                </div>
                <div className="flex justify-between py-2">
                  <dt className="text-ink-400">Model</dt>
                  <dd className="break-all text-right font-medium text-ink-900 num">{serverConfig?.model}</dd>
                </div>
                {serverConfig?.location && (
                  <div className="flex justify-between py-2">
                    <dt className="text-ink-400">Region</dt>
                    <dd className="font-medium text-ink-900 num">{serverConfig.location}</dd>
                  </div>
                )}
                {serverConfig?.project && (
                  <div className="flex justify-between py-2">
                    <dt className="text-ink-400">GCP Project</dt>
                    <dd className="break-all text-right font-medium text-ink-900 num">{serverConfig.project}</dd>
                  </div>
                )}
              </dl>

              {serverConfig?.uses_vertex && (
                <div className="space-y-3">
                  <div>
                    <FieldLabel>Model</FieldLabel>
                    <select
                      className={inputClass}
                      value={customModel ? '' : vertexModel}
                      onChange={(e) => { setVertexModel(e.target.value); setCustomModel(''); }}
                    >
                      {vertexOptions.map((o) => (
                        <option key={o.id} value={o.id}>{o.label}</option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <FieldLabel>Or enter a model id</FieldLabel>
                    <input
                      className={inputClass}
                      placeholder="e.g. claude-sonnet-4-5@20250929"
                      value={customModel}
                      onChange={(e) => setCustomModel(e.target.value)}
                    />
                    <p className="mt-1 text-xs text-ink-400">
                      Paste an exact id from your Vertex Model Garden console (overrides the dropdown).
                    </p>
                  </div>
                </div>
              )}

              {serverConfig?.is_local_endpoint && (
                <div className="space-y-3">
                  <div>
                    <FieldLabel>LM Studio server URL</FieldLabel>
                    <input
                      className={inputClass}
                      placeholder="http://localhost:1234/v1"
                      value={localBaseUrl}
                      onChange={(e) => setLocalBaseUrl(e.target.value)}
                    />
                    <p className="mt-1 text-xs text-ink-400">
                      The base URL of LM Studio's local server (Developer tab → Start Server).
                    </p>
                  </div>
                  <div>
                    <FieldLabel>Loaded model</FieldLabel>
                    <select
                      className={inputClass}
                      value={customModel ? '' : vertexModel}
                      onChange={(e) => { setVertexModel(e.target.value); setCustomModel(''); }}
                    >
                      {lmStudioOptions.map((o) => (
                        <option key={o.id} value={o.id}>{o.label}</option>
                      ))}
                    </select>
                    <p className="mt-1 text-xs text-ink-400">
                      {lmStudioModels.length > 0
                        ? 'Models currently loaded in LM Studio.'
                        : 'LM Studio not detected — start it and load a model, or type an id below.'}
                    </p>
                  </div>
                  <div>
                    <FieldLabel>Or enter a model id</FieldLabel>
                    <input
                      className={inputClass}
                      placeholder="e.g. qwen2.5-7b-instruct"
                      value={customModel}
                      onChange={(e) => setCustomModel(e.target.value)}
                    />
                    <p className="mt-1 text-xs text-ink-400">
                      Use the model identifier shown in LM Studio (overrides the dropdown).
                    </p>
                  </div>
                </div>
              )}

              {validationError && (
                <p className="text-sm text-rust-600">{validationError}</p>
              )}

              <Button
                type="button"
                className="w-full"
                disabled={apiStatus === 'offline' || isValidating}
                onClick={onServerManagedSignIn}
              >
                {isValidating ? 'Connecting...' : 'Continue'}
              </Button>
            </div>
          ) : (
            // ── API-key entry (direct providers) ──────────────────────────
            <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
              <div>
                <FieldLabel htmlFor="modelName">LLM Provider</FieldLabel>
                <select
                  id="modelName"
                  className={inputClass}
                  value={selectedModel}
                  onChange={(e) => setValue('modelName', e.target.value)}
                >
                  {(Object.entries(getModelsByProvider()) as [Provider, typeof MODELS[number][]][]).map(
                    ([provider, models]) =>
                      models.map((m) => (
                        <option key={m.id} value={m.id}>
                          {PROVIDER_LABELS[provider]} — {m.label}
                        </option>
                      ))
                  )}
                </select>
              </div>

              <div>
                <FieldLabel htmlFor="apiKey">API Key</FieldLabel>
                <div className="relative mt-1">
                  <div className="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3">
                    <KeyRound className="h-4 w-4 text-ink-300" />
                  </div>
                  <input
                    id="apiKey"
                    type="password"
                    placeholder="Enter your API key"
                    className={`${inputClass} pl-9 num`}
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
                  <p className="mt-2 text-sm text-rust-600">{errors.apiKey.message}</p>
                )}
                {validationError && (
                  <p className="mt-2 text-sm text-rust-600">{validationError}</p>
                )}
              </div>

              <Button
                type="submit"
                className="w-full"
                disabled={apiStatus === 'offline' || isValidating}
              >
                {isValidating ? 'Validating...' : 'Sign In'}
              </Button>
            </form>
          )}

          {/* API status indicator */}
          <div className="mt-6 border-t border-line-200 pt-4">
            <div className="flex items-center justify-between text-sm">
              <span className="text-ink-400">API Status</span>
              <div className="flex items-center">
                {apiStatus === 'checking' && (
                  <>
                    <span className="mr-2 h-2 w-2 animate-pulse rounded-full bg-gold-300" />
                    <span className="text-gold-700">Checking...</span>
                  </>
                )}
                {apiStatus === 'online' && (
                  <>
                    <span className="mr-2 h-2 w-2 rounded-full bg-sage-600" />
                    <span className="text-sage-800">Online</span>
                  </>
                )}
                {apiStatus === 'offline' && (
                  <>
                    <span className="mr-2 h-2 w-2 rounded-full bg-rust-600" />
                    <span className="text-rust-600">Offline</span>
                  </>
                )}
              </div>
            </div>
          </div>

          {apiStatus === 'offline' && (
            <div className="mt-4 rounded-md bg-rust-100 p-3">
              <div className="flex">
                <AlertCircle className="h-5 w-5 text-rust-600" />
                <div className="ml-3">
                  <h3 className="text-sm font-medium text-rust-700">
                    API server is not reachable
                  </h3>
                  <p className="mt-1 text-sm text-rust-600">
                    Make sure the backend server is running at http://localhost:8000
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Help text */}
        <p className="text-center text-sm text-ink-400">
          {serverManaged
            ? 'Credentials are configured on the server.'
            : 'Need an API key? Contact your administrator or check the documentation.'}
        </p>
      </div>
    </div>
  );
}

export default LoginPage;
