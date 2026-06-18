import axios from 'axios';
import type { AxiosError, AxiosInstance, AxiosResponse, InternalAxiosRequestConfig } from 'axios';
import type { ApiError } from './types';
import { getProviderForModel } from '../constants/models';

// API Base URL.
// - If VITE_API_URL is set, use it verbatim (absolute URL).
// - Otherwise in dev, use the relative "/api" prefix so requests go same-origin
//   through the Vite dev-server proxy (see vite.config.ts). This means only the
//   Vite port needs to be forwarded/tunneled — the backend is reachable through it.
// - In a production build with no override, fall back to the legacy absolute URL.
const API_BASE_URL =
  import.meta.env.VITE_API_URL || (import.meta.env.DEV ? '/api' : 'http://localhost:8000');

// Create axios instance
export const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Storage keys
const API_KEY_STORAGE_KEY = 'mmm_api_key';
const MODEL_NAME_STORAGE_KEY = 'mmm_model_name';
const PROVIDER_KEYS_STORAGE_KEY = 'mmm_provider_keys';
const BASE_URL_STORAGE_KEY = 'mmm_base_url';
const PROVIDER_STORAGE_KEY = 'mmm_provider';
// Strong "expert" tier selection (delegate_to_expert). Sent as X-Expert-* headers.
const EXPERT_MODEL_STORAGE_KEY = 'mmm_expert_model';
const EXPERT_PROVIDER_STORAGE_KEY = 'mmm_expert_provider';
const EXPERT_BASE_URL_STORAGE_KEY = 'mmm_expert_base_url';

// JWT bearer auth (optional, additive). When present, sent as an Authorization
// header alongside the existing X-API-Key flow. Absent => header omitted, so
// behavior is unchanged when JWT auth is disabled / the user is not logged in.
const ACCESS_TOKEN_STORAGE_KEY = 'mmm_access_token';
const REFRESH_TOKEN_STORAGE_KEY = 'mmm_refresh_token';

export function getAccessToken(): string | null {
  return localStorage.getItem(ACCESS_TOKEN_STORAGE_KEY);
}
export function getRefreshToken(): string | null {
  return localStorage.getItem(REFRESH_TOKEN_STORAGE_KEY);
}
export function setStoredTokens(access: string | null, refresh: string | null): void {
  if (access && access.trim()) {
    localStorage.setItem(ACCESS_TOKEN_STORAGE_KEY, access.trim());
  } else {
    localStorage.removeItem(ACCESS_TOKEN_STORAGE_KEY);
  }
  if (refresh && refresh.trim()) {
    localStorage.setItem(REFRESH_TOKEN_STORAGE_KEY, refresh.trim());
  } else {
    localStorage.removeItem(REFRESH_TOKEN_STORAGE_KEY);
  }
}
export function clearStoredTokens(): void {
  localStorage.removeItem(ACCESS_TOKEN_STORAGE_KEY);
  localStorage.removeItem(REFRESH_TOKEN_STORAGE_KEY);
}

// Authorization header for raw fetch() sites. Returns {} when no token is
// stored, so it can be spread/merged unconditionally without changing behavior.
export function bearerHeader(): Record<string, string> {
  const t = getAccessToken();
  return t ? { Authorization: 'Bearer ' + t } : {};
}

// Get API key and model from localStorage
export function getStoredApiKey(): string | null {
  return localStorage.getItem(API_KEY_STORAGE_KEY);
}
export function getStoredModelName(): string | null {
  return localStorage.getItem(MODEL_NAME_STORAGE_KEY);
}

// Optional backend provider override (e.g. 'lmstudio', 'openai', 'vertex_gemini').
// Sent as X-Provider; the server honors it only when it is NOT Vertex-locked.
export function getStoredProvider(): string | null {
  return localStorage.getItem(PROVIDER_STORAGE_KEY);
}
export function setStoredProvider(provider: string | null): void {
  if (provider && provider.trim()) {
    localStorage.setItem(PROVIDER_STORAGE_KEY, provider.trim());
  } else {
    localStorage.removeItem(PROVIDER_STORAGE_KEY);
  }
}

// Optional override for a local OpenAI-compatible endpoint (LM Studio). Sent as
// the X-Base-Url header; the server honors it only for the lmstudio provider.
export function getStoredBaseUrl(): string | null {
  return localStorage.getItem(BASE_URL_STORAGE_KEY);
}
export function setStoredBaseUrl(baseUrl: string | null): void {
  if (baseUrl && baseUrl.trim()) {
    localStorage.setItem(BASE_URL_STORAGE_KEY, baseUrl.trim());
  } else {
    localStorage.removeItem(BASE_URL_STORAGE_KEY);
  }
}

// Expert (strong) tier selection. Each is optional; when unset the server's
// configured expert (or the chat model) is used. The expert's API key reuses the
// per-provider bucket (getStoredKeyForProvider), so there is no separate store.
export function getStoredExpertModel(): string | null {
  return localStorage.getItem(EXPERT_MODEL_STORAGE_KEY);
}
export function getStoredExpertProvider(): string | null {
  return localStorage.getItem(EXPERT_PROVIDER_STORAGE_KEY);
}
export function getStoredExpertBaseUrl(): string | null {
  return localStorage.getItem(EXPERT_BASE_URL_STORAGE_KEY);
}
export function setStoredExpert(
  model: string | null,
  provider: string | null,
  baseUrl: string | null,
): void {
  const put = (k: string, v: string | null) =>
    v && v.trim() ? localStorage.setItem(k, v.trim()) : localStorage.removeItem(k);
  put(EXPERT_MODEL_STORAGE_KEY, model);
  put(EXPERT_PROVIDER_STORAGE_KEY, provider);
  put(EXPERT_BASE_URL_STORAGE_KEY, baseUrl);
}

// Map a backend provider id to the UI provider bucket used for per-provider keys.
// Vertex (ADC) and LM Studio (local) need no key, so they map to null.
function uiProviderFor(backendProvider: string | null): string | null {
  if (backendProvider === 'anthropic') return 'anthropic';
  if (backendProvider === 'openai') return 'openai';
  if (backendProvider === 'google_genai') return 'google';
  return null;
}

// Build the X-Expert-* headers from the stored expert selection. The expert's API
// key is pulled from the per-provider bucket for its provider (direct providers
// only; Vertex/LM Studio need none). Returns {} when no expert override is set.
export function expertHeaders(): Record<string, string> {
  const h: Record<string, string> = {};
  const model = getStoredExpertModel();
  const provider = getStoredExpertProvider();
  const baseUrl = getStoredExpertBaseUrl();
  if (model) h['X-Expert-Model'] = model;
  if (provider) {
    h['X-Expert-Provider'] = provider;
    const ui = uiProviderFor(provider);
    const key = ui ? getStoredKeyForProvider(ui) : null;
    if (key) h['X-Expert-Api-Key'] = key;
  }
  if (baseUrl) h['X-Expert-Base-Url'] = baseUrl;
  return h;
}

// Per-provider key storage
export function getStoredProviderKeys(): Record<string, string> {
  try {
    return JSON.parse(localStorage.getItem(PROVIDER_KEYS_STORAGE_KEY) || '{}');
  } catch {
    return {};
  }
}

export function getStoredKeyForProvider(provider: string): string | null {
  return getStoredProviderKeys()[provider] ?? null;
}

export function setStoredKeyForProvider(provider: string, key: string): void {
  const keys = getStoredProviderKeys();
  keys[provider] = key;
  localStorage.setItem(PROVIDER_KEYS_STORAGE_KEY, JSON.stringify(keys));
}

// Set API key and model in localStorage (also persists to per-provider bucket)
export function setStoredAuth(apiKey: string, modelName: string): void {
  localStorage.setItem(API_KEY_STORAGE_KEY, apiKey);
  localStorage.setItem(MODEL_NAME_STORAGE_KEY, modelName);
  setStoredKeyForProvider(getProviderForModel(modelName), apiKey);
}

// Remove API key and model from localStorage
export function clearStoredAuth(): void {
  localStorage.removeItem(API_KEY_STORAGE_KEY);
  localStorage.removeItem(MODEL_NAME_STORAGE_KEY);
  localStorage.removeItem(BASE_URL_STORAGE_KEY);
  localStorage.removeItem(PROVIDER_STORAGE_KEY);
}

// Request interceptor - add API key header
apiClient.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    const apiKey = getStoredApiKey();
    const modelName = getStoredModelName();
    const baseUrl = getStoredBaseUrl();
    const provider = getStoredProvider();
    if (apiKey) {
      config.headers['X-API-Key'] = apiKey;
    }
    if (modelName) {
      config.headers['X-Model-Name'] = modelName;
    }
    if (baseUrl) {
      config.headers['X-Base-Url'] = baseUrl;
    }
    if (provider) {
      config.headers['X-Provider'] = provider;
    }
    for (const [k, v] of Object.entries(expertHeaders())) {
      config.headers[k] = v;
    }
    const accessToken = getAccessToken();
    if (accessToken) {
      config.headers['Authorization'] = 'Bearer ' + accessToken;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// ── JWT bearer login / refresh ──────────────────────────────────────────────

interface TokenPair {
  access_token: string;
  refresh_token: string;
}

// Password sign-in. Stores nothing itself — the caller (authStore) persists the
// returned tokens via setStoredTokens.
export async function loginWithPassword(email: string, password: string): Promise<TokenPair> {
  const response = await axios.post<TokenPair>(
    `${API_BASE_URL}/auth/login`,
    { email, password },
    { timeout: 15000 },
  );
  return response.data;
}

// Single-flight refresh: concurrent 401s share one /auth/refresh call. On
// success the new tokens are stored; on failure the tokens are cleared and the
// error re-thrown (callers fall back to the unauthorized path).
let _refreshInFlight: Promise<string> | null = null;
export function refreshAccessToken(): Promise<string> {
  if (_refreshInFlight) return _refreshInFlight;
  _refreshInFlight = (async () => {
    const refreshToken = getRefreshToken();
    if (!refreshToken) throw new Error('No refresh token');
    try {
      const response = await axios.post<TokenPair>(
        `${API_BASE_URL}/auth/refresh`,
        { refresh_token: refreshToken },
        { timeout: 15000 },
      );
      const { access_token, refresh_token } = response.data;
      setStoredTokens(access_token, refresh_token ?? refreshToken);
      return access_token;
    } catch (e) {
      clearStoredTokens();
      throw e;
    } finally {
      _refreshInFlight = null;
    }
  })();
  return _refreshInFlight;
}

// Response interceptor - handle errors
apiClient.interceptors.response.use(
  (response: AxiosResponse) => response,
  async (error: AxiosError) => {
    const original = error.config as
      | (InternalAxiosRequestConfig & { _retried?: boolean })
      | undefined;

    // JWT expiry: try a one-time silent refresh + retry before giving up. Only
    // when we actually have a refresh token and this request wasn't already a
    // retry. Does NOT clear the LLM API key — only the JWT tokens.
    if (error.response?.status === 401 && original && !original._retried && getRefreshToken()) {
      original._retried = true;
      try {
        const newToken = await refreshAccessToken();
        original.headers = original.headers ?? {};
        (original.headers as Record<string, string>)['Authorization'] = 'Bearer ' + newToken;
        return apiClient(original);
      } catch {
        clearStoredTokens();
        window.dispatchEvent(new CustomEvent('auth:unauthorized'));
        return Promise.reject({
          status: 401,
          message: getErrorMessage(error),
          details: error.response?.data,
        } as ApiError);
      }
    }

    const apiError: ApiError = {
      status: error.response?.status || 500,
      message: getErrorMessage(error),
      details: error.response?.data,
    };

    // Handle 401 (no JWT refresh path) - clear API key and redirect to login
    if (error.response?.status === 401) {
      clearStoredAuth();
      // Dispatch event for auth state listeners
      window.dispatchEvent(new CustomEvent('auth:unauthorized'));
    }

    return Promise.reject(apiError);
  }
);

// Extract error message from various error formats
function getErrorMessage(error: AxiosError): string {
  if (error.response?.data) {
    const data = error.response.data as Record<string, unknown>;
    if (typeof data.detail === 'string') {
      return data.detail;
    }
    if (typeof data.message === 'string') {
      return data.message;
    }
    if (typeof data.error === 'string') {
      return data.error;
    }
  }
  return error.message || 'An unexpected error occurred';
}

// Validate API key by making a health check request
export async function validateApiKey(apiKey: string, modelName: string): Promise<boolean> {
  try {
    const response = await axios.get(`${API_BASE_URL}/health`, {
      headers: {
        'X-API-Key': apiKey,
        'X-Model-Name': modelName,
      },
      timeout: 5000,
    });
    return response.status === 200;
  } catch {
    return false;
  }
}

// Check if API is reachable (without auth)
export async function checkApiHealth(): Promise<boolean> {
  try {
    const response = await axios.get(`${API_BASE_URL}/health`, {
      timeout: 5000,
    });
    return response.status === 200;
  } catch {
    return false;
  }
}

// Active LLM configuration reported by the server (non-secret).
export interface ServerModelConfig {
  provider: string;
  model: string;
  uses_vertex: boolean;
  uses_adc: boolean;
  project: string | null;
  location: string | null;
  temperature: number;
  max_tokens: number | null;
  // False when the server authenticates itself (Vertex AI / ADC or an env key),
  // so the UI can skip prompting the user for an API key.
  requires_api_key: boolean;
  // True when the model is served from a local OpenAI-compatible endpoint
  // (e.g. LM Studio); base_url is that endpoint.
  is_local_endpoint?: boolean;
  base_url?: string | null;
  // The strong "expert" tier used by delegate_to_expert. `configured` is false
  // when no expert block is set (delegation reuses the chat model); in that case
  // provider/model mirror the resolved chat tier.
  expert?: { provider: string; model: string; configured: boolean };
}

// Fetch the server's active model configuration, or null if unavailable.
export async function getModelConfig(): Promise<ServerModelConfig | null> {
  try {
    const response = await axios.get<ServerModelConfig>(`${API_BASE_URL}/model-config`, {
      timeout: 5000,
    });
    return response.data;
  } catch {
    return null;
  }
}

// Placeholder "key" used when the server manages credentials itself (Vertex AI /
// ADC or a server-side env key). The backend normalises it away — see
// _SENTINEL_API_KEYS in mmm_framework/agents/llm.py.
export const SERVER_MANAGED_KEY = 'server-managed';

// A selectable Vertex AI model discovered by the server.
export interface VertexModel {
  id: string;
  provider: string;       // vertex_gemini | vertex_anthropic
  family: string;         // gemini | claude
  display_name: string;
  source: string;         // live | catalog | config
  location: string | null;
}

export interface VertexModelsResponse {
  project: string | null;
  location: string | null;
  active_model: string;
  models: VertexModel[];
}

// List Vertex models available to the server's configured project/region.
// Returns an empty list (not an error) when discovery is unavailable, so the
// caller can fall back to free-text entry.
export async function getVertexModels(): Promise<VertexModelsResponse | null> {
  try {
    const response = await axios.get<VertexModelsResponse>(`${API_BASE_URL}/vertex-models`, {
      timeout: 15000,
    });
    return response.data;
  } catch {
    return null;
  }
}

// A model currently loaded in LM Studio (its OpenAI-compatible /v1/models).
export interface LmStudioModel {
  id: string;
  provider: string;       // lmstudio
  family: string;         // lmstudio
  display_name: string;
  source: string;         // live
}

export interface LmStudioModelsResponse {
  base_url: string;
  active_model: string | null;
  models: LmStudioModel[];
}

// List models loaded in LM Studio. Optionally probe a specific base URL (so the
// UI can discover models at a URL the user just typed). Returns null/empty if
// LM Studio isn't running, so the caller falls back to free-text entry.
export async function getLmStudioModels(baseUrl?: string | null): Promise<LmStudioModelsResponse | null> {
  try {
    const response = await axios.get<LmStudioModelsResponse>(`${API_BASE_URL}/lmstudio-models`, {
      params: baseUrl ? { base_url: baseUrl } : undefined,
      timeout: 8000,
    });
    return response.data;
  } catch {
    return null;
  }
}

// Export base URL for use in other modules
export { API_BASE_URL };
