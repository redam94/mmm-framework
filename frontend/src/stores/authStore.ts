import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { getStoredApiKey, getStoredModelName, setStoredAuth, clearStoredAuth, validateApiKey, getStoredKeyForProvider, setStoredKeyForProvider, getStoredBaseUrl, setStoredBaseUrl, getStoredProvider, setStoredProvider, getStoredExpertModel, getStoredExpertProvider, getStoredExpertBaseUrl, setStoredExpert, getAccessToken, getRefreshToken, setStoredTokens, clearStoredTokens, loginWithPassword } from '../api/client';
import { getProviderForModel } from '../constants/models';

interface AuthState {
  apiKey: string | null;
  modelName: string | null;
  baseUrl: string | null;
  provider: string | null;
  // Strong "expert" tier (delegate_to_expert). When all null the server's
  // configured expert (or the chat model) is used.
  expertModel: string | null;
  expertProvider: string | null;
  expertBaseUrl: string | null;
  // JWT bearer auth (optional, additive). null when JWT auth is unused.
  accessToken: string | null;
  refreshToken: string | null;
  isAuthenticated: boolean;
  isValidating: boolean;
  validationError: string | null;

  // Actions
  setApiKey: (apiKey: string, modelName: string) => Promise<boolean>;
  login: (email: string, password: string) => Promise<boolean>;
  logout: () => void;
  switchModel: (modelName: string) => boolean;
  setBaseUrl: (baseUrl: string | null) => void;
  setProvider: (provider: string | null) => void;
  setExpert: (sel: { model: string | null; provider: string | null; baseUrl?: string | null; apiKey?: string | null }) => void;
  clearApiKey: () => void;
  validateStoredKey: () => Promise<boolean>;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      apiKey: getStoredApiKey(),
      modelName: getStoredModelName(),
      baseUrl: getStoredBaseUrl(),
      provider: getStoredProvider(),
      expertModel: getStoredExpertModel(),
      expertProvider: getStoredExpertProvider(),
      expertBaseUrl: getStoredExpertBaseUrl(),
      accessToken: getAccessToken(),
      refreshToken: getRefreshToken(),
      isAuthenticated: !!getStoredApiKey() || !!getAccessToken(),
      isValidating: false,
      validationError: null,

      setApiKey: async (apiKey: string, modelName: string) => {
        set({ isValidating: true, validationError: null });

        try {
          const isValid = await validateApiKey(apiKey, modelName);

          if (isValid) {
            setStoredAuth(apiKey, modelName);
            set({
              apiKey,
              modelName,
              isAuthenticated: true,
              isValidating: false,
              validationError: null,
            });
            return true;
          } else {
            set({
              isValidating: false,
              validationError: 'Invalid API key',
            });
            return false;
          }
        } catch {
          set({
            isValidating: false,
            validationError: 'Failed to validate API key. Is the server running?',
          });
          return false;
        }
      },

      login: async (email: string, password: string) => {
        set({ isValidating: true, validationError: null });
        try {
          const { access_token, refresh_token } = await loginWithPassword(email, password);
          setStoredTokens(access_token, refresh_token);
          set({
            accessToken: access_token,
            refreshToken: refresh_token,
            isAuthenticated: true,
            isValidating: false,
            validationError: null,
          });
          return true;
        } catch {
          set({
            isValidating: false,
            validationError: 'Invalid email or password.',
          });
          return false;
        }
      },

      logout: () => {
        clearStoredTokens();
        clearStoredAuth();
        set({
          apiKey: null,
          modelName: null,
          baseUrl: null,
          provider: null,
          accessToken: null,
          refreshToken: null,
          isAuthenticated: false,
          validationError: null,
        });
      },

      switchModel: (modelName: string) => {
        const provider = getProviderForModel(modelName);
        const key = getStoredKeyForProvider(provider);
        if (key) {
          setStoredAuth(key, modelName);
          set({ apiKey: key, modelName });
          return true;
        }
        return false;
      },

      setBaseUrl: (baseUrl: string | null) => {
        setStoredBaseUrl(baseUrl);
        set({ baseUrl: baseUrl && baseUrl.trim() ? baseUrl.trim() : null });
      },

      setProvider: (provider: string | null) => {
        setStoredProvider(provider);
        set({ provider: provider && provider.trim() ? provider.trim() : null });
      },

      setExpert: ({ model, provider, baseUrl = null, apiKey = null }) => {
        // Persist the expert's API key into the shared per-provider bucket, so
        // expertHeaders() can find it (direct providers only; Vertex/LM Studio
        // need none). getProviderForModel maps a model id to the UI bucket key.
        if (apiKey && apiKey.trim() && provider) {
          const ui =
            provider === 'anthropic'
              ? 'anthropic'
              : provider === 'openai'
                ? 'openai'
                : provider === 'google_genai'
                  ? 'google'
                  : null;
          if (ui) setStoredKeyForProvider(ui, apiKey.trim());
        }
        setStoredExpert(model, provider, baseUrl);
        set({
          expertModel: model && model.trim() ? model.trim() : null,
          expertProvider: provider && provider.trim() ? provider.trim() : null,
          expertBaseUrl: baseUrl && baseUrl.trim() ? baseUrl.trim() : null,
        });
      },

      clearApiKey: () => {
        clearStoredAuth();
        clearStoredTokens();
        set({
          apiKey: null,
          modelName: null,
          baseUrl: null,
          provider: null,
          accessToken: null,
          refreshToken: null,
          isAuthenticated: false,
          validationError: null,
        });
      },

      validateStoredKey: async () => {
        const apiKey = get().apiKey;
        const modelName = get().modelName;
        if (!apiKey || !modelName) {
          set({ isAuthenticated: false });
          return false;
        }

        try {
          const isValid = await validateApiKey(apiKey, modelName);
          set({ isAuthenticated: isValid });
          if (!isValid) {
            clearStoredAuth();
            set({ apiKey: null, modelName: null });
          }
          return isValid;
        } catch {
          // Keep authenticated if server is unreachable
          return true;
        }
      },
    }),
    {
      name: 'mmm-auth',
      partialize: (state) => ({ apiKey: state.apiKey, modelName: state.modelName, baseUrl: state.baseUrl, provider: state.provider, expertModel: state.expertModel, expertProvider: state.expertProvider, expertBaseUrl: state.expertBaseUrl, accessToken: state.accessToken, refreshToken: state.refreshToken }),
    }
  )
);

// Listen for unauthorized events from API client
if (typeof window !== 'undefined') {
  window.addEventListener('auth:unauthorized', () => {
    useAuthStore.getState().clearApiKey();
  });
}
