import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { getStoredApiKey, getStoredModelName, setStoredAuth, clearStoredAuth, validateApiKey } from '../api/client';

interface AuthState {
  apiKey: string | null;
  modelName: string | null;
  isAuthenticated: boolean;
  isValidating: boolean;
  validationError: string | null;

  // Actions
  setApiKey: (apiKey: string, modelName: string) => Promise<boolean>;
  clearApiKey: () => void;
  validateStoredKey: () => Promise<boolean>;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      apiKey: getStoredApiKey(),
      modelName: getStoredModelName(),
      isAuthenticated: !!getStoredApiKey(),
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

      clearApiKey: () => {
        clearStoredAuth();
        set({
          apiKey: null,
          modelName: null,
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
      partialize: (state) => ({ apiKey: state.apiKey, modelName: state.modelName }),
    }
  )
);

// Listen for unauthorized events from API client
if (typeof window !== 'undefined') {
  window.addEventListener('auth:unauthorized', () => {
    useAuthStore.getState().clearApiKey();
  });
}
