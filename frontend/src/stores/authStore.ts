import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { getStoredApiKey, setStoredApiKey, clearStoredApiKey, validateApiKey } from '../api/client';

interface AuthState {
  apiKey: string | null;
  isAuthenticated: boolean;
  isValidating: boolean;
  validationError: string | null;

  // Actions
  setApiKey: (apiKey: string) => Promise<boolean>;
  clearApiKey: () => void;
  validateStoredKey: () => Promise<boolean>;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      apiKey: getStoredApiKey(),
      isAuthenticated: !!getStoredApiKey(),
      isValidating: false,
      validationError: null,

      setApiKey: async (apiKey: string) => {
        set({ isValidating: true, validationError: null });

        try {
          const isValid = await validateApiKey(apiKey);

          if (isValid) {
            setStoredApiKey(apiKey);
            set({
              apiKey,
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
        clearStoredApiKey();
        set({
          apiKey: null,
          isAuthenticated: false,
          validationError: null,
        });
      },

      validateStoredKey: async () => {
        const apiKey = get().apiKey;
        if (!apiKey) {
          set({ isAuthenticated: false });
          return false;
        }

        try {
          const isValid = await validateApiKey(apiKey);
          set({ isAuthenticated: isValid });
          if (!isValid) {
            clearStoredApiKey();
            set({ apiKey: null });
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
      partialize: (state) => ({ apiKey: state.apiKey }),
    }
  )
);

// Listen for unauthorized events from API client
if (typeof window !== 'undefined') {
  window.addEventListener('auth:unauthorized', () => {
    useAuthStore.getState().clearApiKey();
  });
}
