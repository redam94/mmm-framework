import axios from 'axios';
import type { AxiosError, AxiosInstance, AxiosResponse, InternalAxiosRequestConfig } from 'axios';
import type { ApiError } from './types';
import { getProviderForModel } from '../constants/models';

// API Base URL - defaults to localhost for development
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

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

// Get API key and model from localStorage
export function getStoredApiKey(): string | null {
  return localStorage.getItem(API_KEY_STORAGE_KEY);
}
export function getStoredModelName(): string | null {
  return localStorage.getItem(MODEL_NAME_STORAGE_KEY);
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
}

// Request interceptor - add API key header
apiClient.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    const apiKey = getStoredApiKey();
    const modelName = getStoredModelName();
    if (apiKey) {
      config.headers['X-API-Key'] = apiKey;
    }
    if (modelName) {
      config.headers['X-Model-Name'] = modelName;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor - handle errors
apiClient.interceptors.response.use(
  (response: AxiosResponse) => response,
  (error: AxiosError) => {
    const apiError: ApiError = {
      status: error.response?.status || 500,
      message: getErrorMessage(error),
      details: error.response?.data,
    };

    // Handle 401 - clear API key and redirect to login
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

// Export base URL for use in other modules
export { API_BASE_URL };
