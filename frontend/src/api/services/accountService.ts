import { apiClient } from '../client';

// Org RBAC roles, least -> most privileged (mirrors auth.models.Role).
export type OrgRole = 'viewer' | 'analyst' | 'admin' | 'owner';

export interface Me {
  user_id: string;
  email: string;
  name?: string | null;
  org_id: string;
  org_role: OrgRole;
}

export interface UsageBucket {
  used: number;
  limit: number | null;
  remaining: number | null;
  over: boolean;
}

export interface OrgUsage {
  plan: string;
  plan_name: string;
  features: string[];
  seats: UsageBucket;
  projects: UsageBucket;
  fits_this_month: UsageBucket;
}

export interface TokenResponse {
  access_token: string;
  refresh_token: string;
  token_type?: string;
  expires_in?: number;
}

// ── Integrations catalog (GET /integrations/catalog) ────────────────────────────

export interface CatalogField {
  name: string;
  label: string;
  required: boolean;
  placeholder?: string;
}

export interface DataSourceCatalogEntry {
  kind: string;
  label: string;
  description: string;
  auth: string;
  installed: boolean;
  install_extra: string;
  fields: CatalogField[];
}

export interface AdPlatformCatalogEntry {
  platform: string;
  label: string;
  status: string;
  ease: 'easy' | 'moderate' | 'hard';
  official_sdk: string | null;
  auth: string;
  recommended_path: string;
  metrics: string[];
  notes: string;
  sdk_installed: boolean;
}

export interface IntegrationsCatalog {
  data_sources: DataSourceCatalogEntry[];
  ad_platforms: AdPlatformCatalogEntry[];
}

export const accountService = {
  async getMe(): Promise<Me> {
    const { data } = await apiClient.get<Me>('/auth/me');
    return data;
  },

  async getUsage(): Promise<OrgUsage> {
    const { data } = await apiClient.get<OrgUsage>('/auth/usage');
    return data;
  },

  async changePassword(currentPassword: string, newPassword: string): Promise<TokenResponse> {
    const { data } = await apiClient.post<TokenResponse>('/auth/change-password', {
      current_password: currentPassword,
      new_password: newPassword,
    });
    return data;
  },

  async getIntegrationsCatalog(): Promise<IntegrationsCatalog> {
    const { data } = await apiClient.get<IntegrationsCatalog>('/integrations/catalog');
    return data;
  },
};
