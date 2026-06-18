import { apiClient } from '../client';
import type { ProjectMember, ProjectMemberInput } from './teamService';

/** Onboarding profile stored in projects.meta_json (all fields optional). */
export interface ProjectMeta {
  client_name?: string;
  industry?: string;
  website?: string;
  markets?: string;
  audience?: string;
  goals?: string;
  kpis?: string;
  channels?: string;
  constraints?: string;
  notes?: string;
  onboarded?: boolean;
  [key: string]: unknown;
}

export interface ProjectResponse {
  project_id: string;
  name: string;
  description: string | null;
  meta?: ProjectMeta;
  created_at: string;
  updated_at: string;
  data_count: number;
  config_count: number;
  model_count: number;
  session_count: number;
}

export interface ProjectListResponse {
  projects: ProjectResponse[];
  total: number;
}

export interface ProjectCreateRequest {
  name: string;
  description?: string;
}

export interface ProjectUpdateRequest {
  name?: string;
  description?: string;
}

export interface ProjectOnboardingRequest {
  client_name?: string;
  industry?: string;
  website?: string;
  markets?: string;
  audience?: string;
  goals?: string;
  kpis?: string;
  channels?: string;
  constraints?: string;
  notes?: string;
  members?: ProjectMemberInput[];
}

export interface ProjectOnboardingResponse {
  project: ProjectResponse;
  members: ProjectMember[];
  /** 'ready' when the project brief was ingested into the knowledge base. */
  brief_status: string;
}

export interface OnboardingStep {
  key: string;
  title: string;
  done: boolean;
  hint: string;
}

/** Path-to-first-model checklist computed from real project state. */
export interface OnboardingStatus {
  project_id: string;
  project_name: string;
  steps: OnboardingStep[];
  completed: number;
  total: number;
  percent: number;
  complete: boolean;
  next_step: string | null;
  next_hint: string;
  counts: { data_files: number; model_runs: number; experiments: number };
}

export const projectService = {
  async listProjects(): Promise<ProjectListResponse> {
    const { data } = await apiClient.get<ProjectListResponse>('/projects');
    return data;
  },

  async getProject(projectId: string): Promise<ProjectResponse> {
    const { data } = await apiClient.get<ProjectResponse>(`/projects/${projectId}`);
    return data;
  },

  async createProject(request: ProjectCreateRequest): Promise<ProjectResponse> {
    const { data } = await apiClient.post<ProjectResponse>('/projects', request);
    return data;
  },

  async updateProject(projectId: string, request: ProjectUpdateRequest): Promise<ProjectResponse> {
    const { data } = await apiClient.patch<ProjectResponse>(`/projects/${projectId}`, request);
    return data;
  },

  async deleteProject(projectId: string): Promise<void> {
    await apiClient.delete(`/projects/${projectId}`);
  },

  async onboardProject(
    projectId: string,
    request: ProjectOnboardingRequest,
  ): Promise<ProjectOnboardingResponse> {
    const { data } = await apiClient.post<ProjectOnboardingResponse>(
      `/projects/${projectId}/onboarding`,
      request,
    );
    return data;
  },

  async getOnboardingStatus(projectId: string): Promise<OnboardingStatus> {
    const { data } = await apiClient.get<OnboardingStatus>(
      `/projects/${projectId}/onboarding-status`,
    );
    return data;
  },
};
