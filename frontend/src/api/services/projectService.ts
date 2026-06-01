import { apiClient } from '../client';

export interface ProjectResponse {
  project_id: string;
  name: string;
  description: string | null;
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
};
