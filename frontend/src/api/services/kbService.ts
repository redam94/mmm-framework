import { apiClient } from '../client';

/** A document in a project's knowledge base (kb_documents row). */
export interface KbDocument {
  id: string;
  project_id: string;
  name: string;
  kind: string;
  size_bytes: number | null;
  n_chunks: number;
  status: 'pending' | 'ready' | 'error' | string;
  error: string | null;
  /** e.g. { source: 'onboarding', template: true, content_type: ... } */
  meta: Record<string, unknown>;
  created_at: number;
}

export interface KbDocumentListResponse {
  documents: KbDocument[];
  total: number;
}

export interface KbSearchResult {
  document: string;
  chunk_index: number;
  text: string;
  score: number;
}

export const kbService = {
  async listDocuments(projectId: string): Promise<KbDocumentListResponse> {
    const { data } = await apiClient.get<KbDocumentListResponse>(`/projects/${projectId}/kb`);
    return data;
  },

  /** Upload + ingest a file. `template` tags it for the agent's list_templates. */
  async uploadDocument(projectId: string, file: File, template = false): Promise<KbDocument> {
    const fd = new FormData();
    fd.append('file', file);
    fd.append('template', String(template));
    const { data } = await apiClient.post<KbDocument>(`/projects/${projectId}/kb`, fd, {
      // Let the browser set the multipart boundary; embedding can take a while.
      headers: { 'Content-Type': undefined as unknown as string },
      timeout: 120000,
    });
    return data;
  },

  async deleteDocument(documentId: string): Promise<void> {
    await apiClient.delete(`/kb/${documentId}`);
  },

  async search(projectId: string, query: string, k = 6): Promise<KbSearchResult[]> {
    const { data } = await apiClient.get<{ results: KbSearchResult[] }>(
      `/projects/${projectId}/kb/search`,
      { params: { q: query, k } },
    );
    return data.results ?? [];
  },
};
