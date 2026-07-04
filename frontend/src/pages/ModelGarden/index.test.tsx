import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import type { GardenModel } from '../../api/services/modelGardenService';

// Monaco cannot load in jsdom — stub the editor surface.
vi.mock('@monaco-editor/react', () => ({
  default: () => <div data-testid="editor" />,
}));
// The copilot rail and notebook fetch their own state — stub them; this test
// pins the Atelier page shell (list → detail → tabs), not their internals.
vi.mock('../../components/modelGarden/CopilotPanel', () => ({
  CopilotPanel: () => <div data-testid="copilot" />,
}));
vi.mock('../../components/modelGarden/AtelierNotebook', () => ({
  AtelierNotebook: () => <div data-testid="notebook" />,
}));

const models = { data: { models: [] as GardenModel[] }, isLoading: false };
const mutation = () => ({ mutate: vi.fn(), mutateAsync: vi.fn(), isPending: false });
vi.mock('../../api/hooks', () => ({
  copilotChatKeys: { all: ['copilot'] },
  useGardenModels: () => models,
  useGardenVersions: () => ({ data: null }),
  useGardenModel: () => ({ data: null }),
  useGardenSource: () => ({ data: null }),
  useGardenTest: () => ({ start: mutation(), job: { data: null }, reset: vi.fn() }),
  useDeleteGardenModel: mutation,
  usePromoteGardenModel: mutation,
  useRegisterGardenModel: mutation,
  useUpdateGardenDocs: mutation,
}));
vi.mock('@tanstack/react-query', async (orig) => ({
  ...(await orig<Record<string, unknown>>()),
  useQueryClient: () => ({ invalidateQueries: vi.fn() }),
}));

const { ModelGardenPage } = await import('./index');

describe('ModelGardenPage (Atelier)', () => {
  beforeEach(() => {
    models.data = { models: [] };
    models.isLoading = false;
  });

  it('renders the empty garden without crashing', () => {
    render(<ModelGardenPage />);
    // page frame is up (no models yet) — nothing exploded on mount
    expect(document.body.textContent).toMatch(/garden|atelier|model/i);
  });

  it('lists registered models', () => {
    models.data = {
      models: [
        {
          name: 'awareness_mmm',
          latest_version: 2,
          status: 'published' as const,
          model_kind: 'mmm',
          updated_at: 1751000000,
        } as unknown as GardenModel,
      ],
    };
    render(<ModelGardenPage />);
    expect(screen.getByText('awareness_mmm')).toBeInTheDocument();
  });
});
