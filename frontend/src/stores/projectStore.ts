import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface ProjectState {
  // Current selections
  currentProjectId: string | null;
  selectedDataId: string | null;
  selectedConfigId: string | null;
  selectedModelId: string | null;

  // Recent items for quick access
  recentDataIds: string[];
  recentConfigIds: string[];
  recentModelIds: string[];

  // Actions
  setProject: (id: string | null) => void;
  setSelectedData: (id: string | null) => void;
  setSelectedConfig: (id: string | null) => void;
  setSelectedModel: (id: string | null) => void;
  clearSelections: () => void;

  // Recent items management
  addRecentData: (id: string) => void;
  addRecentConfig: (id: string) => void;
  addRecentModel: (id: string) => void;
}

const MAX_RECENT_ITEMS = 10;

// Add to recent list, keeping most recent at front
function addToRecent(list: string[], id: string): string[] {
  const filtered = list.filter((item) => item !== id);
  return [id, ...filtered].slice(0, MAX_RECENT_ITEMS);
}

export const useProjectStore = create<ProjectState>()(
  persist(
    (set) => ({
      currentProjectId: null,
      selectedDataId: null,
      selectedConfigId: null,
      selectedModelId: null,
      recentDataIds: [],
      recentConfigIds: [],
      recentModelIds: [],

      setProject: (id) =>
        set({
          currentProjectId: id,
          // Clear selections when changing projects
          selectedDataId: null,
          selectedConfigId: null,
          selectedModelId: null,
        }),

      setSelectedData: (id) =>
        set((state) => ({
          selectedDataId: id,
          recentDataIds: id ? addToRecent(state.recentDataIds, id) : state.recentDataIds,
        })),

      setSelectedConfig: (id) =>
        set((state) => ({
          selectedConfigId: id,
          recentConfigIds: id ? addToRecent(state.recentConfigIds, id) : state.recentConfigIds,
        })),

      setSelectedModel: (id) =>
        set((state) => ({
          selectedModelId: id,
          recentModelIds: id ? addToRecent(state.recentModelIds, id) : state.recentModelIds,
        })),

      clearSelections: () =>
        set({
          selectedDataId: null,
          selectedConfigId: null,
          selectedModelId: null,
        }),

      addRecentData: (id) =>
        set((state) => ({
          recentDataIds: addToRecent(state.recentDataIds, id),
        })),

      addRecentConfig: (id) =>
        set((state) => ({
          recentConfigIds: addToRecent(state.recentConfigIds, id),
        })),

      addRecentModel: (id) =>
        set((state) => ({
          recentModelIds: addToRecent(state.recentModelIds, id),
        })),
    }),
    {
      name: 'mmm-project',
    }
  )
);
