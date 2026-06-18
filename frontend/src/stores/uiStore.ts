import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface UiState {
  /** App sidebar collapsed to an icon rail (shared by Sidebar + AppShell
   * so the content padding tracks the rail width). */
  sidebarCollapsed: boolean;
  toggleSidebar: () => void;
}

export const useUiStore = create<UiState>()(
  persist(
    (set) => ({
      sidebarCollapsed: false,
      toggleSidebar: () => set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),
    }),
    { name: 'mmm-ui' },
  ),
);
