import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  atelierNotebookService,
  type NotebookCell,
  type NotebookDataset,
  type NotebookDoc,
} from "../services/atelierNotebookService";

export const notebookKeys = {
  all: ["atelierNotebook"] as const,
  doc: (name: string, version?: number | null) =>
    [...notebookKeys.all, name, version ?? "draft"] as const,
};

/** Load the persisted notebook doc for a model (or a seeded starter). */
export function useNotebookDoc(name: string | null, version?: number | null) {
  return useQuery({
    queryKey: notebookKeys.doc(name ?? "", version),
    queryFn: () => atelierNotebookService.getNotebook(name!, version),
    enabled: !!name,
    // The doc is the source of truth on mount; the component owns it after that,
    // so don't auto-refetch and clobber unsaved edits.
    staleTime: Infinity,
    refetchOnWindowFocus: false,
  });
}

/** Upsert the notebook doc (debounced autosave from the component). */
export function useSaveNotebook() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (req: {
      name: string;
      version?: number | null;
      cells: NotebookCell[];
      dataset?: NotebookDataset | null;
    }) => atelierNotebookService.saveNotebook(req),
    // Update the cached doc OPTIMISTICALLY (synchronously, before the network
    // round-trip). This is what makes a flush-on-unmount race-safe: when the
    // notebook unmounts on a tab switch it fires this mutation, and a remount
    // immediately reads the fresh cells from the cache rather than the stale doc.
    onMutate: (req) => {
      qc.setQueryData<NotebookDoc>(
        notebookKeys.doc(req.name, req.version),
        (prev) => ({
          ...(prev ?? { cells: [] }),
          cells: req.cells,
          dataset: req.dataset ?? null,
          name: req.name,
          version: req.version ?? null,
          seeded: false,
        }),
      );
    },
  });
}
