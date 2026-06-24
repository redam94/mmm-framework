import { useCallback, useEffect, useRef, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  copilotService,
  type CopilotChatDoc,
  type CopilotChatMessage,
  type CopilotSurface,
} from '../services/copilotService';

export const copilotChatKeys = {
  all: ['copilotChat'] as const,
  doc: (name: string, version: number | null | undefined, surface: CopilotSurface) =>
    [...copilotChatKeys.all, surface, name, version ?? 'draft'] as const,
};

/** Load the persisted copilot chat for a (model, version, surface). */
export function useCopilotChat(
  name: string | null,
  version: number | null | undefined,
  surface: CopilotSurface,
  enabled = true,
) {
  return useQuery({
    queryKey: copilotChatKeys.doc(name ?? '', version, surface),
    queryFn: () => copilotService.getChat(name!, version, surface),
    enabled: enabled && !!name,
    // The doc is the source of truth on mount; the component owns it after that,
    // so don't auto-refetch and clobber an in-flight conversation.
    staleTime: Infinity,
    refetchOnWindowFocus: false,
  });
}

/** Upsert the copilot chat (debounced autosave from the panel). */
export function useSaveCopilotChat() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (req: {
      name: string;
      version?: number | null;
      surface: CopilotSurface;
      messages: CopilotChatMessage[];
    }) => copilotService.saveChat(req),
    // Update the cached doc OPTIMISTICALLY so a flush-on-unmount is race-safe:
    // a remount reads the fresh messages from cache rather than the stale doc.
    onMutate: (req) => {
      qc.setQueryData<CopilotChatDoc>(
        copilotChatKeys.doc(req.name, req.version, req.surface),
        (prev) => ({
          ...(prev ?? {}),
          messages: req.messages,
          name: req.name,
          version: req.version ?? null,
          surface: req.surface,
        }),
      );
    },
  });
}

/** Persisted message shape the panels use (a superset of CopilotChatMessage —
 * `error` bubbles live in local state but are never persisted). */
export interface PersistedMsg {
  id: string;
  role: 'user' | 'assistant' | 'error';
  content: string;
  targetCellId?: string | null;
}

/** Strip the transient turns (errors + the empty assistant placeholder shown
 * while a response streams) before persisting. */
function persistable(msgs: PersistedMsg[]): CopilotChatMessage[] {
  return msgs
    .filter((m) => m.role !== 'error' && m.content.trim() !== '')
    .map((m) => ({
      id: m.id,
      role: m.role,
      content: m.content,
      ...(m.targetCellId ? { targetCellId: m.targetCellId } : {}),
    }));
}

/**
 * Stateful copilot chat scoped to one (model, version, surface): loads the
 * persisted conversation, seeds local state once per key, debounce-autosaves on
 * change, flushes on unmount, and clears server-side. The whole point is that
 * each model/version remembers ITS OWN chat — switching models swaps the
 * conversation, and Clear wipes only the current one.
 */
export function useCopilotChatState(opts: {
  name: string | null;
  version?: number | null;
  surface: CopilotSurface;
  enabled?: boolean;
}) {
  const { name, version = null, surface, enabled = true } = opts;
  const query = useCopilotChat(name, version, surface, enabled);
  const save = useSaveCopilotChat();

  const key = `${surface}:${name ?? ''}:${version ?? 'draft'}`;
  const [messages, setMessages] = useState<PersistedMsg[]>([]);
  const [seededKey, setSeededKey] = useState<string | null>(null);

  // Seed local state when the key's doc arrives — the documented "adjust state
  // while rendering" pattern (React re-renders immediately, no commit in
  // between). No effect ⇒ no race between seeding and autosave: until a key is
  // seeded `ready` is false, so a model switch can never flush the previous
  // model's messages into the new key. The `seededKey !== key` guard makes this
  // run exactly once per key (no render loop), and a later refetch of the SAME
  // key can't clobber an in-flight conversation.
  if (name && query.data && seededKey !== key) {
    setSeededKey(key);
    setMessages((query.data.messages ?? []) as PersistedMsg[]);
  }
  const ready = seededKey === key;

  // Latest snapshot + flush closure, kept current in an effect (never written
  // during render) so the unmount flush + debounced timer see fresh values.
  const dirtyRef = useRef(false);
  const flushRef = useRef<() => void>(() => {});
  const baseline = query.data?.messages;
  useEffect(() => {
    flushRef.current = () => {
      if (!dirtyRef.current) return;
      dirtyRef.current = false;
      if (!name || !ready) return;
      const next = persistable(messages);
      // Skip a no-op write (e.g. right after seeding, or an unchanged chat):
      // query.data is the last-persisted baseline (the optimistic save keeps it
      // current), so an equal payload means there's nothing new to store.
      if (
        JSON.stringify(next) ===
        JSON.stringify(persistable((baseline ?? []) as PersistedMsg[]))
      )
        return;
      save.mutate({ name, version: version ?? null, surface, messages: next });
    };
  });

  // Debounced autosave once the current key is seeded (ready).
  useEffect(() => {
    if (!ready || !name) return;
    dirtyRef.current = true;
    const t = setTimeout(() => flushRef.current(), 800);
    return () => clearTimeout(t);
  }, [messages, ready, name, version, surface]);

  // Flush any pending save on unmount (panel closed / model switched).
  useEffect(
    () => () => {
      if (dirtyRef.current) flushRef.current();
    },
    [],
  );

  const clear = useCallback(() => {
    setMessages([]);
    dirtyRef.current = false;
    if (name) {
      save.mutate({ name, version: version ?? null, surface, messages: [] });
    }
  }, [name, version, surface, save]);

  return {
    messages,
    setMessages,
    clear,
    loading: enabled && !!name && query.isLoading,
  };
}
