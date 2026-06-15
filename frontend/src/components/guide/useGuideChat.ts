import { useCallback, useEffect, useRef, useState } from 'react';
import { API_BASE_URL, expertHeaders } from '../../api/client';
import { useAuthStore } from '../../stores/authStore';

export interface GuideMessage {
  id: string;
  role: 'user' | 'assistant' | 'error';
  content: string;
}

/** One streamed SSE frame (or one /state message) from the agent backend. */
interface StreamEvent {
  type?: string;
  content?: unknown;
  tool_calls?: { name?: string }[];
}

/** The backend sends content as a string or a list of text blocks. */
function normalizeContent(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    return content
      .map((b) => {
        if (typeof b === 'string') return b;
        if (b && typeof b === 'object' && 'text' in b) {
          const t = (b as { text?: unknown }).text;
          return typeof t === 'string' ? t : '';
        }
        return '';
      })
      .filter(Boolean)
      .join('\n');
  }
  if (content == null) return '';
  if (typeof content === 'object') return JSON.stringify(content);
  return String(content);
}

/**
 * Per-project guide chat. Lazily creates (idempotently) the project's guide
 * session on first open, hydrates its history, and streams /chat turns.
 */
export function useGuideChat(projectId: string | null) {
  const { apiKey, modelName, baseUrl, provider, expertModel, expertProvider, expertBaseUrl } =
    useAuthStore();
  const [messages, setMessages] = useState<GuideMessage[]>([]);
  const [threadId, setThreadId] = useState<string | null>(null);
  const [streaming, setStreaming] = useState(false);
  const [activeTool, setActiveTool] = useState<string | null>(null);
  // Which project the current threadId/messages belong to.
  const initializedForRef = useRef<string | null>(null);
  const initializingRef = useRef(false);

  const headers = useCallback((): Record<string, string> => {
    const h: Record<string, string> = {
      'X-API-Key': apiKey || '',
      'X-Model-Name': modelName || '',
    };
    if (baseUrl) h['X-Base-Url'] = baseUrl;
    if (provider) h['X-Provider'] = provider;
    Object.assign(h, expertHeaders());
    return h;
  }, [apiKey, modelName, baseUrl, provider, expertModel, expertProvider, expertBaseUrl]);

  // Re-hydrate when the project changes.
  useEffect(() => {
    if (initializedForRef.current !== projectId) {
      initializedForRef.current = null;
      setThreadId(null);
      setMessages([]);
    }
  }, [projectId]);

  /** Ensure the per-project guide session exists and hydrate its history. */
  const init = useCallback(async () => {
    if (!projectId || initializedForRef.current === projectId || initializingRef.current) return;
    initializingRef.current = true;
    try {
      const guideRes = await fetch(`${API_BASE_URL}/projects/${projectId}/guide`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...headers() },
      });
      if (!guideRes.ok) throw new Error(`Guide session failed (${guideRes.status})`);
      const { thread_id: tid } = (await guideRes.json()) as { thread_id: string };
      setThreadId(tid);

      const stateRes = await fetch(`${API_BASE_URL}/state/${tid}`, { headers: headers() });
      if (stateRes.ok) {
        const state = (await stateRes.json()) as { messages?: StreamEvent[] };
        const hydrated: GuideMessage[] = [];
        (state.messages || []).forEach((m, i) => {
          const content = normalizeContent(m.content);
          if (m.type === 'human' && content) {
            hydrated.push({ id: `guide-${tid}-${i}`, role: 'user', content });
          } else if (m.type === 'ai' && content) {
            hydrated.push({ id: `guide-${tid}-${i}`, role: 'assistant', content });
          }
          // type === 'tool' and empty ai messages are not displayed.
        });
        setMessages(hydrated);
      }
      initializedForRef.current = projectId;
    } catch (e) {
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: 'error',
          content: e instanceof Error ? e.message : 'Could not start the guide session.',
        },
      ]);
    } finally {
      initializingRef.current = false;
    }
  }, [projectId, headers]);

  const send = useCallback(
    async (text: string, pageContext: string) => {
      const trimmed = text.trim();
      if (!trimmed || !threadId || streaming) return;

      setStreaming(true);
      setActiveTool(null);
      setMessages((prev) => [
        ...prev,
        { id: crypto.randomUUID(), role: 'user', content: trimmed },
      ]);
      const aiId = crypto.randomUUID();
      setMessages((prev) => [...prev, { id: aiId, role: 'assistant', content: '' }]);

      const setAiContent = (content: string) =>
        setMessages((prev) =>
          prev.map((m) => (m.id === aiId ? { ...m, content } : m)),
        );

      try {
        const response = await fetch(`${API_BASE_URL}/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', ...headers() },
          body: JSON.stringify({ message: trimmed, thread_id: threadId, page_context: pageContext }),
        });
        if (!response.ok || !response.body) {
          throw new Error(`The guide is unavailable right now (${response.status}).`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let aiContent = '';

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';
          for (const line of lines) {
            if (!line.startsWith('data: ') || line === 'data: [DONE]') continue;
            let data: StreamEvent;
            try {
              data = JSON.parse(line.substring(6)) as StreamEvent;
            } catch {
              continue;
            }
            if (data.type === 'dashboard_update') continue;
            if (data.type === 'error') {
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === aiId
                    ? {
                        ...m,
                        role: 'error' as const,
                        content: normalizeContent(data.content) || 'Something went wrong.',
                      }
                    : m,
                ),
              );
              continue;
            }
            if (data.type === 'ai') {
              if (Array.isArray(data.tool_calls) && data.tool_calls.length > 0) {
                const last = data.tool_calls[data.tool_calls.length - 1];
                setActiveTool(last?.name || null);
              }
              const chunk = normalizeContent(data.content);
              if (chunk) {
                aiContent += chunk + '\n';
                setAiContent(aiContent);
              }
            }
            // type === 'tool': keep showing the activity line from the
            // preceding ai message's tool_calls (activeTool).
          }
        }
      } catch (e) {
        const msg = e instanceof Error ? e.message : 'Connection failed.';
        setMessages((prev) => [
          ...prev.filter((m) => !(m.id === aiId && !m.content)),
          { id: crypto.randomUUID(), role: 'error', content: msg },
        ]);
      } finally {
        // Drop the placeholder bubble if the turn produced no text.
        setMessages((prev) => prev.filter((m) => !(m.id === aiId && !m.content.trim())));
        setActiveTool(null);
        setStreaming(false);
      }
    },
    [threadId, streaming, headers],
  );

  return { messages, threadId, streaming, activeTool, init, send };
}
