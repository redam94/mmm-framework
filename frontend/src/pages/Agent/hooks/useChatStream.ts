import { useCallback, useRef, useState } from 'react';
import { API_BASE, authHeaders } from '../constants';
import { normalizeContent } from '../utils/text';
import { extractPythonOutput, pythonOutputsFromArtifacts } from '../utils/python';
import { mergeDashboardData } from '../utils/dashboard';
import type { Artifact, ChatMessage, DashboardData, PythonOutput, ToolCall } from '../types';

/** Raw message shape from the persisted thread state (`GET /state/{tid}`). */
interface RawStateMessage {
  type?: string;
  content?: unknown;
  tool_calls?: RawToolCall[];
  tool_call_id?: string;
}

/** Raw tool-call shape from a persisted AI message. */
interface RawToolCall {
  id?: string;
  name?: string;
  args?: Record<string, unknown>;
}

export function useChatStream({ threadId, apiKey, modelName, onTurnSettled, onArtifactsLoaded }: {
  threadId: string | null;
  apiKey: string | null;
  modelName: string | null;
  /** Called after each chat turn settles (success, error, or abort) so the
   *  caller can refresh artifacts / causal panels / workspace files. */
  onTurnSettled?: () => void | Promise<void>;
  /** Called by loadThreadState with the artifacts fetched alongside the thread
   *  state (artifact state lives in the page, not this hook). MUST be stable
   *  (useCallback) — it is a dependency of loadThreadState. */
  onArtifactsLoaded?: (arts: Artifact[]) => void;
}) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [dashboardData, setDashboardData] = useState<DashboardData>({});
  const [pythonOutputs, setPythonOutputs] = useState<PythonOutput[]>([]);
  const [loading, setLoading] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const loadThreadState = useCallback(async (tid: string) => {
    try {
      const [stateRes, artRes] = await Promise.all([
        fetch(`${API_BASE}/state/${tid}`, { headers: authHeaders(apiKey, modelName) }).then(r => r.json()),
        fetch(`${API_BASE}/artifacts/${tid}`, { headers: authHeaders(apiKey, modelName) }).then(r => r.json()),
      ]);

      // Build messages, pairing AI tool_calls with their ToolMessage results.
      const parsed: ChatMessage[] = [];
      // Maps tool_call_id → index in parsed so tool results can be stitched back in.
      const tcIdToMsgIdx: Record<string, number> = {};

      (stateRes.messages || []).forEach((m: RawStateMessage, i: number) => {
        if (m.type === 'human') {
          const content = normalizeContent(m.content);
          if (content) parsed.push({ id: `loaded-${tid}-${i}`, type: 'human', content });
        } else if (m.type === 'ai') {
          const content = normalizeContent(m.content);
          const toolCalls: ToolCall[] = (m.tool_calls || []).map((tc: RawToolCall) => ({
            id: tc.id ?? `tc-${i}-${tc.name}`,
            name: tc.name ?? 'unknown',
            args: tc.args ?? {},
            status: 'done' as const,
            result: undefined,
          }));
          if (!content && !toolCalls.length) return;
          const msgIdx = parsed.length;
          parsed.push({ id: `loaded-${tid}-${i}`, type: 'ai', content, toolCalls });
          toolCalls.forEach(tc => { tcIdToMsgIdx[tc.id] = msgIdx; });
        } else if (m.type === 'tool') {
          // Stitch the result back into the matching tool call on the AI message.
          const tcId = m.tool_call_id;
          const msgIdx = tcId != null ? tcIdToMsgIdx[tcId] : undefined;
          if (msgIdx != null) {
            const msg = parsed[msgIdx];
            const tc = msg?.toolCalls?.find(t => t.id === tcId);
            if (tc) tc.result = normalizeContent(m.content);
          }
        }
      });

      setMessages(parsed);
      setDashboardData(stateRes.dashboard_data || {});
      const arts: Artifact[] = Array.isArray(artRes) ? artRes : [];
      onArtifactsLoaded?.(arts);

      // Rehydrate persisted python outputs from text_output artifacts, pairing
      // each with its code_snippet by call_id (code may be unavailable for some).
      setPythonOutputs(pythonOutputsFromArtifacts(arts));
    } catch (e) { console.error('Failed to load thread state', e); }
  }, [apiKey, modelName, onArtifactsLoaded]);

  // ── Chat actions ───────────────────────────────────────────────────────────
  const clear = async () => {
    if (!threadId) return;
    setMessages([]);
    setDashboardData({});
    setPythonOutputs([]);
    try {
      await fetch(`${API_BASE}/state/${threadId}`, {
        method: 'DELETE', headers: authHeaders(apiKey, modelName),
      });
    } catch { /* ignore */ }
  };

  const stop = () => {
    abortRef.current?.abort();
    abortRef.current = null;
  };

  const send = async (textToSend: string) => {
    if (!threadId) return;
    if (!textToSend.trim()) return;

    const humanId = crypto.randomUUID();
    setMessages(prev => [...prev, { id: humanId, type: 'human', content: textToSend }]);
    setLoading(true);

    const tempAiId = crypto.randomUUID();
    setMessages(prev => [...prev, { id: tempAiId, type: 'ai', content: '', toolCalls: [] }]);
    const toolCallMap: Record<string, ToolCall> = {};

    const updateMsg = (updater: (m: ChatMessage) => ChatMessage) =>
      setMessages(prev => prev.map(m => m.id === tempAiId ? updater(m) : m));

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const response = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...authHeaders(apiKey, modelName) },
        body: JSON.stringify({ message: textToSend, thread_id: threadId }),
        signal: controller.signal,
      });
      if (!response.body) return;

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let aiContent = '';
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          if (line === 'data: [DONE]') {
            // Mark any tools still spinning as done (stream ended without their result)
            const runningKeys = Object.keys(toolCallMap).filter(k => toolCallMap[k].status === 'running');
            if (runningKeys.length > 0) {
              runningKeys.forEach(k => { toolCallMap[k] = { ...toolCallMap[k], status: 'done', result: toolCallMap[k].result ?? '' }; });
              updateMsg(m => ({ ...m, toolCalls: Object.values(toolCallMap) }));
            }
            continue;
          }
          try {
            const data = JSON.parse(line.substring(6));
            if (data.dashboard_data && Object.keys(data.dashboard_data).length > 0)
              // Ref-list-aware merge — a plain spread would let a payload
              // carrying a subset of plots/tables wipe the accumulated ones
              // from the Results tab until reload.
              setDashboardData((prev: DashboardData) => mergeDashboardData(prev, data.dashboard_data));
            if (data.type === 'dashboard_update') continue;
            if (data.type === 'error') {
              // Replace the pending AI bubble with an error notice
              setMessages(prev => prev.map(m =>
                m.id === tempAiId
                  ? { ...m, type: 'error' as const, content: data.content || 'An unknown error occurred.' }
                  : m
              ));
              continue;
            }
            if (data.type === 'ai') {
              if (Array.isArray(data.tool_calls)) {
                for (const tc of data.tool_calls) {
                  const id = tc.id || tc.name + '_' + Date.now();
                  toolCallMap[id] = { id, name: tc.name || 'unknown', args: tc.args || {}, status: 'running' };
                }
                updateMsg(m => ({ ...m, toolCalls: Object.values(toolCallMap) }));
              }
              const cs = normalizeContent(data.content);
              if (cs) { aiContent += cs + '\n'; updateMsg(m => ({ ...m, content: aiContent })); }
            }
            if (data.type === 'tool') {
              const rs = normalizeContent(data.content);
              const key = data.tool_call_id && toolCallMap[data.tool_call_id]
                ? data.tool_call_id
                : Object.keys(toolCallMap).find(k => toolCallMap[k].status === 'running');
              if (key) {
                toolCallMap[key] = { ...toolCallMap[key], result: rs, status: 'done' };
                updateMsg(m => ({ ...m, toolCalls: Object.values(toolCallMap) }));
                // Capture execute_python output for the right-panel REPL widget
                if (toolCallMap[key].name === 'execute_python') {
                  const code = toolCallMap[key].args?.code ?? '';
                  const output = extractPythonOutput(rs);
                  const plotMatch = rs.match(/Generated (\d+) Plotly/);
                  setPythonOutputs(prev => [...prev, {
                    id: key,
                    code,
                    output,
                    hasError: /Traceback \(most recent call last\)|^\w+Error:|^\w+Exception:/m.test(output),
                    plotCount: plotMatch ? parseInt(plotMatch[1], 10) : 0,
                  }]);
                }
              }
            }
          } catch { /* ignore parse errors */ }
        }
      }
    } catch (e: unknown) {
      // AbortError may arrive as a DOMException (not always `instanceof Error`),
      // so read `.name` structurally to match the prior `e?.name` behavior exactly.
      const aborted = (e as { name?: unknown } | null | undefined)?.name === 'AbortError';
      if (!aborted) console.error(e);
      const runningKeys = Object.keys(toolCallMap).filter(k => toolCallMap[k].status === 'running');
      if (runningKeys.length > 0) {
        runningKeys.forEach(k => {
          toolCallMap[k] = {
            ...toolCallMap[k],
            status: aborted ? 'done' : 'error',
            result: toolCallMap[k].result ?? (aborted ? 'Stopped by user' : 'Connection error'),
          };
        });
        updateMsg(m => ({ ...m, toolCalls: Object.values(toolCallMap) }));
      }
      if (aborted) {
        updateMsg(m => ({ ...m, content: (m.content || '') + (m.content ? '\n\n' : '') + '_⏹ Stopped by user._' }));
      }
    } finally {
      abortRef.current = null;
      setLoading(false);
      // Refresh artifacts + causal panels after the turn so newly-saved
      // snippets, assumptions, files, DAG, and workflow status all show up.
      if (threadId) {
        await onTurnSettled?.();
      }
    }
  };

  return {
    messages,
    dashboardData,
    pythonOutputs,
    loading,
    send,
    stop,
    clear,
    loadThreadState,
    setDashboardData,
    setMessages,
    setPythonOutputs,
    setLoading,
  };
}
