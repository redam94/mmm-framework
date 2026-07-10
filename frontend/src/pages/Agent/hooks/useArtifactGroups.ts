import { useMemo } from 'react';
import { buildArtifactGroups } from '../utils/artifactGroups';
import type { ArtifactGroup } from '../utils/artifactGroups';
import type { ChatMessage, PlotRef, PythonOutput, TableRef } from '../types';

/** Memoized grouped-by-question artifact timeline for the Results tab. */
export function useArtifactGroups(
  messages: ChatMessage[],
  plots: PlotRef[] | undefined,
  tables: TableRef[] | undefined,
  pythonOutputs: PythonOutput[] | undefined,
): ArtifactGroup[] {
  return useMemo(
    () => buildArtifactGroups(messages, plots, tables, pythonOutputs),
    [messages, plots, tables, pythonOutputs],
  );
}
