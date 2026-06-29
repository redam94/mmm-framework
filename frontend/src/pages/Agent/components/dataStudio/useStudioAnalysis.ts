import { useEffect, useState } from 'react';
import type { StudioEdaResult } from '../../types';

type RunEda = (analyses: string[], opts?: { sensitivity?: string }) => Promise<StudioEdaResult | null>;

// Fetch the requested analyses on mount and whenever the pipeline changes (rev).
export function useStudioAnalysis(
  runEda: RunEda, analyses: string[], rev: number, sensitivity?: string,
) {
  const [res, setRes] = useState<StudioEdaResult | null>(null);
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    let alive = true;
    setLoading(true);
    runEda(analyses, sensitivity ? { sensitivity } : undefined).then(r => {
      if (alive) { setRes(r); setLoading(false); }
    });
    return () => { alive = false; };
    // analyses is a stable per-panel literal; refetch on pipeline edits + sensitivity.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rev, sensitivity]);
  return { res, loading };
}
