import { useEffect, useMemo, useState } from 'react';
import { clsx } from 'clsx';
import { Check, FlaskConical, Loader2 } from 'lucide-react';
import { Button, DataTable, Drawer, type Column } from '../../components/ui';
import { useDesignWave, useIngestWave } from '../../api/hooks/useLearning';
import type { LearningProgram } from '../../api/services/learningService';
import { effectiveChannels, errorDetail, fmtDollars } from './format';

interface CellRow {
  label: string;
  dollars: number[];
}

/**
 * Design the next wave: a central-composite design over the program's
 * channels (delta = multiplicative spend variation, optional synergy probe
 * pairs, optional holdout geos), then paste the observed geo panel back in to
 * ingest + refit (non-blocking job).
 */
export function DesignWaveStudio({
  open,
  onClose,
  projectId,
  program,
}: {
  open: boolean;
  onClose: () => void;
  projectId: string | null;
  program: LearningProgram;
}) {
  const dims = useMemo(() => effectiveChannels(program), [program]);

  const [delta, setDelta] = useState(0.6);
  const [nHoldout, setNHoldout] = useState(0);
  const [selectedPairs, setSelectedPairs] = useState<Set<string>>(new Set());
  const [stratify, setStratify] = useState(true);
  const [optimize, setOptimize] = useState(false);
  const [candidateDeltasText, setCandidateDeltasText] = useState('0.3, 0.6, 0.9');
  const [csvText, setCsvText] = useState('');
  const [error, setError] = useState<string | null>(null);

  const design = useDesignWave(projectId, program.id);
  const ingest = useIngestWave(projectId, program.id);

  const fitted = program.summary != null; // KG needs a fitted posterior

  useEffect(() => {
    if (open) {
      setDelta(0.6);
      setNHoldout(0);
      setSelectedPairs(new Set());
      setStratify(true);
      setOptimize(false);
      setCandidateDeltasText('0.3, 0.6, 0.9');
      setCsvText('');
      setError(null);
      design.reset();
      ingest.reset();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, program.id]);

  const candidateDeltas = useMemo(
    () =>
      candidateDeltasText
        .split(/[,\s]+/)
        .map(Number)
        .filter((d) => Number.isFinite(d) && d > 0 && d <= 1)
        .slice(0, 8),
    [candidateDeltasText],
  );

  const allPairs = useMemo(() => {
    const out: [number, number][] = [];
    for (let i = 0; i < dims.length; i += 1) {
      for (let j = i + 1; j < dims.length; j += 1) out.push([i, j]);
    }
    return out;
  }, [dims]);

  const togglePair = (key: string) =>
    setSelectedPairs((s) => {
      const next = new Set(s);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });

  const runDesign = async () => {
    setError(null);
    const probe = allPairs.filter(([i, j]) => selectedPairs.has(`${i}-${j}`));
    try {
      await design.mutateAsync({
        delta,
        // always sent explicitly: [] = probe-free design; omitting the key
        // would make the backend probe ALL program pairs
        probe_pairs: probe,
        ...(nHoldout > 0 ? { n_holdout: nHoldout } : {}),
        stratify,
        ...(optimize
          ? {
              optimize: true,
              ...(candidateDeltas.length > 0 ? { candidate_deltas: candidateDeltas } : {}),
            }
          : {}),
      });
    } catch (e) {
      setError(errorDetail(e));
    }
  };

  const result = design.data ?? null;
  const cellRows: CellRow[] = useMemo(() => {
    if (!result) return [];
    return result.cell_labels.map((label, r) => ({
      label,
      dollars: result.cells_dollars[r] ?? [],
    }));
  }, [result]);

  const cellColumns: Column<CellRow>[] = useMemo(
    () => [
      {
        key: 'cell',
        header: 'Cell',
        render: (r) => <span className="font-medium text-ink-900">{r.label}</span>,
      },
      ...dims.map((ch, k) => ({
        key: ch,
        header: ch,
        numeric: true,
        render: (r: CellRow) => fmtDollars(r.dollars[k]),
      })),
    ],
    [dims],
  );

  const jobStatus = ingest.job.data?.status ?? null;
  const running =
    ingest.start.isPending || jobStatus === 'pending' || jobStatus === 'running';
  const jobResult = jobStatus === 'done' ? ingest.job.data?.result ?? null : null;
  const jobError = ingest.start.isError
    ? errorDetail(ingest.start.error)
    : jobStatus === 'error'
      ? ingest.job.data?.error ?? 'Fit failed'
      : null;

  return (
    <Drawer open={open} onClose={onClose} title="Design the next wave" width="max-w-2xl">
      <div className="space-y-5">
        {/* ── Inputs ── */}
        <label className="block text-sm">
          <span className="mb-1 flex items-baseline justify-between font-medium text-ink-700">
            Spend variation (δ, multiplicative)
            <span className="num text-xs text-ink-400">±{Math.round(delta * 100)}%</span>
          </span>
          <input
            type="range"
            min={0.1}
            max={1.0}
            step={0.05}
            value={delta}
            onChange={(e) => setDelta(Number(e.target.value))}
            className="w-full accent-sage-700"
          />
          <span className="mt-0.5 block text-xs text-ink-400">
            Axial cells move each channel to center × (1 ± δ); bigger swings identify the
            curve faster but cost more short-term efficiency.
          </span>
        </label>

        {allPairs.length > 0 && (
          <div>
            <span className="mb-1.5 block text-sm font-medium text-ink-700">
              Synergy probe pairs <span className="font-normal text-ink-400">(optional)</span>
            </span>
            <div className="flex flex-wrap gap-1.5">
              {allPairs.map(([i, j]) => {
                const key = `${i}-${j}`;
                const on = selectedPairs.has(key);
                return (
                  <button
                    key={key}
                    type="button"
                    onClick={() => togglePair(key)}
                    className={clsx(
                      'rounded-full border px-2.5 py-1 text-xs transition-colors',
                      on
                        ? 'border-sage-600 bg-sage-100 font-medium text-sage-800'
                        : 'border-line-300 bg-white text-ink-600 hover:bg-cream-100',
                    )}
                  >
                    {dims[i]} × {dims[j]}
                  </button>
                );
              })}
            </div>
            <p className="mt-1 text-xs text-ink-400">
              Each probed pair adds 2 off-axis cells that move both channels together —
              the only way to identify γ (cannibalization / reinforcement).
            </p>
          </div>
        )}

        <label className="block text-sm">
          <span className="mb-1 block font-medium text-ink-700">Holdout geos</span>
          <input
            type="number"
            min={0}
            max={20}
            value={nHoldout}
            onChange={(e) => setNHoldout(Math.max(0, Math.round(Number(e.target.value) || 0)))}
            className="num w-28 rounded-md border border-line-300 bg-white px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-sage-600"
          />
        </label>

        <label className="flex items-start gap-2 text-sm">
          <input
            type="checkbox"
            checked={stratify}
            onChange={(e) => setStratify(e.target.checked)}
            className="mt-0.5 accent-sage-700"
          />
          <span>
            <span className="font-medium text-ink-700">Stratified geo assignment</span>
            <span className="mt-0.5 block text-xs text-ink-400">
              Block-randomize geos into cells on their accumulated KPI (once the program has
              ingested data) so no cell is dominated by big markets. Off = shuffled round-robin.
            </span>
          </span>
        </label>

        <div className="space-y-2">
          <label className="flex items-start gap-2 text-sm">
            <input
              type="checkbox"
              checked={optimize}
              onChange={(e) => setOptimize(e.target.checked)}
              disabled={!fitted}
              className="mt-0.5 accent-sage-700"
            />
            <span>
              <span className={clsx('font-medium', fitted ? 'text-ink-700' : 'text-ink-400')}>
                Let the model pick δ (knowledge gradient)
              </span>
              <span className="mt-0.5 block text-xs text-ink-400">
                {fitted
                  ? 'Scores each candidate δ by the expected value of the information it buys ' +
                    '(Laplace-KG) and designs the best one; the slider above becomes the fallback.'
                  : 'Needs a fitted posterior — record a wave or import experiments and fit first.'}
              </span>
            </span>
          </label>
          {optimize && fitted && (
            <label className="block pl-6 text-sm">
              <span className="mb-1 block text-xs font-medium text-ink-600">
                Candidate δ values (comma-separated, ≤ 8)
              </span>
              <input
                value={candidateDeltasText}
                onChange={(e) => setCandidateDeltasText(e.target.value)}
                placeholder="0.3, 0.6, 0.9"
                className="num w-48 rounded-md border border-line-300 bg-white px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-sage-600"
              />
            </label>
          )}
        </div>

        <Button onClick={runDesign} disabled={design.isPending || !projectId}>
          {design.isPending
            ? optimize
              ? 'Scoring candidates…'
              : 'Designing…'
            : result
              ? 'Redesign wave'
              : 'Design wave'}
        </Button>

        {error && <p className="text-sm text-rust-600">{error}</p>}

        {/* ── Design result ── */}
        {result && (
          <div className="space-y-3 border-t border-line-200 pt-4">
            {result.kg?.used && (
              <div className="rounded-md border border-sage-300 bg-sage-100/60 px-3 py-2.5 text-xs text-sage-800">
                <p className="font-medium">
                  Knowledge gradient chose δ ={' '}
                  <span className="num">{result.kg.chosen_delta.toFixed(2)}</span>
                </p>
                <p className="mt-1 text-sage-700">
                  {result.kg.scores
                    .map((s) => `δ ${s.delta.toFixed(2)} → EVSI ${s.score.toFixed(3)}`)
                    .join(' · ')}
                </p>
              </div>
            )}
            <h4 className="text-sm font-semibold text-ink-900">
              Wave cells{' '}
              <span className="font-normal text-ink-400">
                (<span className="num">{result.n_cells}</span> cells · $/period per treated geo)
              </span>
            </h4>
            <DataTable columns={cellColumns} rows={cellRows} rowKey={(r) => r.label} />
            <p className="text-xs text-ink-400">
              Cells rotate round-robin across the geo panel
              {nHoldout > 0 && (
                <>
                  {' '}with <span className="num">{nHoldout}</span> holdout geo
                  {nHoldout > 1 ? 's' : ''} kept at business-as-usual
                </>
              )}
              . The shutoff cells (a channel at $0) are what separate each channel's β from
              the interactions — keep them in the flight.
            </p>
            {result.warnings.length > 0 && (
              <ul className="space-y-1">
                {result.warnings.map((w) => (
                  <li key={w} className="rounded-md bg-gold-100 px-3 py-2 text-xs text-gold-700">
                    {w}
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}

        {/* ── Record results ── */}
        <div className="space-y-3 border-t border-line-200 pt-4">
          <h4 className="flex items-center gap-1.5 text-sm font-semibold text-ink-900">
            <FlaskConical className="h-4 w-4 text-sage-700" />
            I ran this wave — record results
          </h4>
          <p className="text-xs text-ink-400">
            Paste the observed panel as CSV — one row per geo-period: the geo id, each
            channel's actual $ spend, then the KPI outcome.
          </p>
          <textarea
            value={csvText}
            onChange={(e) => setCsvText(e.target.value)}
            rows={6}
            spellCheck={false}
            placeholder={`geo,${dims.join(',')},y\ngeo_01,120000,90000,48210\ngeo_02,60000,150000,51840`}
            className="num w-full rounded-md border border-line-300 bg-white px-3 py-2 font-mono text-xs text-ink-900 placeholder:text-ink-300 focus:outline-none focus:ring-2 focus:ring-sage-600"
          />
          <div className="flex items-center gap-3">
            <Button
              onClick={() => ingest.start.mutate({ csv_text: csvText })}
              disabled={!csvText.trim() || running || !projectId}
            >
              {running ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" /> Fitting on all evidence…
                </>
              ) : (
                'Ingest & refit'
              )}
            </Button>
            {running && (
              <span className="text-xs text-ink-400">
                Refitting the response surface on every wave so far — usually under a minute.
              </span>
            )}
          </div>

          {jobError && <p className="text-sm text-rust-600">{jobError}</p>}

          {jobResult && (
            <div className="space-y-1.5 rounded-md border border-sage-300 bg-sage-100/60 px-3 py-2.5 text-sm text-sage-800">
              <p className="flex items-center gap-1.5 font-medium">
                <Check className="h-4 w-4" /> Wave ingested — surface refit.
              </p>
              <p className="text-xs">
                E[regret] now{' '}
                <span className="num font-medium">
                  {fmtDollars(jobResult.regret.e_regret_dollars)}
                </span>{' '}
                · ENBS <span className="num font-medium">{fmtDollars(jobResult.regret.enbs)}</span>{' '}
                · verdict:{' '}
                <span className="font-medium">
                  {jobResult.regret.stop ? 'stop testing' : 'keep testing'}
                </span>
                . The program page has the full readout.
              </p>
            </div>
          )}
        </div>
      </div>
    </Drawer>
  );
}
