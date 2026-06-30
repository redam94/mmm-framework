import { Download, Loader2, Presentation, AlertCircle } from 'lucide-react';

import { useSlideDeck } from '../../../../api/hooks/useMeasurement';
import { API_BASE } from '../../constants';
import { DashWidget } from '../common/DashWidget';

/**
 * "Slide deck (PowerPoint)" widget for the Results tab. Starts the non-blocking
 * project deck job (model numbers + charts + AI per-slide insights + a
 * synthesized headline → the filled template), polls its stages, and offers the
 * .pptx for download. Also surfaces a deck the chat agent generated for this
 * session (``dashboardData.slide_deck_path``).
 */
export function SlideDeckWidget({
  projectId,
  threadId,
  chatDeckPath,
}: {
  projectId: string | null;
  threadId: string | null;
  chatDeckPath?: string | null;
}) {
  const { start, job, jobId, reset } = useSlideDeck(projectId);
  const status = job.data?.status;
  const stage = job.data?.stage;
  const generating = start.isPending || status === 'pending' || status === 'running';
  const done = status === 'done';
  const failed = status === 'error' || start.isError;
  const errMsg = job.data?.error || (start.error as Error | undefined)?.message;

  const projectDownload =
    done && jobId
      ? `${API_BASE}/projects/${projectId}/generate-deck/${jobId}/download`
      : null;
  const chatDownload =
    chatDeckPath && threadId
      ? `${API_BASE}/slide-deck/download?thread_id=${encodeURIComponent(threadId)}`
      : null;

  return (
    <DashWidget title="Slide deck (PowerPoint)" dotColor="bg-orange-500" color="orange">
      <div className="flex flex-col gap-3">
        <p className="text-sm text-ink-400">
          A branded client readout: headline KPIs, the channel scorecard, and a deep-dive per
          channel with breakthrough / optimal / saturation spend zones (on ROI &amp; marginal ROI).
          Each slide gets an AI insight; the headline is synthesized across the deck.
        </p>

        <div className="flex flex-wrap items-center gap-3">
          <button
            type="button"
            disabled={!projectId || generating}
            onClick={() => {
              reset();
              start.mutate({ kpi_name: 'Revenue', currency: '$' });
            }}
            className="flex items-center gap-2 px-4 py-2 bg-orange-600 hover:bg-orange-500 disabled:opacity-50 disabled:cursor-not-allowed text-white text-sm rounded-xl transition-colors font-medium"
          >
            {generating ? <Loader2 size={15} className="animate-spin" /> : <Presentation size={15} />}
            {generating ? 'Generating…' : 'Generate slide deck'}
          </button>

          {projectDownload && (
            <a
              href={projectDownload}
              download="mmm_slide_deck.pptx"
              className="flex items-center gap-2 px-4 py-2 bg-cream-100 hover:bg-gray-200 text-ink-700 text-sm rounded-xl transition-colors font-medium border border-line-200"
            >
              <Download size={15} /> Download .pptx
            </a>
          )}
        </div>

        {generating && (
          <p className="text-xs text-ink-400">
            {stage ? `${stage}…` : 'Building deck…'} — model → AI insights → render. This can take a
            minute.
          </p>
        )}
        {done && job.data?.result && (
          <p className="text-xs text-emerald-600">
            Built {job.data.result.n_slides} slides
            {job.data.result.n_insights ? ` with ${job.data.result.n_insights} AI insights` : ''}.
          </p>
        )}
        {failed && (
          <p className="flex items-center gap-1.5 text-xs text-rose-600">
            <AlertCircle size={13} /> {errMsg || 'Deck generation failed.'}
          </p>
        )}

        {chatDownload && (
          <div className="mt-1 border-t border-line-200 pt-3">
            <p className="text-xs text-ink-400 mb-2">A deck generated from chat is available:</p>
            <a
              href={chatDownload}
              download="mmm_slide_deck.pptx"
              className="inline-flex items-center gap-2 px-3 py-1.5 bg-cream-100 hover:bg-gray-200 text-ink-700 text-xs rounded-lg transition-colors font-medium border border-line-200"
            >
              <Download size={14} /> Download chat deck
            </a>
          </div>
        )}
      </div>
    </DashWidget>
  );
}
