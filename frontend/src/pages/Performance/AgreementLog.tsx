import { useMemo } from 'react';
import { FlaskConical } from 'lucide-react';
import { clsx } from 'clsx';
import { DataTable, EmptyState, StatusChip } from '../../components/ui';
import type { Column } from '../../components/ui';
import { useExperimentRegistry, useProjectHistory } from '../../api/hooks/useMeasurement';
import type { ExperimentRecord, RoiPoint } from '../../api/services/measurementService';

interface AgreementRow {
  exp: ExperimentRecord;
  roi: RoiPoint | undefined;
  /** Standardized disagreement; null when the estimands aren't comparable. */
  z: number | null;
}

function AgreementBadge({ row }: { row: AgreementRow }) {
  if (row.z == null) {
    return (
      <span
        className="text-ink-300"
        title={
          row.exp.estimand !== 'roas'
            ? `Estimand "${row.exp.estimand ?? 'unknown'}" is not directly comparable to the model's ROI posterior — only ROAS readouts get a z-score.`
            : 'Missing experiment or model uncertainty — cannot compute a z-score.'
        }
      >
        —
      </span>
    );
  }
  const abs = Math.abs(row.z);
  const tone =
    abs < 1
      ? 'bg-sage-100 text-sage-800'
      : abs < 2
        ? 'bg-gold-100 text-gold-700'
        : 'bg-rust-100 text-rust-700';
  const label = abs < 1 ? 'agree' : abs < 2 ? 'tension' : 'disagree';
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className={clsx('rounded-full px-2 py-0.5 text-xs font-medium', tone)}>{label}</span>
      <span className="num text-xs text-ink-400">z = {row.z.toFixed(2)}</span>
    </span>
  );
}

const COLUMNS: Column<AgreementRow>[] = [
  {
    key: 'channel',
    header: 'Channel',
    render: (r) => <span className="font-medium text-ink-900">{r.exp.channel}</span>,
  },
  {
    key: 'window',
    header: 'Window',
    render: (r) =>
      r.exp.start_date || r.exp.end_date ? (
        <span className="num text-xs text-ink-600">
          {r.exp.start_date ?? '—'} → {r.exp.end_date ?? '—'}
        </span>
      ) : (
        <span className="text-ink-300">—</span>
      ),
  },
  {
    key: 'result',
    header: 'Experiment result',
    numeric: true,
    render: (r) =>
      r.exp.value != null ? (
        <span>
          {r.exp.value.toFixed(2)}
          {r.exp.se != null && <span className="text-ink-400"> ± {r.exp.se.toFixed(2)}</span>}
        </span>
      ) : (
        <span className="text-ink-300">—</span>
      ),
  },
  {
    key: 'model_roi',
    header: 'Model ROI at calibration',
    numeric: true,
    render: (r) =>
      r.roi?.mean != null ? (
        <span>
          {r.roi.mean.toFixed(2)}
          {r.roi.hdi_low != null && r.roi.hdi_high != null && (
            <span className="text-ink-400">
              {' '}
              [{r.roi.hdi_low.toFixed(2)}, {r.roi.hdi_high.toFixed(2)}]
            </span>
          )}
        </span>
      ) : (
        <span className="text-ink-300">—</span>
      ),
  },
  {
    key: 'agreement',
    header: 'Agreement',
    render: (r) => <AgreementBadge row={r} />,
  },
  {
    key: 'status',
    header: 'Status',
    render: (r) => <StatusChip status={r.exp.status} />,
  },
];

export function AgreementLog({ projectId }: { projectId: string }) {
  const { data: experiments, isLoading: experimentsLoading } = useExperimentRegistry(projectId);
  const { data: history } = useProjectHistory(projectId);

  const rows = useMemo<AgreementRow[]>(() => {
    const readouts = (experiments ?? []).filter(
      (e) => e.status === 'completed' || e.status === 'calibrated',
    );
    return readouts.map((exp) => {
      const roiPts = history?.series.roi[exp.channel] ?? [];
      const roi = exp.calibrated_run_id
        ? roiPts.find((p) => p.run_id === exp.calibrated_run_id)
        : undefined;
      let z: number | null = null;
      if (
        exp.estimand === 'roas' &&
        exp.value != null &&
        exp.se != null &&
        roi?.mean != null &&
        roi?.sd != null
      ) {
        const denom = Math.sqrt(exp.se ** 2 + roi.sd ** 2);
        z = denom > 0 ? (exp.value - roi.mean) / denom : null;
      }
      return { exp, roi, z };
    });
  }, [experiments, history]);

  if (experimentsLoading) {
    return <p className="text-sm text-ink-400">Loading experiments…</p>;
  }

  return (
    <div className="space-y-4">
      <p className="max-w-3xl text-sm text-ink-600">
        Each row pairs an independent experimental estimate with the model's posterior at the run
        where it was calibrated. Agreement within noise builds trust in the model's causal reads;
        systematic disagreement is a sign of model bias worth chasing down.
      </p>
      <DataTable
        columns={COLUMNS}
        rows={rows}
        rowKey={(r) => r.exp.id}
        empty={
          <EmptyState
            icon={FlaskConical}
            title="No experiment readouts yet"
            description="Completed and calibrated experiments appear here once their readouts land — each one is an independent check on the model."
          />
        }
      />
    </div>
  );
}
