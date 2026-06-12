import { EXPERIMENT_STATUS, type ExperimentStatus } from '../../theme/colors';

interface StatusChipProps {
  status: string;
  label?: string;
}

/** Experiment lifecycle chip (draft → planned → running → completed → calibrated). */
export function StatusChip({ status, label }: StatusChipProps) {
  const s =
    EXPERIMENT_STATUS[status as ExperimentStatus] ?? {
      fg: '#4a5a48',
      bg: '#f0ede0',
      label: status,
    };
  return (
    <span
      className="inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium"
      style={{ color: s.fg, backgroundColor: s.bg }}
    >
      {label ?? s.label}
    </span>
  );
}
