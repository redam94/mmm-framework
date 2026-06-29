import { PlotCard } from '../plots/PlotCard';
import { DataTable } from '../common/DataTable';
import type { StudioChart, StudioTableSpec, TableSpec } from '../../types';

// Studio EDA charts arrive with inline data/layout (no content-addressed id), so
// PlotCard/usePlotFigure render them directly — never fetching /plots/{id}. The
// `key` is used only as a React key, never as a plot ref.
export function StudioEdaChart({ chart, idx }: { chart: StudioChart; idx: number }) {
  return <PlotCard plot={{ title: chart.title, data: chart.data, layout: chart.layout }} idx={idx} />;
}

// Adapt the studio's compact {title, columns:[str], rows:[[...]]} table into the
// TableSpec the shared DataTable renders.
export function StudioTable({ table }: { table: StudioTableSpec }) {
  const spec: TableSpec = {
    title: table.title,
    source: 'data_studio',
    columns: table.columns.map(c => ({ key: c, label: c })),
    rows: table.rows.map(r => Object.fromEntries(table.columns.map((c, i) => [c, r[i]]))),
  };
  return <DataTable table={spec} maxHeight={320} />;
}
