import Plot from 'react-plotly.js';
import type { Config, Data, Layout } from 'plotly.js';
import { CHART_COLORWAY } from '../../theme/colors';
import { mmmPlotlyLayout, PLOTLY_CONFIG } from '../../theme/plotlyTheme';
import type { FlightingSchedule } from '../../api/services/plannerService';

/** Forward budget calendar (B6): one stacked bar per future period, segmented by
 * channel. The temporal half of a plan — when each channel's allocated budget
 * lands, week by week. */
export function FlightingCalendar({
  flighting,
  height = 260,
}: {
  flighting: FlightingSchedule;
  height?: number;
}) {
  const { periods, channels, by_channel } = flighting;
  const data: Data[] = channels.map((ch, i) => ({
    x: periods,
    y: by_channel[ch] ?? [],
    type: 'bar',
    name: ch,
    marker: { color: CHART_COLORWAY[i % CHART_COLORWAY.length] },
  }));

  const layout = mmmPlotlyLayout({
    height,
    barmode: 'stack',
    margin: { t: 10, l: 60, r: 16, b: 56 },
    xaxis: { title: { text: 'period' } },
    yaxis: { title: { text: 'spend' } },
    showlegend: true,
    legend: { orientation: 'h', y: -0.25 },
  } as Partial<Layout>);

  return (
    <Plot
      data={data}
      layout={layout}
      config={PLOTLY_CONFIG as Partial<Config>}
      style={{ width: '100%' }}
      useResizeHandler
    />
  );
}
