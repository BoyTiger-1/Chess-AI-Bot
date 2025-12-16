import React from 'react';
import { Box, Stack } from '@mui/material';
import { PageHeader } from '@/components/PageHeader';
import { ExportPdfButton } from '@/components/ExportPdfButton';
import { DashboardActions } from '@/components/DashboardActions';
import { DashboardGrid, type DashboardWidget } from '@/components/DashboardGrid';
import { KpiCard } from '@/components/KpiCard';
import { WidgetCard } from '@/components/WidgetCard';
import { Plot } from '@/components/charts/Plot';
import { usePreferences } from '@/app/state/preferences';
import { useLocalStorageState } from '@/hooks/useLocalStorageState';
import { useSimulatedStream } from '@/hooks/useSimulatedStream';

function clamp(n: number, min: number, max: number) {
  return Math.max(min, Math.min(max, n));
}

export default function MarketAnalysisDashboard() {
  const containerRef = React.useRef<HTMLDivElement>(null);
  const { notifications } = usePreferences();

  const trend = useSimulatedStream(
    Array.from({ length: 24 }, (_, i) => ({ t: i, v: 100 + i * 2 + Math.random() * 4 })),
    (prev) => {
      const next = prev.slice(1);
      const last = prev[prev.length - 1];
      const v = clamp(last.v + (Math.random() - 0.4) * 6, 80, 180);
      next.push({ t: last.t + 1, v });
      return next;
    },
    { enabled: notifications.liveUpdates }
  );

  const sentiment = useSimulatedStream(62, (prev) => clamp(prev + (Math.random() - 0.45) * 8, 0, 100), {
    enabled: notifications.liveUpdates,
    intervalMs: 2500
  });

  const heat = useSimulatedStream(
    {
      x: ['North', 'South', 'East', 'West', 'Online'],
      y: ['Enterprise', 'SMB', 'Mid-market', 'Consumer'],
      z: [
        [0.42, 0.38, 0.33, 0.28, 0.46],
        [0.31, 0.29, 0.25, 0.21, 0.34],
        [0.37, 0.35, 0.29, 0.24, 0.41],
        [0.28, 0.26, 0.23, 0.19, 0.30]
      ]
    },
    (prev) => ({
      ...prev,
      z: prev.z.map((row) => row.map((v) => clamp(v + (Math.random() - 0.5) * 0.02, 0.12, 0.62)))
    }),
    { enabled: notifications.liveUpdates, intervalMs: 4000 }
  );

  const layoutKey = 'aiba.layout.market.v1';
  const [hiddenIds, setHiddenIds] = useLocalStorageState<string[]>('aiba.widgets.market.hidden.v1', []);
  const [layoutNonce, setLayoutNonce] = React.useState(0);
  const resetLayout = () => {
    localStorage.removeItem(layoutKey);
    setLayoutNonce((n) => n + 1);
  };

  const widgetList = [
    { id: 'kpi1', title: 'Market growth' },
    { id: 'kpi2', title: 'Demand index' },
    { id: 'kpi3', title: 'Volatility' },
    { id: 'trend', title: 'Trend' },
    { id: 'heatmap', title: 'Segment heatmap' },
    { id: 'sentiment', title: 'Sentiment' }
  ];

  const widgets: DashboardWidget[] = [
    {
      id: 'kpi1',
      default: { x: 0, y: 0, w: 4, h: 2 },
      render: () => <KpiCard title="Market growth" value="+8.4%" delta={{ label: '+0.6pp WoW', tone: 'positive' }} />
    },
    {
      id: 'kpi2',
      default: { x: 4, y: 0, w: 4, h: 2 },
      render: () => <KpiCard title="Demand index" value="112" delta={{ label: 'steady', tone: 'neutral' }} />
    },
    {
      id: 'kpi3',
      default: { x: 8, y: 0, w: 4, h: 2 },
      render: () => <KpiCard title="Volatility" value="low" delta={{ label: '-12% vs last month', tone: 'positive' }} />
    },
    {
      id: 'trend',
      minH: 3,
      default: { x: 0, y: 2, w: 8, h: 4 },
      render: () => (
        <WidgetCard title="Trend" subtitle="Real-time demand signal">
          <Plot
            data={[
              {
                x: trend.map((p) => p.t),
                y: trend.map((p) => p.v),
                type: 'scatter',
                mode: 'lines',
                line: { color: '#6EE7FF', width: 3 }
              }
            ]}
            layout={{
              autosize: true,
              height: 280,
              margin: { l: 40, r: 10, t: 10, b: 40 },
              paper_bgcolor: 'rgba(0,0,0,0)',
              plot_bgcolor: 'rgba(0,0,0,0)',
              xaxis: { title: 'time' },
              yaxis: { title: 'index', zeroline: false }
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: '100%' }}
          />
        </WidgetCard>
      )
    },
    {
      id: 'heatmap',
      minH: 3,
      default: { x: 8, y: 2, w: 4, h: 4 },
      render: () => (
        <WidgetCard title="Segment heatmap" subtitle="Conversion probability">
          <Plot
            data={[
              {
                type: 'heatmap',
                x: heat.x,
                y: heat.y,
                z: heat.z,
                colorscale: [
                  [0, '#0F1B33'],
                  [0.5, '#2563EB'],
                  [1, '#6EE7FF']
                ]
              }
            ]}
            layout={{
              autosize: true,
              height: 280,
              margin: { l: 70, r: 10, t: 10, b: 30 },
              paper_bgcolor: 'rgba(0,0,0,0)',
              plot_bgcolor: 'rgba(0,0,0,0)'
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: '100%' }}
          />
        </WidgetCard>
      )
    },
    {
      id: 'sentiment',
      default: { x: 0, y: 6, w: 12, h: 3 },
      render: () => (
        <WidgetCard title="Sentiment" subtitle="News + social composite">
          <Plot
            data={[
              {
                type: 'indicator',
                mode: 'gauge+number',
                value: sentiment,
                number: { suffix: '/100', font: { size: 40 } },
                gauge: {
                  axis: { range: [0, 100] },
                  bar: { color: sentiment >= 60 ? '#34D399' : sentiment >= 45 ? '#F59E0B' : '#F87171' },
                  steps: [
                    { range: [0, 45], color: 'rgba(248,113,113,0.18)' },
                    { range: [45, 60], color: 'rgba(245,158,11,0.18)' },
                    { range: [60, 100], color: 'rgba(52,211,153,0.18)' }
                  ]
                }
              }
            ]}
            layout={{
              autosize: true,
              height: 170,
              margin: { l: 20, r: 20, t: 10, b: 10 },
              paper_bgcolor: 'rgba(0,0,0,0)'
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: '100%' }}
          />
        </WidgetCard>
      )
    }
  ];

  return (
    <Box ref={containerRef}>
      <Stack direction={{ xs: 'column', sm: 'row' }} alignItems={{ sm: 'flex-end' }} justifyContent="space-between" gap={1}>
        <PageHeader title="Market analysis" subtitle="Trends, heatmaps, and sentiment indicators." />
        <Stack direction="row" spacing={1} alignItems="center">
          <DashboardActions
            layoutStorageKey={layoutKey}
            widgetList={widgetList}
            hiddenIds={hiddenIds}
            setHiddenIds={setHiddenIds}
            onResetLayout={resetLayout}
          />
          <ExportPdfButton targetRef={containerRef} fileName="market-analysis.pdf" />
        </Stack>
      </Stack>
      <DashboardGrid key={layoutNonce} widgets={widgets.filter((w) => !hiddenIds.includes(w.id))} storageKey={layoutKey} />
    </Box>
  );
}
