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

export default function FinancialForecastingDashboard() {
  const containerRef = React.useRef<HTMLDivElement>(null);
  const { notifications } = usePreferences();

  const series = useSimulatedStream(
    Array.from({ length: 18 }, (_, i) => ({ t: i, actual: 120 + i * 4 + Math.random() * 6 })),
    (prev) => {
      const last = prev[prev.length - 1];
      const next = prev.slice(1);
      next.push({ t: last.t + 1, actual: clamp(last.actual + (Math.random() - 0.35) * 10, 110, 260) });
      return next;
    },
    { enabled: notifications.liveUpdates, intervalMs: 3000 }
  );

  const forecastHorizon = 8;
  const last = series[series.length - 1];
  const fc = Array.from({ length: forecastHorizon }, (_, i) => {
    const t = last.t + i + 1;
    const base = last.actual + (i + 1) * 5 + (Math.random() - 0.5) * 4;
    const band = 14 + i * 4;
    return { t, mean: base, low: base - band, high: base + band };
  });

  const scenario = [
    { name: 'Base', v: fc[fc.length - 1]?.mean ?? 0 },
    { name: 'Upside', v: (fc[fc.length - 1]?.mean ?? 0) * 1.09 },
    { name: 'Downside', v: (fc[fc.length - 1]?.mean ?? 0) * 0.92 }
  ];

  const layoutKey = 'aiba.layout.finance.v1';
  const [hiddenIds, setHiddenIds] = useLocalStorageState<string[]>('aiba.widgets.finance.hidden.v1', []);
  const [layoutNonce, setLayoutNonce] = React.useState(0);
  const resetLayout = () => {
    localStorage.removeItem(layoutKey);
    setLayoutNonce((n) => n + 1);
  };

  const widgetList = [
    { id: 'kpiRev', title: 'Run-rate' },
    { id: 'kpiConf', title: 'Forecast confidence' },
    { id: 'kpiRisk', title: 'Risk' },
    { id: 'forecast', title: 'Revenue forecast' },
    { id: 'scenarios', title: 'Scenarios' }
  ];

  const widgets: DashboardWidget[] = [
    {
      id: 'kpiRev',
      default: { x: 0, y: 0, w: 4, h: 2 },
      render: () => <KpiCard title="Run-rate" value="$4.2M" delta={{ label: '+3.1% QoQ', tone: 'positive' }} />
    },
    {
      id: 'kpiConf',
      default: { x: 4, y: 0, w: 4, h: 2 },
      render: () => <KpiCard title="Forecast confidence" value="0.78" delta={{ label: 'stable', tone: 'neutral' }} />
    },
    {
      id: 'kpiRisk',
      default: { x: 8, y: 0, w: 4, h: 2 },
      render: () => <KpiCard title="Risk" value="medium" delta={{ label: 'macro-driven', tone: 'neutral' }} />
    },
    {
      id: 'forecast',
      minH: 4,
      default: { x: 0, y: 2, w: 8, h: 5 },
      render: () => (
        <WidgetCard title="Revenue forecast" subtitle="Confidence interval and projections">
          <Plot
            data={[
              {
                type: 'scatter',
                mode: 'lines',
                x: series.map((p) => p.t),
                y: series.map((p) => p.actual),
                line: { color: '#6EE7FF', width: 3 },
                name: 'Actual'
              },
              {
                type: 'scatter',
                mode: 'lines',
                x: fc.map((p) => p.t),
                y: fc.map((p) => p.mean),
                line: { color: '#B794F4', width: 3, dash: 'dash' },
                name: 'Forecast'
              },
              {
                type: 'scatter',
                mode: 'lines',
                x: [...fc.map((p) => p.t), ...fc.map((p) => p.t).reverse()],
                y: [...fc.map((p) => p.high), ...fc.map((p) => p.low).reverse()],
                fill: 'toself',
                fillcolor: 'rgba(183,148,244,0.18)',
                line: { color: 'rgba(0,0,0,0)' },
                hoverinfo: 'skip',
                name: 'Confidence'
              }
            ]}
            layout={{
              autosize: true,
              height: 360,
              margin: { l: 50, r: 10, t: 10, b: 40 },
              paper_bgcolor: 'rgba(0,0,0,0)',
              plot_bgcolor: 'rgba(0,0,0,0)',
              xaxis: { title: 'time' },
              yaxis: { title: 'revenue (index)', zeroline: false },
              legend: { orientation: 'h' }
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: '100%' }}
          />
        </WidgetCard>
      )
    },
    {
      id: 'scenarios',
      minH: 4,
      default: { x: 8, y: 2, w: 4, h: 5 },
      render: () => (
        <WidgetCard title="Scenarios" subtitle="Outcome comparison">
          <Plot
            data={[
              {
                type: 'bar',
                x: scenario.map((s) => s.name),
                y: scenario.map((s) => s.v),
                marker: {
                  color: ['#6EE7FF', '#34D399', '#F87171']
                }
              }
            ]}
            layout={{
              autosize: true,
              height: 360,
              margin: { l: 50, r: 10, t: 10, b: 40 },
              paper_bgcolor: 'rgba(0,0,0,0)',
              plot_bgcolor: 'rgba(0,0,0,0)',
              yaxis: { title: 'projection' }
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
        <PageHeader title="Financial forecasting" subtitle="Forecasts with confidence intervals and scenario comparisons." />
        <Stack direction="row" spacing={1} alignItems="center">
          <DashboardActions
            layoutStorageKey={layoutKey}
            widgetList={widgetList}
            hiddenIds={hiddenIds}
            setHiddenIds={setHiddenIds}
            onResetLayout={resetLayout}
          />
          <ExportPdfButton targetRef={containerRef} fileName="financial-forecast.pdf" />
        </Stack>
      </Stack>
      <DashboardGrid key={layoutNonce} widgets={widgets.filter((w) => !hiddenIds.includes(w.id))} storageKey={layoutKey} />
    </Box>
  );
}
