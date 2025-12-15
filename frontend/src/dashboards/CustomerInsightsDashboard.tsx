import React from 'react';
import { Box, Chip, Stack, Table, TableBody, TableCell, TableHead, TableRow } from '@mui/material';
import { PageHeader } from '@/components/PageHeader';
import { ExportPdfButton } from '@/components/ExportPdfButton';
import { DashboardActions } from '@/components/DashboardActions';
import { DashboardGrid, type DashboardWidget } from '@/components/DashboardGrid';
import { WidgetCard } from '@/components/WidgetCard';
import { Plot } from '@/components/charts/Plot';
import { KpiCard } from '@/components/KpiCard';
import { usePreferences } from '@/app/state/preferences';
import { useLocalStorageState } from '@/hooks/useLocalStorageState';
import { useSimulatedStream } from '@/hooks/useSimulatedStream';
import { useSearch } from '@/app/state/search';

function clamp(n: number, min: number, max: number) {
  return Math.max(min, Math.min(max, n));
}

const SEGMENTS = [
  { name: 'High growth', value: 0.28 },
  { name: 'Core', value: 0.42 },
  { name: 'At-risk', value: 0.17 },
  { name: 'New', value: 0.13 }
] as const;

const CHURN = [
  { account: 'Orchid Retail', risk: 'high', signal: 'usage drop' },
  { account: 'BluePeak Logistics', risk: 'medium', signal: 'support volume' },
  { account: 'Helio Fintech', risk: 'low', signal: 'renewal confirmed' },
  { account: 'Summit Health', risk: 'medium', signal: 'seat contraction' },
  { account: 'Riverstone SaaS', risk: 'high', signal: 'champion left' }
] as const;

export default function CustomerInsightsDashboard() {
  const containerRef = React.useRef<HTMLDivElement>(null);
  const { notifications } = usePreferences();
  const { query } = useSearch();

  const behavior = useSimulatedStream(
    Array.from({ length: 20 }, (_, i) => ({ t: i, active: 74 + i * 0.4 + Math.random() * 2.4 })),
    (prev) => {
      const last = prev[prev.length - 1];
      const next = prev.slice(1);
      next.push({ t: last.t + 1, active: clamp(last.active + (Math.random() - 0.45) * 2.4, 68, 86) });
      return next;
    },
    { enabled: notifications.liveUpdates, intervalMs: 2800 }
  );

  const filteredChurn = CHURN.filter((c) => `${c.account} ${c.signal} ${c.risk}`.toLowerCase().includes(query.trim().toLowerCase()));

  const layoutKey = 'aiba.layout.customer.v1';
  const [hiddenIds, setHiddenIds] = useLocalStorageState<string[]>('aiba.widgets.customer.hidden.v1', []);
  const [layoutNonce, setLayoutNonce] = React.useState(0);
  const resetLayout = () => {
    localStorage.removeItem(layoutKey);
    setLayoutNonce((n) => n + 1);
  };

  const widgetList = [
    { id: 'kpiRetention', title: 'Retention' },
    { id: 'kpiNps', title: 'NPS' },
    { id: 'kpiChurn', title: 'Churn risk' },
    { id: 'segmentation', title: 'Segmentation' },
    { id: 'behavior', title: 'Behavior trends' },
    { id: 'churn', title: 'Churn predictions' }
  ];

  const widgets: DashboardWidget[] = [
    {
      id: 'kpiRetention',
      default: { x: 0, y: 0, w: 4, h: 2 },
      render: () => <KpiCard title="Retention" value="93.1%" delta={{ label: '+0.4pp MoM', tone: 'positive' }} />
    },
    {
      id: 'kpiNps',
      default: { x: 4, y: 0, w: 4, h: 2 },
      render: () => <KpiCard title="NPS" value="47" delta={{ label: '+3', tone: 'positive' }} />
    },
    {
      id: 'kpiChurn',
      default: { x: 8, y: 0, w: 4, h: 2 },
      render: () => <KpiCard title="Churn risk" value="elevated" delta={{ label: 'watchlist: 5', tone: 'neutral' }} />
    },
    {
      id: 'segmentation',
      minH: 4,
      default: { x: 0, y: 2, w: 5, h: 5 },
      render: () => (
        <WidgetCard title="Segmentation" subtitle="Portfolio distribution">
          <Plot
            data={[
              {
                type: 'pie',
                labels: SEGMENTS.map((s) => s.name),
                values: SEGMENTS.map((s) => s.value),
                hole: 0.6
              }
            ]}
            layout={{
              autosize: true,
              height: 360,
              margin: { l: 10, r: 10, t: 10, b: 10 },
              paper_bgcolor: 'rgba(0,0,0,0)',
              legend: { orientation: 'h' }
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: '100%' }}
          />
        </WidgetCard>
      )
    },
    {
      id: 'behavior',
      minH: 4,
      default: { x: 5, y: 2, w: 7, h: 5 },
      render: () => (
        <WidgetCard title="Behavior trends" subtitle="Active usage">
          <Plot
            data={[
              {
                type: 'scatter',
                mode: 'lines',
                x: behavior.map((p) => p.t),
                y: behavior.map((p) => p.active),
                line: { color: '#6EE7FF', width: 3 }
              }
            ]}
            layout={{
              autosize: true,
              height: 360,
              margin: { l: 50, r: 10, t: 10, b: 40 },
              paper_bgcolor: 'rgba(0,0,0,0)',
              plot_bgcolor: 'rgba(0,0,0,0)',
              xaxis: { title: 'time' },
              yaxis: { title: 'active (%)', range: [65, 90] }
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: '100%' }}
          />
        </WidgetCard>
      )
    },
    {
      id: 'churn',
      minH: 4,
      default: { x: 0, y: 7, w: 12, h: 4 },
      render: () => (
        <WidgetCard title="Churn predictions" subtitle="Accounts requiring attention">
          <Box sx={{ overflowX: 'auto' }}>
            <Table size="small" aria-label="Churn predictions">
              <TableHead>
                <TableRow>
                  <TableCell sx={{ fontWeight: 800 }}>Account</TableCell>
                  <TableCell sx={{ fontWeight: 800 }}>Risk</TableCell>
                  <TableCell sx={{ fontWeight: 800 }}>Signal</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {filteredChurn.map((row) => (
                  <TableRow key={row.account} hover>
                    <TableCell>{row.account}</TableCell>
                    <TableCell>
                      <Chip
                        size="small"
                        label={row.risk}
                        color={row.risk === 'high' ? 'error' : row.risk === 'medium' ? 'warning' : 'success'}
                      />
                    </TableCell>
                    <TableCell>{row.signal}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Box>
        </WidgetCard>
      )
    }
  ];

  return (
    <Box ref={containerRef}>
      <Stack direction={{ xs: 'column', sm: 'row' }} alignItems={{ sm: 'flex-end' }} justifyContent="space-between" gap={1}>
        <PageHeader title="Customer insights" subtitle="Segmentation, behavior trends, and churn predictions." />
        <Stack direction="row" spacing={1} alignItems="center">
          <DashboardActions
            layoutStorageKey={layoutKey}
            widgetList={widgetList}
            hiddenIds={hiddenIds}
            setHiddenIds={setHiddenIds}
            onResetLayout={resetLayout}
          />
          <ExportPdfButton targetRef={containerRef} fileName="customer-insights.pdf" />
        </Stack>
      </Stack>

      <DashboardGrid key={layoutNonce} widgets={widgets.filter((w) => !hiddenIds.includes(w.id))} storageKey={layoutKey} />
    </Box>
  );
}
