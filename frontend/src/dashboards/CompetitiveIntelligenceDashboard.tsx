import React from 'react';
import {
  Box,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Typography,
  Chip
} from '@mui/material';
import { PageHeader } from '@/components/PageHeader';
import { ExportPdfButton } from '@/components/ExportPdfButton';
import { DashboardActions } from '@/components/DashboardActions';
import { DashboardGrid, type DashboardWidget } from '@/components/DashboardGrid';
import { WidgetCard } from '@/components/WidgetCard';
import { Plot } from '@/components/charts/Plot';
import { useLocalStorageState } from '@/hooks/useLocalStorageState';
import { useSearch } from '@/app/state/search';

const COMPETITORS = [
  { name: 'ApexIQ', share: 0.24, price: 0.72, features: 0.82, momentum: 'up' },
  { name: 'Northstar', share: 0.19, price: 0.58, features: 0.74, momentum: 'flat' },
  { name: 'ClarityOps', share: 0.14, price: 0.41, features: 0.69, momentum: 'down' },
  { name: 'KiteMetrics', share: 0.11, price: 0.36, features: 0.61, momentum: 'up' },
  { name: 'AtlasWorks', share: 0.09, price: 0.47, features: 0.57, momentum: 'flat' }
] as const;

const FEATURE_MATRIX = [
  { feature: 'Real-time dashboards', ApexIQ: true, Northstar: true, ClarityOps: false, KiteMetrics: true, AtlasWorks: false },
  { feature: 'Explainable recommendations', ApexIQ: true, Northstar: false, ClarityOps: true, KiteMetrics: false, AtlasWorks: false },
  { feature: 'RBAC & SSO', ApexIQ: true, Northstar: true, ClarityOps: true, KiteMetrics: false, AtlasWorks: true },
  { feature: 'PDF reporting', ApexIQ: true, Northstar: true, ClarityOps: false, KiteMetrics: true, AtlasWorks: false },
  { feature: 'Competitive landscape maps', ApexIQ: false, Northstar: false, ClarityOps: true, KiteMetrics: true, AtlasWorks: false }
] as const;

export default function CompetitiveIntelligenceDashboard() {
  const containerRef = React.useRef<HTMLDivElement>(null);
  const { query } = useSearch();

  const competitorsFiltered = COMPETITORS.filter((c) =>
    `${c.name} ${c.momentum}`.toLowerCase().includes(query.trim().toLowerCase())
  );

  const layoutKey = 'aiba.layout.competitive.v1';
  const [hiddenIds, setHiddenIds] = useLocalStorageState<string[]>('aiba.widgets.competitive.hidden.v1', []);
  const [layoutNonce, setLayoutNonce] = React.useState(0);
  const resetLayout = () => {
    localStorage.removeItem(layoutKey);
    setLayoutNonce((n) => n + 1);
  };

  const widgetList = [
    { id: 'positioning', title: 'Positioning' },
    { id: 'share', title: 'Market share' },
    { id: 'matrix', title: 'Feature matrix' },
    { id: 'network', title: 'Landscape network' }
  ];

  const widgets: DashboardWidget[] = [
    {
      id: 'positioning',
      minH: 4,
      default: { x: 0, y: 0, w: 7, h: 5 },
      render: () => (
        <WidgetCard title="Positioning" subtitle="Price vs. feature depth (bubble = share)">
          <Plot
            data={[
              {
                type: 'scatter',
                mode: 'markers+text',
                x: competitorsFiltered.map((c) => c.price),
                y: competitorsFiltered.map((c) => c.features),
                text: competitorsFiltered.map((c) => c.name),
                textposition: 'top center',
                marker: {
                  size: competitorsFiltered.map((c) => 40 + c.share * 160),
                  color: competitorsFiltered.map((c) => (c.momentum === 'up' ? '#34D399' : c.momentum === 'down' ? '#F87171' : '#6EE7FF')),
                  opacity: 0.9,
                  line: { width: 1, color: 'rgba(255,255,255,0.3)' }
                }
              }
            ]}
            layout={{
              autosize: true,
              height: 360,
              margin: { l: 50, r: 10, t: 10, b: 40 },
              paper_bgcolor: 'rgba(0,0,0,0)',
              plot_bgcolor: 'rgba(0,0,0,0)',
              xaxis: { title: 'relative price', range: [0.25, 0.85] },
              yaxis: { title: 'feature depth', range: [0.45, 0.9] }
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: '100%' }}
          />
        </WidgetCard>
      )
    },
    {
      id: 'share',
      minH: 4,
      default: { x: 7, y: 0, w: 5, h: 5 },
      render: () => (
        <WidgetCard title="Market share" subtitle="Estimated distribution">
          <Plot
            data={[
              {
                type: 'pie',
                labels: competitorsFiltered.map((c) => c.name),
                values: competitorsFiltered.map((c) => c.share),
                hole: 0.62
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
      id: 'matrix',
      minH: 4,
      default: { x: 0, y: 5, w: 12, h: 5 },
      render: () => (
        <WidgetCard title="Feature matrix" subtitle="Capabilities snapshot">
          <Box sx={{ overflowX: 'auto' }}>
            <Table size="small" aria-label="Feature matrix">
              <TableHead>
                <TableRow>
                  <TableCell sx={{ fontWeight: 800 }}>Feature</TableCell>
                  {COMPETITORS.map((c) => (
                    <TableCell key={c.name} sx={{ fontWeight: 800 }}>
                      {c.name}
                    </TableCell>
                  ))}
                </TableRow>
              </TableHead>
              <TableBody>
                {FEATURE_MATRIX.filter((r) => r.feature.toLowerCase().includes(query.trim().toLowerCase())).map((row) => (
                  <TableRow key={row.feature} hover>
                    <TableCell>
                      <Typography variant="body2" sx={{ fontWeight: 650 }}>
                        {row.feature}
                      </Typography>
                    </TableCell>
                    {COMPETITORS.map((c) => (
                      <TableCell key={c.name}>
                        <Chip
                          size="small"
                          label={(row as Record<string, unknown>)[c.name] ? 'Yes' : 'No'}
                          color={(row as Record<string, unknown>)[c.name] ? 'success' : 'default'}
                          variant={(row as Record<string, unknown>)[c.name] ? 'filled' : 'outlined'}
                        />
                      </TableCell>
                    ))}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Box>
        </WidgetCard>
      )
    },
    {
      id: 'network',
      minH: 4,
      default: { x: 0, y: 10, w: 12, h: 4 },
      render: () => {
        const nodes = competitorsFiltered.length > 1 ? competitorsFiltered : COMPETITORS;
        const n = nodes.length;
        const coords = nodes.map((c, i) => {
          const a = (i / n) * Math.PI * 2;
          return { name: c.name, x: Math.cos(a), y: Math.sin(a) };
        });

        const edges = coords.flatMap((p, i) => {
          const q = coords[(i + 1) % n];
          return [p, q, null];
        });

        return (
          <WidgetCard title="Landscape network" subtitle="Relationship map (placeholder scaffold)">
            <Plot
              data={[
                {
                  type: 'scatter',
                  mode: 'lines',
                  x: edges.map((p) => (p ? p.x : null)),
                  y: edges.map((p) => (p ? p.y : null)),
                  line: { color: 'rgba(110,231,255,0.35)', width: 2 },
                  hoverinfo: 'skip'
                },
                {
                  type: 'scatter',
                  mode: 'markers+text',
                  x: coords.map((p) => p.x),
                  y: coords.map((p) => p.y),
                  text: coords.map((p) => p.name),
                  textposition: 'bottom center',
                  marker: { size: 18, color: '#6EE7FF' }
                }
              ]}
              layout={{
                autosize: true,
                height: 300,
                margin: { l: 10, r: 10, t: 10, b: 10 },
                xaxis: { visible: false },
                yaxis: { visible: false },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)'
              }}
              config={{ displayModeBar: false, responsive: true }}
              style={{ width: '100%' }}
            />
          </WidgetCard>
        );
      }
    }
  ];

  return (
    <Box ref={containerRef}>
      <Stack direction={{ xs: 'column', sm: 'row' }} alignItems={{ sm: 'flex-end' }} justifyContent="space-between" gap={1}>
        <PageHeader title="Competitive intelligence" subtitle="Positioning, market share, and feature matrices." />
        <Stack direction="row" spacing={1} alignItems="center">
          <DashboardActions
            layoutStorageKey={layoutKey}
            widgetList={widgetList}
            hiddenIds={hiddenIds}
            setHiddenIds={setHiddenIds}
            onResetLayout={resetLayout}
          />
          <ExportPdfButton targetRef={containerRef} fileName="competitive-intelligence.pdf" />
        </Stack>
      </Stack>

      <DashboardGrid key={layoutNonce} widgets={widgets.filter((w) => !hiddenIds.includes(w.id))} storageKey={layoutKey} />
    </Box>
  );
}
