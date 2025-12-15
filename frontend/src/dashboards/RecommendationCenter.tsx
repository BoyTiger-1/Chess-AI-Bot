import React from 'react';
import {
  Box,
  Button,
  Card,
  CardActions,
  CardContent,
  Chip,
  Divider,
  Stack,
  Typography
} from '@mui/material';
import LaunchIcon from '@mui/icons-material/Launch';
import { PageHeader } from '@/components/PageHeader';
import { ExportPdfButton } from '@/components/ExportPdfButton';
import { useSearch } from '@/app/state/search';

interface Recommendation {
  id: string;
  title: string;
  category: 'pricing' | 'growth' | 'product' | 'risk';
  impact: 'low' | 'medium' | 'high';
  explanation: string;
  actions: string[];
}

const RECS: Recommendation[] = [
  {
    id: 'rec-1',
    title: 'Introduce a mid-market bundle with usage-based overages',
    category: 'pricing',
    impact: 'high',
    explanation:
      'Churn signals cluster around seat contractions and price sensitivity. A bundle with transparent overages reduces procurement friction and protects ARPA.',
    actions: ['Define bundle tiers', 'Pilot with 10 accounts', 'Measure NRR uplift']
  },
  {
    id: 'rec-2',
    title: 'Prioritize real-time competitive landscape mapping',
    category: 'product',
    impact: 'medium',
    explanation:
      'Competitive dashboard queries show high engagement. Adding network/map views improves decision latency for exec and analyst workflows.',
    actions: ['Add map widget', 'Integrate competitor feeds', 'Validate with analyst group']
  },
  {
    id: 'rec-3',
    title: 'Target expansion plays for high-growth segment',
    category: 'growth',
    impact: 'high',
    explanation:
      'High-growth accounts exhibit stable sentiment and rising usage. Expansion probability is highest when triggered within 7 days of a usage milestone.',
    actions: ['Create milestone triggers', 'Launch expansion sequences', 'Track win-rate by cohort']
  },
  {
    id: 'rec-4',
    title: 'Mitigate churn risk by automating champion-change detection',
    category: 'risk',
    impact: 'medium',
    explanation:
      'Champion departures correlate strongly with cancellations. Combine CRM ownership changes and email bounce events into the churn model to alert CSMs.',
    actions: ['Add CRM webhook', 'Tune churn model features', 'Create alert workflow']
  }
];

export default function RecommendationCenter() {
  const containerRef = React.useRef<HTMLDivElement>(null);
  const { query } = useSearch();

  const q = query.trim().toLowerCase();
  const filtered = !q
    ? RECS
    : RECS.filter((r) => `${r.title} ${r.category} ${r.impact} ${r.explanation} ${r.actions.join(' ')}`.toLowerCase().includes(q));

  return (
    <Box ref={containerRef}>
      <Stack direction={{ xs: 'column', sm: 'row' }} alignItems={{ sm: 'flex-end' }} justifyContent="space-between" gap={1}>
        <PageHeader title="Recommendation center" subtitle="Strategic recommendations with impact estimates and action items." />
        <ExportPdfButton targetRef={containerRef} fileName="recommendations.pdf" />
      </Stack>

      <Stack spacing={2}>
        {filtered.map((r) => (
          <Card key={r.id}>
            <CardContent>
              <Stack direction={{ xs: 'column', md: 'row' }} gap={1.5} alignItems={{ md: 'center' }} justifyContent="space-between">
                <Box>
                  <Typography variant="h6" sx={{ fontWeight: 800 }}>
                    {r.title}
                  </Typography>
                  <Typography color="text.secondary" sx={{ mt: 0.5 }}>
                    {r.explanation}
                  </Typography>
                </Box>
                <Stack direction="row" spacing={1} alignItems="center">
                  <Chip label={r.category} variant="outlined" />
                  <Chip
                    label={`impact: ${r.impact}`}
                    color={r.impact === 'high' ? 'success' : r.impact === 'medium' ? 'warning' : 'default'}
                  />
                </Stack>
              </Stack>

              <Divider sx={{ my: 2 }} />

              <Typography variant="subtitle2" sx={{ fontWeight: 800, mb: 1 }}>
                Action items
              </Typography>
              <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                {r.actions.map((a) => (
                  <Chip key={a} label={a} size="small" variant="outlined" />
                ))}
              </Stack>
            </CardContent>
            <CardActions sx={{ justifyContent: 'flex-end', px: 2, pb: 2 }}>
              <Button variant="contained" endIcon={<LaunchIcon />}>Create workflow</Button>
            </CardActions>
          </Card>
        ))}

        {filtered.length === 0 && (
          <Card>
            <CardContent>
              <Typography variant="body2" color="text.secondary">
                No recommendations match your search.
              </Typography>
            </CardContent>
          </Card>
        )}
      </Stack>
    </Box>
  );
}
