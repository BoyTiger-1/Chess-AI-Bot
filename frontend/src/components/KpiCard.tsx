import React from 'react';
import { Box, Typography } from '@mui/material';
import { WidgetCard } from '@/components/WidgetCard';

export function KpiCard({
  title,
  value,
  delta,
  footnote
}: {
  title: string;
  value: string;
  delta?: { label: string; tone: 'positive' | 'negative' | 'neutral' };
  footnote?: string;
}) {
  return (
    <WidgetCard title={title}>
      <Box sx={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', gap: 2 }}>
        <Typography variant="h4" sx={{ fontWeight: 800, letterSpacing: -0.4 }}>
          {value}
        </Typography>
        {delta && (
          <Typography
            variant="body2"
            sx={{
              fontWeight: 700,
              color:
                delta.tone === 'positive'
                  ? 'success.main'
                  : delta.tone === 'negative'
                    ? 'error.main'
                    : 'text.secondary'
            }}
          >
            {delta.label}
          </Typography>
        )}
      </Box>
      {footnote && (
        <Typography variant="caption" color="text.secondary">
          {footnote}
        </Typography>
      )}
    </WidgetCard>
  );
}
