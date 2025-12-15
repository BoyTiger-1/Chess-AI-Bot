import React from 'react';
import { Box, Card, CardContent, Chip, Divider, Stack, Typography } from '@mui/material';
import { useTranslation } from 'react-i18next';
import { useAuth } from '@/app/state/auth';
import { useLive } from '@/app/state/live';
import { useSearch } from '@/app/state/search';
import { PageHeader } from '@/components/PageHeader';

export default function HomePage() {
  const { t } = useTranslation();
  const { user } = useAuth();
  const { status, lastMessage } = useLive();
  const { savedFilters, setQuery } = useSearch();

  return (
    <Box>
      <PageHeader title={t('nav.home')} subtitle={user ? `Welcome back, ${user.name}.` : undefined} />

      <Stack direction={{ xs: 'column', lg: 'row' }} spacing={2}>
        <Card sx={{ flex: 1 }}>
          <CardContent>
            <Typography variant="subtitle1" sx={{ fontWeight: 750, mb: 1 }}>
              Live updates
            </Typography>
            <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 2 }}>
              <Chip label={status} size="small" variant="outlined" />
              {lastMessage && <Chip label={lastMessage.type} size="small" color="primary" variant="outlined" />}
            </Stack>
            {lastMessage ? (
              <Typography
                component="pre"
                sx={{
                  fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace',
                  fontSize: 12,
                  whiteSpace: 'pre-wrap',
                  m: 0,
                  opacity: 0.9
                }}
              >
                {JSON.stringify(lastMessage, null, 2)}
              </Typography>
            ) : (
              <Typography color="text.secondary">Connect a backend WebSocket stream to see live data here.</Typography>
            )}
          </CardContent>
        </Card>

        <Card sx={{ flex: 1 }}>
          <CardContent>
            <Typography variant="subtitle1" sx={{ fontWeight: 750, mb: 1 }}>
              Saved filters
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              One-click recall for common queries across modules.
            </Typography>

            <Divider sx={{ mb: 2 }} />

            <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
              {savedFilters.length === 0 && <Typography color="text.secondary">No filters saved yet.</Typography>}
              {savedFilters.slice(0, 8).map((f) => (
                <Chip
                  key={f.id}
                  label={f.name}
                  variant="outlined"
                  onClick={() => setQuery(f.query)}
                  sx={{ mb: 1 }}
                />
              ))}
            </Stack>
          </CardContent>
        </Card>
      </Stack>
    </Box>
  );
}
