import React, { PropsWithChildren } from 'react';
import { Box, Card, CardContent, CardHeader, IconButton, Tooltip } from '@mui/material';
import DragIndicatorIcon from '@mui/icons-material/DragIndicator';

export function WidgetCard({
  title,
  subtitle,
  actions,
  children
}: PropsWithChildren<{ title: string; subtitle?: string; actions?: React.ReactNode }>) {
  return (
    <Card sx={{ height: '100%' }}>
      <CardHeader
        title={title}
        subheader={subtitle}
        titleTypographyProps={{ variant: 'subtitle1', sx: { fontWeight: 700 } }}
        subheaderTypographyProps={{ variant: 'caption' }}
        action={
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            {actions}
            <Tooltip title="Drag">
              <IconButton size="small" className="drag-handle" aria-label="Drag widget">
                <DragIndicatorIcon fontSize="small" />
              </IconButton>
            </Tooltip>
          </Box>
        }
      />
      <CardContent sx={{ pt: 0 }}>{children}</CardContent>
    </Card>
  );
}
