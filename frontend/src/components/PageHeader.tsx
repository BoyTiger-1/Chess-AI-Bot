import React from 'react';
import { Box, Breadcrumbs, Typography } from '@mui/material';

export function PageHeader({
  title,
  subtitle,
  breadcrumbs
}: {
  title: string;
  subtitle?: string;
  breadcrumbs?: string[];
}) {
  return (
    <Box sx={{ mb: 2 }}>
      {breadcrumbs && breadcrumbs.length > 0 && (
        <Breadcrumbs aria-label="breadcrumb" sx={{ mb: 0.5 }}>
          {breadcrumbs.map((b) => (
            <Typography key={b} variant="caption" color="text.secondary">
              {b}
            </Typography>
          ))}
        </Breadcrumbs>
      )}
      <Typography variant="h4" sx={{ fontWeight: 750, letterSpacing: -0.2 }}>
        {title}
      </Typography>
      {subtitle && (
        <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
          {subtitle}
        </Typography>
      )}
    </Box>
  );
}
