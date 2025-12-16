import React from 'react';
import {
  Box,
  Card,
  CardContent,
  FormControlLabel,
  Stack,
  Switch,
  ToggleButton,
  ToggleButtonGroup,
  Typography
} from '@mui/material';
import { useTranslation } from 'react-i18next';
import { usePreferences } from '@/app/state/preferences';
import { PageHeader } from '@/components/PageHeader';

export default function SettingsPage() {
  const { t, i18n } = useTranslation();
  const { mode, setMode, language, setLanguage, notifications, setNotifications } = usePreferences();

  return (
    <Box>
      <PageHeader title={t('nav.settings')} subtitle="Preferences for dashboards, notifications, and export behavior." />

      <Stack spacing={2}>
        <Card>
          <CardContent>
            <Typography variant="subtitle1" sx={{ fontWeight: 750, mb: 1 }}>
              Appearance
            </Typography>
            <ToggleButtonGroup
              exclusive
              value={mode}
              onChange={(_, next) => {
                if (!next) return;
                setMode(next);
              }}
              size="small"
            >
              <ToggleButton value="dark">Dark</ToggleButton>
              <ToggleButton value="light">Light</ToggleButton>
            </ToggleButtonGroup>
          </CardContent>
        </Card>

        <Card>
          <CardContent>
            <Typography variant="subtitle1" sx={{ fontWeight: 750, mb: 1 }}>
              Language
            </Typography>
            <ToggleButtonGroup
              exclusive
              value={language}
              onChange={(_, next) => {
                if (!next) return;
                setLanguage(next);
                void i18n.changeLanguage(next);
              }}
              size="small"
            >
              <ToggleButton value="en">English</ToggleButton>
              <ToggleButton value="es">Español</ToggleButton>
            </ToggleButtonGroup>
          </CardContent>
        </Card>

        <Card>
          <CardContent>
            <Typography variant="subtitle1" sx={{ fontWeight: 750, mb: 1 }}>
              Notifications
            </Typography>
            <Stack spacing={1}>
              <FormControlLabel
                control={
                  <Switch
                    checked={notifications.liveUpdates}
                    onChange={(e) => setNotifications({ ...notifications, liveUpdates: e.target.checked })}
                  />
                }
                label="Live updates"
              />
              <FormControlLabel
                control={
                  <Switch
                    checked={notifications.weeklyDigest}
                    onChange={(e) => setNotifications({ ...notifications, weeklyDigest: e.target.checked })}
                  />
                }
                label="Weekly digest"
              />
            </Stack>
          </CardContent>
        </Card>
      </Stack>
    </Box>
  );
}
