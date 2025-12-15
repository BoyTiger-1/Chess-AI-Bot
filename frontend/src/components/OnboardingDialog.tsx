import React from 'react';
import { Button, Dialog, DialogActions, DialogContent, DialogTitle, Typography } from '@mui/material';
import { useTranslation } from 'react-i18next';

const STORAGE_KEY = 'aiba.onboarding.v1';

export function OnboardingDialog() {
  const { t } = useTranslation();
  const [open, setOpen] = React.useState(() => {
    try {
      return localStorage.getItem(STORAGE_KEY) !== 'done';
    } catch {
      return true;
    }
  });

  const close = () => {
    setOpen(false);
    try {
      localStorage.setItem(STORAGE_KEY, 'done');
    } catch {
      // ignore
    }
  };

  return (
    <Dialog open={open} onClose={close} aria-labelledby="onboarding-title" maxWidth="sm" fullWidth>
      <DialogTitle id="onboarding-title">{t('onboarding.title')}</DialogTitle>
      <DialogContent>
        <Typography color="text.secondary">{t('onboarding.body')}</Typography>
      </DialogContent>
      <DialogActions>
        <Button onClick={close} variant="contained">
          {t('onboarding.cta')}
        </Button>
      </DialogActions>
    </Dialog>
  );
}
