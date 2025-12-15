import React from 'react';
import { Chip } from '@mui/material';
import WifiTetheringIcon from '@mui/icons-material/WifiTethering';
import WifiOffIcon from '@mui/icons-material/WifiOff';
import { useTranslation } from 'react-i18next';
import { useLive } from '@/app/state/live';

export function LiveStatusChip() {
  const { status } = useLive();
  const { t } = useTranslation();

  if (status === 'connected') {
    return <Chip icon={<WifiTetheringIcon />} size="small" color="success" label={t('common.live')} />;
  }

  return <Chip icon={<WifiOffIcon />} size="small" variant="outlined" label={t('common.disconnected')} />;
}
