import React from 'react';
import { Button, Stack } from '@mui/material';
import TuneIcon from '@mui/icons-material/Tune';
import RestartAltIcon from '@mui/icons-material/RestartAlt';
import { ManageWidgetsDialog } from '@/components/ManageWidgetsDialog';

export function DashboardActions({
  layoutStorageKey,
  widgetList,
  hiddenIds,
  setHiddenIds,
  onResetLayout
}: {
  layoutStorageKey: string;
  widgetList: { id: string; title: string }[];
  hiddenIds: string[];
  setHiddenIds: (next: string[]) => void;
  onResetLayout?: () => void;
}) {
  const [open, setOpen] = React.useState(false);

  return (
    <Stack direction="row" spacing={1} alignItems="center">
      <Button
        variant="outlined"
        size="small"
        startIcon={<RestartAltIcon />}
        onClick={() => {
          if (onResetLayout) {
            onResetLayout();
            return;
          }
          localStorage.removeItem(layoutStorageKey);
        }}
      >
        Reset layout
      </Button>
      <Button variant="outlined" size="small" startIcon={<TuneIcon />} onClick={() => setOpen(true)}>
        Widgets
      </Button>
      <ManageWidgetsDialog
        open={open}
        onClose={() => setOpen(false)}
        widgets={widgetList}
        hiddenIds={hiddenIds}
        setHiddenIds={setHiddenIds}
      />
    </Stack>
  );
}
