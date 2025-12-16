import React from 'react';
import {
  Button,
  Checkbox,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  FormControlLabel,
  Stack,
  Typography
} from '@mui/material';

export function ManageWidgetsDialog({
  open,
  onClose,
  widgets,
  hiddenIds,
  setHiddenIds
}: {
  open: boolean;
  onClose: () => void;
  widgets: { id: string; title: string }[];
  hiddenIds: string[];
  setHiddenIds: (next: string[]) => void;
}) {
  const toggle = (id: string) => {
    if (hiddenIds.includes(id)) setHiddenIds(hiddenIds.filter((x) => x !== id));
    else setHiddenIds([...hiddenIds, id]);
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>Manage widgets</DialogTitle>
      <DialogContent>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
          Show or hide widgets. Layout changes are saved automatically.
        </Typography>
        <Stack>
          {widgets.map((w) => (
            <FormControlLabel
              key={w.id}
              control={<Checkbox checked={!hiddenIds.includes(w.id)} onChange={() => toggle(w.id)} />}
              label={w.title}
            />
          ))}
        </Stack>
      </DialogContent>
      <DialogActions>
        <Button onClick={() => setHiddenIds([])}>Show all</Button>
        <Button variant="contained" onClick={onClose}>
          Done
        </Button>
      </DialogActions>
    </Dialog>
  );
}
