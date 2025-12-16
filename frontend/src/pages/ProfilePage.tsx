import React from 'react';
import { Box, Button, Card, CardContent, FormControl, InputLabel, MenuItem, Select, Stack, TextField } from '@mui/material';
import { PageHeader } from '@/components/PageHeader';
import { useAuth } from '@/app/state/auth';
import type { UserRole } from '@/types/auth';

export default function ProfilePage() {
  const { user, updateProfile } = useAuth();
  const [name, setName] = React.useState(user?.name ?? '');
  const [email, setEmail] = React.useState(user?.email ?? '');
  const [role, setRole] = React.useState<UserRole>(user?.role ?? 'analyst');

  if (!user) return null;

  return (
    <Box>
      <PageHeader title="Profile" subtitle="Manage your identity and role-based views." />

      <Card>
        <CardContent>
          <Stack spacing={2} maxWidth={520}>
            <TextField label="Name" value={name} onChange={(e) => setName(e.target.value)} />
            <TextField label="Email" value={email} onChange={(e) => setEmail(e.target.value)} />

            <FormControl fullWidth>
              <InputLabel id="role">Role</InputLabel>
              <Select labelId="role" label="Role" value={role} onChange={(e) => setRole(e.target.value as UserRole)}>
                <MenuItem value="executive">Executive</MenuItem>
                <MenuItem value="analyst">Analyst</MenuItem>
                <MenuItem value="stakeholder">Stakeholder</MenuItem>
                <MenuItem value="admin">Admin</MenuItem>
              </Select>
            </FormControl>

            <Button
              variant="contained"
              onClick={() => {
                updateProfile({ name, email, role });
              }}
            >
              Save profile
            </Button>
          </Stack>
        </CardContent>
      </Card>
    </Box>
  );
}
