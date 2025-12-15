import React from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  Container,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  TextField,
  Typography
} from '@mui/material';
import { useLocation, useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { useAuth } from '@/app/state/auth';
import type { UserRole } from '@/types/auth';

export default function LoginPage() {
  const { t } = useTranslation();
  const { login, user } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  const [email, setEmail] = React.useState('analyst@company.com');
  const [password, setPassword] = React.useState('');
  const [role, setRole] = React.useState<UserRole>('analyst');
  const [busy, setBusy] = React.useState(false);

  React.useEffect(() => {
    if (user) navigate('/');
  }, [user, navigate]);

  return (
    <Container maxWidth="sm" sx={{ py: 10 }}>
      <Typography variant="h3" sx={{ fontWeight: 850, letterSpacing: -0.5, mb: 1 }}>
        AI Business Assistant
      </Typography>
      <Typography color="text.secondary" sx={{ mb: 3 }}>
        Sign in to access dashboards, workflows, and recommendations.
      </Typography>

      <Card>
        <CardContent>
          <Box
            component="form"
            onSubmit={async (e) => {
              e.preventDefault();
              setBusy(true);
              try {
                await login({ email, password, role });
                const from = (location.state as { from?: string } | null)?.from;
                navigate(from || '/');
              } finally {
                setBusy(false);
              }
            }}
            sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}
          >
            <TextField
              label={t('auth.email')}
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              type="email"
              autoComplete="email"
              required
            />
            <TextField
              label={t('auth.password')}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              type="password"
              autoComplete="current-password"
              required
            />

            <FormControl fullWidth>
              <InputLabel id="role-label">{t('auth.role')}</InputLabel>
              <Select
                labelId="role-label"
                label={t('auth.role')}
                value={role}
                onChange={(e) => setRole(e.target.value as UserRole)}
              >
                <MenuItem value="executive">Executive</MenuItem>
                <MenuItem value="analyst">Analyst</MenuItem>
                <MenuItem value="stakeholder">Stakeholder</MenuItem>
                <MenuItem value="admin">Admin</MenuItem>
              </Select>
            </FormControl>

            <Button type="submit" variant="contained" size="large" disabled={busy}>
              {t('auth.signIn')}
            </Button>

            <Typography variant="caption" color="text.secondary">
              Demo auth only (UI scaffolding). Replace with real OAuth/SAML/SSO integration.
            </Typography>
          </Box>
        </CardContent>
      </Card>
    </Container>
  );
}
