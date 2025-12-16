import React, { PropsWithChildren } from 'react';
import {
  AppBar,
  Box,
  Divider,
  Drawer,
  IconButton,
  List,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Menu,
  MenuItem,
  Toolbar,
  Tooltip,
  Typography,
  useMediaQuery
} from '@mui/material';
import { useTheme } from '@mui/material/styles';
import MenuIcon from '@mui/icons-material/Menu';
import DashboardIcon from '@mui/icons-material/Dashboard';
import InsightsIcon from '@mui/icons-material/Insights';
import MonetizationOnIcon from '@mui/icons-material/MonetizationOn';
import VisibilityIcon from '@mui/icons-material/Visibility';
import GroupIcon from '@mui/icons-material/Group';
import RecommendIcon from '@mui/icons-material/Recommend';
import SettingsIcon from '@mui/icons-material/Settings';
import PersonIcon from '@mui/icons-material/Person';
import DarkModeIcon from '@mui/icons-material/DarkMode';
import LightModeIcon from '@mui/icons-material/LightMode';
import TranslateIcon from '@mui/icons-material/Translate';

import { NavLink, useLocation, useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';

import { useAuth } from '@/app/state/auth';
import { usePreferences } from '@/app/state/preferences';
import { GlobalSearch } from '@/components/GlobalSearch';
import { LiveStatusChip } from '@/components/LiveStatusChip';
import { OnboardingDialog } from '@/components/OnboardingDialog';

const DRAWER_WIDTH = 280;

function NavItem({
  to,
  label,
  icon,
  disabled
}: {
  to: string;
  label: string;
  icon: React.ReactNode;
  disabled?: boolean;
}) {
  const location = useLocation();
  const selected = location.pathname === to;

  return (
    <ListItemButton component={NavLink} to={to} selected={selected} disabled={disabled}>
      <ListItemIcon>{icon}</ListItemIcon>
      <ListItemText primary={label} />
    </ListItemButton>
  );
}

export function AppShell({ children }: PropsWithChildren) {
  const theme = useTheme();
  const isDesktop = useMediaQuery(theme.breakpoints.up('md'));
  const [mobileOpen, setMobileOpen] = React.useState(false);

  const { t, i18n } = useTranslation();
  const { user, logout } = useAuth();
  const { mode, setMode, language, setLanguage } = usePreferences();
  const navigate = useNavigate();

  const [anchorEl, setAnchorEl] = React.useState<null | HTMLElement>(null);
  const userMenuOpen = Boolean(anchorEl);

  const closeMobile = () => setMobileOpen(false);

  const drawer = (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Box sx={{ px: 2, py: 2 }}>
        <Typography variant="h6" sx={{ fontWeight: 700, letterSpacing: 0.2 }}>
          AI Business Assistant
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Dashboards & workflows
        </Typography>
      </Box>
      <Divider />
      <List sx={{ flex: 1 }}>
        <NavItem to="/" label={t('nav.home')} icon={<DashboardIcon />} />
        <NavItem
          to="/dashboards/market"
          label={t('nav.market')}
          icon={<InsightsIcon />}
          disabled={!user || !['executive', 'analyst', 'admin'].includes(user.role)}
        />
        <NavItem
          to="/dashboards/finance"
          label={t('nav.finance')}
          icon={<MonetizationOnIcon />}
          disabled={!user || !['executive', 'analyst', 'admin'].includes(user.role)}
        />
        <NavItem
          to="/dashboards/competitive"
          label={t('nav.competitive')}
          icon={<VisibilityIcon />}
          disabled={!user || !['analyst', 'admin'].includes(user.role)}
        />
        <NavItem
          to="/dashboards/customer"
          label={t('nav.customer')}
          icon={<GroupIcon />}
          disabled={!user || !['executive', 'analyst', 'stakeholder', 'admin'].includes(user.role)}
        />
        <NavItem
          to="/recommendations"
          label={t('nav.recommendations')}
          icon={<RecommendIcon />}
          disabled={!user || !['executive', 'analyst', 'stakeholder', 'admin'].includes(user.role)}
        />
      </List>
      <Divider />
      <List>
        <NavItem to="/settings" label={t('nav.settings')} icon={<SettingsIcon />} />
        <NavItem to="/profile" label={t('nav.profile')} icon={<PersonIcon />} />
      </List>
    </Box>
  );

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh' }}>
      <AppBar position="fixed" color="transparent" elevation={0} sx={{ borderBottom: '1px solid', borderColor: 'divider' }}>
        <Toolbar sx={{ gap: 1.5 }}>
          {!isDesktop && (
            <IconButton aria-label="Open navigation" onClick={() => setMobileOpen(true)}>
              <MenuIcon />
            </IconButton>
          )}

          <Box sx={{ display: { xs: 'none', sm: 'block' } }}>
            <LiveStatusChip />
          </Box>

          <Box sx={{ flex: 1, maxWidth: 820 }}>
            <GlobalSearch />
          </Box>

          <Tooltip title={mode === 'dark' ? 'Light mode' : 'Dark mode'}>
            <IconButton aria-label="Toggle theme" onClick={() => setMode(mode === 'dark' ? 'light' : 'dark')}>
              {mode === 'dark' ? <LightModeIcon /> : <DarkModeIcon />}
            </IconButton>
          </Tooltip>

          <Tooltip title="Language">
            <IconButton
              aria-label="Change language"
              onClick={() => {
                const next = language === 'en' ? 'es' : 'en';
                setLanguage(next);
                void i18n.changeLanguage(next);
              }}
            >
              <TranslateIcon />
            </IconButton>
          </Tooltip>

          <Tooltip title={user ? `${user.name} (${user.role})` : 'User'}>
            <IconButton aria-label="User menu" onClick={(e) => setAnchorEl(e.currentTarget)}>
              <PersonIcon />
            </IconButton>
          </Tooltip>
          <Menu anchorEl={anchorEl} open={userMenuOpen} onClose={() => setAnchorEl(null)}>
            <MenuItem
              onClick={() => {
                setAnchorEl(null);
                navigate('/profile');
              }}
            >
              Profile
            </MenuItem>
            <MenuItem
              onClick={() => {
                setAnchorEl(null);
                navigate('/settings');
              }}
            >
              Settings
            </MenuItem>
            <Divider />
            <MenuItem
              onClick={() => {
                setAnchorEl(null);
                logout();
                navigate('/login');
              }}
            >
              {t('auth.signOut')}
            </MenuItem>
          </Menu>
        </Toolbar>
      </AppBar>

      <Box
        component="nav"
        sx={{ width: { md: DRAWER_WIDTH }, flexShrink: { md: 0 } }}
        aria-label="Navigation"
      >
        <Drawer
          variant={isDesktop ? 'permanent' : 'temporary'}
          open={isDesktop ? true : mobileOpen}
          onClose={closeMobile}
          ModalProps={{ keepMounted: true }}
          sx={{
            '& .MuiDrawer-paper': {
              width: DRAWER_WIDTH,
              boxSizing: 'border-box'
            }
          }}
        >
          {drawer}
        </Drawer>
      </Box>

      <Box component="main" sx={{ flexGrow: 1, px: { xs: 2, md: 3 }, pb: 4 }}>
        <Toolbar />
        <OnboardingDialog />
        {children}
      </Box>
    </Box>
  );
}
