import { createTheme, responsiveFontSizes } from '@mui/material/styles';
import type { Theme } from '@mui/material/styles';
import type { ThemeMode } from '@/app/state/preferences';

export function createAppTheme(mode: ThemeMode): Theme {
  const theme = createTheme({
    palette: {
      mode,
      primary: {
        main: '#6EE7FF'
      },
      secondary: {
        main: '#B794F4'
      },
      background: {
        default: mode === 'dark' ? '#0B1220' : '#F7FAFC',
        paper: mode === 'dark' ? '#0F1B33' : '#FFFFFF'
      },
      success: { main: '#34D399' },
      warning: { main: '#F59E0B' },
      error: { main: '#F87171' }
    },
    shape: {
      borderRadius: 12
    },
    typography: {
      fontFamily:
        "ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, 'Apple Color Emoji', 'Segoe UI Emoji'"
    },
    components: {
      MuiCard: {
        styleOverrides: {
          root: {
            border: '1px solid rgba(255,255,255,0.08)'
          }
        }
      },
      MuiButton: {
        defaultProps: {
          disableElevation: true
        }
      }
    }
  });

  return responsiveFontSizes(theme);
}
