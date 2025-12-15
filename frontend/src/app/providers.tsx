import React, { PropsWithChildren } from 'react';
import { ThemeProvider } from '@mui/material/styles';
import { PreferencesProvider, usePreferences } from '@/app/state/preferences';
import { AuthProvider } from '@/app/state/auth';
import { LiveProvider } from '@/app/state/live';
import { SearchProvider } from '@/app/state/search';
import { createAppTheme } from '@/theme/theme';

function ThemedProviders({ children }: PropsWithChildren) {
  const { mode } = usePreferences();
  const theme = React.useMemo(() => createAppTheme(mode), [mode]);

  return <ThemeProvider theme={theme}>{children}</ThemeProvider>;
}

export function AppProviders({ children }: PropsWithChildren) {
  return (
    <PreferencesProvider>
      <ThemedProviders>
        <AuthProvider>
          <LiveProvider>
            <SearchProvider>{children}</SearchProvider>
          </LiveProvider>
        </AuthProvider>
      </ThemedProviders>
    </PreferencesProvider>
  );
}
