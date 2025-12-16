import type { Preview } from '@storybook/react';
import React from 'react';
import { CssBaseline } from '@mui/material';
import { ThemeProvider } from '@mui/material/styles';
import { createAppTheme } from '../src/theme/theme';

const preview: Preview = {
  decorators: [
    (Story) => (
      <ThemeProvider theme={createAppTheme('dark')}>
        <CssBaseline />
        <div style={{ padding: 24 }}>
          <Story />
        </div>
      </ThemeProvider>
    )
  ],
  parameters: {
    layout: 'fullscreen'
  }
};

export default preview;
