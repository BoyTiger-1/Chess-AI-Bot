import React from 'react';
import ReactDOM from 'react-dom/client';
import { CssBaseline } from '@mui/material';
import 'react-grid-layout/css/styles.css';
import 'react-resizable/css/styles.css';
import { AppProviders } from '@/app/providers';
import { AppRouter } from '@/app/router';
import '@/i18n';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <AppProviders>
      <CssBaseline />
      <AppRouter />
    </AppProviders>
  </React.StrictMode>
);
