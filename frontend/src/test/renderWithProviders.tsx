import React, { PropsWithChildren } from 'react';
import { render, type RenderOptions } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { CssBaseline } from '@mui/material';
import { AppProviders } from '@/app/providers';

function Wrapper({ children }: PropsWithChildren) {
  return (
    <MemoryRouter>
      <AppProviders>
        <CssBaseline />
        {children}
      </AppProviders>
    </MemoryRouter>
  );
}

export function renderWithProviders(ui: React.ReactElement, options?: Omit<RenderOptions, 'wrapper'>) {
  return render(ui, { wrapper: Wrapper, ...options });
}
