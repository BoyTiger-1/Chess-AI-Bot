import React from 'react';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';
import LoginPage from '@/pages/LoginPage';
import { renderWithProviders } from '@/test/renderWithProviders';

describe('LoginPage', () => {
  it('renders and allows typing', async () => {
    renderWithProviders(<LoginPage />);

    const email = screen.getByLabelText(/email/i);
    await userEvent.clear(email);
    await userEvent.type(email, 'exec@company.com');

    expect((email as HTMLInputElement).value).toBe('exec@company.com');
    expect(screen.getByRole('button', { name: /sign in/i })).toBeInTheDocument();
  });
});
