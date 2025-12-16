import React from 'react';
import { describe, expect, it } from 'vitest';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import RecommendationCenter from '@/dashboards/RecommendationCenter';
import { renderWithProviders } from '@/test/renderWithProviders';
import { useSearch } from '@/app/state/search';
import { GlobalSearch } from '@/components/GlobalSearch';

function Harness() {
  const { setQuery } = useSearch();
  React.useEffect(() => {
    setQuery('');
  }, [setQuery]);
  return (
    <div>
      <GlobalSearch />
      <RecommendationCenter />
    </div>
  );
}

describe('RecommendationCenter', () => {
  it('filters recommendations by global search query', async () => {
    renderWithProviders(<Harness />);

    expect(screen.getByText(/mid-market bundle/i)).toBeInTheDocument();

    const input = screen.getByLabelText(/search/i);
    await userEvent.type(input, 'champion');

    expect(screen.getByText(/champion-change detection/i)).toBeInTheDocument();
    expect(screen.queryByText(/mid-market bundle/i)).not.toBeInTheDocument();
  });
});
