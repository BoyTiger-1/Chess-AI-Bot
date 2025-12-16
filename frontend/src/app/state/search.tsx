import React, { createContext, PropsWithChildren, useContext, useMemo, useState } from 'react';
import { readJson, writeJson } from '@/utils/storage';

export interface SavedFilter {
  id: string;
  name: string;
  query: string;
  createdAt: string;
}

interface SearchContextValue {
  query: string;
  setQuery: (q: string) => void;
  savedFilters: SavedFilter[];
  saveFilter: (name: string, query: string) => void;
  deleteFilter: (id: string) => void;
}

const STORAGE_KEY = 'aiba.search.v1';

const SearchContext = createContext<SearchContextValue | null>(null);

export function SearchProvider({ children }: PropsWithChildren) {
  const [query, setQuery] = useState('');
  const [savedFilters, setSavedFilters] = useState<SavedFilter[]>(() => readJson<SavedFilter[]>(STORAGE_KEY, []));

  const value: SearchContextValue = useMemo(
    () => ({
      query,
      setQuery,
      savedFilters,
      saveFilter: (name, query) => {
        const next = [
          {
            id: crypto.randomUUID(),
            name,
            query,
            createdAt: new Date().toISOString()
          },
          ...savedFilters
        ];
        setSavedFilters(next);
        writeJson(STORAGE_KEY, next);
      },
      deleteFilter: (id) => {
        const next = savedFilters.filter((f) => f.id !== id);
        setSavedFilters(next);
        writeJson(STORAGE_KEY, next);
      }
    }),
    [query, savedFilters]
  );

  return <SearchContext.Provider value={value}>{children}</SearchContext.Provider>;
}

export function useSearch() {
  const ctx = useContext(SearchContext);
  if (!ctx) throw new Error('useSearch must be used within SearchProvider');
  return ctx;
}
