import React, { createContext, PropsWithChildren, useContext, useMemo, useState } from 'react';
import { readJson, writeJson } from '@/utils/storage';

export type ThemeMode = 'light' | 'dark';

export interface Preferences {
  mode: ThemeMode;
  language: 'en' | 'es';
  notifications: {
    liveUpdates: boolean;
    weeklyDigest: boolean;
  };
}

const STORAGE_KEY = 'aiba.preferences.v1';

const DEFAULTS: Preferences = {
  mode: 'dark',
  language: 'en',
  notifications: {
    liveUpdates: true,
    weeklyDigest: true
  }
};

interface PreferencesContextValue extends Preferences {
  setMode: (mode: ThemeMode) => void;
  setLanguage: (language: Preferences['language']) => void;
  setNotifications: (next: Preferences['notifications']) => void;
}

const PreferencesContext = createContext<PreferencesContextValue | null>(null);

export function PreferencesProvider({ children }: PropsWithChildren) {
  const [prefs, setPrefs] = useState<Preferences>(() => readJson(STORAGE_KEY, DEFAULTS));

  const value: PreferencesContextValue = useMemo(
    () => ({
      ...prefs,
      setMode: (mode) => {
        const next = { ...prefs, mode };
        setPrefs(next);
        writeJson(STORAGE_KEY, next);
      },
      setLanguage: (language) => {
        const next = { ...prefs, language };
        setPrefs(next);
        writeJson(STORAGE_KEY, next);
      },
      setNotifications: (notifications) => {
        const next = { ...prefs, notifications };
        setPrefs(next);
        writeJson(STORAGE_KEY, next);
      }
    }),
    [prefs]
  );

  return <PreferencesContext.Provider value={value}>{children}</PreferencesContext.Provider>;
}

export function usePreferences() {
  const ctx = useContext(PreferencesContext);
  if (!ctx) throw new Error('usePreferences must be used within PreferencesProvider');
  return ctx;
}
