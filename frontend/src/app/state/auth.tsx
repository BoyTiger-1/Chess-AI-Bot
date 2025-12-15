import React, { createContext, PropsWithChildren, useCallback, useContext, useMemo, useState } from 'react';
import { readJson, writeJson } from '@/utils/storage';
import type { UserProfile, UserRole } from '@/types/auth';

interface AuthState {
  user: UserProfile | null;
}

interface AuthContextValue extends AuthState {
  login: (input: { email: string; password: string; role: UserRole }) => Promise<void>;
  logout: () => void;
  updateProfile: (patch: Partial<Pick<UserProfile, 'name' | 'email' | 'role'>>) => void;
}

const STORAGE_KEY = 'aiba.auth.v1';

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: PropsWithChildren) {
  const [state, setState] = useState<AuthState>(() => readJson<AuthState>(STORAGE_KEY, { user: null }));

  const login = useCallback(async ({ email, role }: { email: string; password: string; role: UserRole }) => {
    const name = email.split('@')[0] || 'User';
    const user: UserProfile = {
      id: crypto.randomUUID(),
      name,
      email,
      role
    };

    const next: AuthState = { user };
    setState(next);
    writeJson(STORAGE_KEY, next);
  }, []);

  const logout = useCallback(() => {
    const next: AuthState = { user: null };
    setState(next);
    writeJson(STORAGE_KEY, next);
  }, []);

  const updateProfile = useCallback(
    (patch: Partial<Pick<UserProfile, 'name' | 'email' | 'role'>>) => {
      if (!state.user) return;
      const next: AuthState = { user: { ...state.user, ...patch } };
      setState(next);
      writeJson(STORAGE_KEY, next);
    },
    [state.user]
  );

  const value: AuthContextValue = useMemo(
    () => ({
      ...state,
      login,
      logout,
      updateProfile
    }),
    [state, login, logout, updateProfile]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
}
