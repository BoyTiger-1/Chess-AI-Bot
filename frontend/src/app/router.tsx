import React, { Suspense, lazy } from 'react';
import { BrowserRouter, Navigate, Outlet, Route, Routes } from 'react-router-dom';
import { RequireAuth, RequireRole } from '@/app/routeGuards';
import { AppShell } from '@/layouts/AppShell';

const LoginPage = lazy(() => import('@/pages/LoginPage'));
const HomePage = lazy(() => import('@/pages/HomePage'));
const ProfilePage = lazy(() => import('@/pages/ProfilePage'));
const SettingsPage = lazy(() => import('@/pages/SettingsPage'));

const MarketAnalysisDashboard = lazy(() => import('@/dashboards/MarketAnalysisDashboard'));
const FinancialForecastingDashboard = lazy(() => import('@/dashboards/FinancialForecastingDashboard'));
const CompetitiveIntelligenceDashboard = lazy(() => import('@/dashboards/CompetitiveIntelligenceDashboard'));
const CustomerInsightsDashboard = lazy(() => import('@/dashboards/CustomerInsightsDashboard'));
const RecommendationCenter = lazy(() => import('@/dashboards/RecommendationCenter'));

function AppFrame() {
  return (
    <RequireAuth>
      <AppShell>
        <Outlet />
      </AppShell>
    </RequireAuth>
  );
}

export function AppRouter() {
  return (
    <BrowserRouter>
      <Suspense fallback={<div style={{ padding: 24 }}>Loading…</div>}>
        <Routes>
          <Route path="/login" element={<LoginPage />} />

          <Route element={<AppFrame />}>
            <Route index element={<HomePage />} />
            <Route path="/profile" element={<ProfilePage />} />
            <Route path="/settings" element={<SettingsPage />} />

            <Route
              path="/dashboards/market"
              element={
                <RequireRole roles={['executive', 'analyst', 'admin']}>
                  <MarketAnalysisDashboard />
                </RequireRole>
              }
            />
            <Route
              path="/dashboards/finance"
              element={
                <RequireRole roles={['executive', 'analyst', 'admin']}>
                  <FinancialForecastingDashboard />
                </RequireRole>
              }
            />
            <Route
              path="/dashboards/competitive"
              element={
                <RequireRole roles={['analyst', 'admin']}>
                  <CompetitiveIntelligenceDashboard />
                </RequireRole>
              }
            />
            <Route
              path="/dashboards/customer"
              element={
                <RequireRole roles={['executive', 'analyst', 'stakeholder', 'admin']}>
                  <CustomerInsightsDashboard />
                </RequireRole>
              }
            />
            <Route
              path="/recommendations"
              element={
                <RequireRole roles={['executive', 'analyst', 'stakeholder', 'admin']}>
                  <RecommendationCenter />
                </RequireRole>
              }
            />

            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </Suspense>
    </BrowserRouter>
  );
}
