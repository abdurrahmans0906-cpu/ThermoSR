import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { DashboardLayout } from './layouts/DashboardLayout';
import { Dashboard } from './pages/Dashboard';
import { Reconstruct } from './pages/Reconstruct';
import { DatasetPage } from './pages/DatasetPage';
import { TrainingLab } from './pages/TrainingLab';
import { Experiments } from './pages/Experiments';
import { ModelRegistry } from './pages/ModelRegistry';
import { SettingsPage } from './pages/SettingsPage';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});

export const App: React.FC = () => {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<DashboardLayout />}>
            <Route index element={<Dashboard />} />
            <Route path="reconstruct" element={<Reconstruct />} />
            <Route path="dataset" element={<DatasetPage />} />
            <Route path="training" element={<TrainingLab />} />
            <Route path="experiments" element={<Experiments />} />
            <Route path="models" element={<ModelRegistry />} />
            <Route path="settings" element={<SettingsPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
};

export default App;
