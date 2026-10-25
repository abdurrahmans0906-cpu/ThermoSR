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

interface ErrorBoundaryProps {
  children: React.ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('Unhandled React Error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-slate-50 flex items-center justify-center p-6 text-slate-800 font-sans">
          <div className="max-w-lg w-full bg-white border border-rose-200 rounded-lg p-6 shadow-sm space-y-4">
            <div className="flex items-center gap-3 text-rose-600">
              <span className="text-xl">⚠️</span>
              <h2 className="text-base font-bold">Something went wrong</h2>
            </div>
            <p className="text-xs text-slate-600 leading-relaxed">
              An unexpected render error occurred in the dashboard:
            </p>
            <div className="p-3 bg-slate-100 rounded border border-slate-200 text-xs font-mono text-rose-700 overflow-x-auto">
              {this.state.error?.message || 'Unknown error'}
            </div>
            <button
              onClick={() => window.location.reload()}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded text-xs font-medium transition-colors"
            >
              Reload Page
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

export const App: React.FC = () => {
  return (
    <ErrorBoundary>
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
    </ErrorBoundary>
  );
};

export default App;
