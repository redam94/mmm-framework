import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import { AppShell } from './components/layout';
import { ProtectedRoute } from './components/common';
import {
  LoginPage,
  DashboardPage,
  PlanningPage,
  DataUploadPage,
  ModelConfigPage,
  ModelFitPage,
  DiagnosticsPage,
  ResultsPage,
} from './pages';
import './index.css';

// Create React Query client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 60000, // 1 minute
      retry: 2,
      refetchOnWindowFocus: false,
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          {/* Public route - Login */}
          <Route path="/login" element={<LoginPage />} />

          {/* Protected routes */}
          <Route
            path="/*"
            element={
              <ProtectedRoute>
                <AppShell>
                  <Routes>
                    {/* Dashboard as default */}
                    <Route path="/" element={<Navigate to="/dashboard" replace />} />
                    <Route path="/dashboard" element={<DashboardPage />} />

                    {/* Bayesian Workflow Pages */}
                    <Route path="/planning" element={<PlanningPage />} />
                    <Route path="/data" element={<DataUploadPage />} />
                    <Route path="/config" element={<ModelConfigPage />} />
                    <Route path="/config/:configId" element={<ModelConfigPage />} />
                    <Route path="/fit" element={<ModelFitPage />} />
                    <Route path="/diagnostics" element={<DiagnosticsPage />} />
                    <Route path="/results" element={<ResultsPage />} />

                    {/* Model-specific routes */}
                    <Route path="/models/:modelId/diagnostics" element={<DiagnosticsPage />} />
                    <Route path="/models/:modelId/results" element={<ResultsPage />} />

                    {/* Catch all - redirect to dashboard */}
                    <Route path="*" element={<Navigate to="/dashboard" replace />} />
                  </Routes>
                </AppShell>
              </ProtectedRoute>
            }
          />
        </Routes>
      </BrowserRouter>
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  );
}

export default App;
