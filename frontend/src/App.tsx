import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import { Component, type ReactNode } from 'react';
import { AppShell } from './components/layout';
import { ProtectedRoute } from './components/common';
import {
  LoginPage,
  DashboardPage,
  PlanningPage,
  AgentPage,
} from './pages';
import './index.css';

class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null };
  static getDerivedStateFromError(error: Error) { return { error }; }
  render() {
    if (this.state.error) {
      const e = this.state.error as Error;
      return (
        <div className="min-h-screen flex items-center justify-center bg-gray-50 p-8">
          <div className="max-w-lg w-full bg-white rounded-xl border border-red-200 shadow p-6">
            <h1 className="text-lg font-semibold text-red-700 mb-2">Something went wrong</h1>
            <pre className="text-sm text-gray-700 whitespace-pre-wrap bg-gray-50 rounded p-3 border border-gray-200">
              {e.message}{'\n\n'}{e.stack}
            </pre>
            <button
              onClick={() => { this.setState({ error: null }); window.location.reload(); }}
              className="mt-4 px-4 py-2 bg-indigo-600 text-white text-sm rounded-lg hover:bg-indigo-700"
            >Reload</button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

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
    <ErrorBoundary>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          {/* Public route - Login */}
          <Route path="/login" element={<LoginPage />} />

          {/* Chat — full-screen layout, no AppShell */}
          <Route
            path="/chat"
            element={
              <ProtectedRoute>
                <AgentPage />
              </ProtectedRoute>
            }
          />

          {/* Protected routes — share AppShell */}
          <Route
            path="/*"
            element={
              <ProtectedRoute>
                <AppShell>
                  <Routes>
                    <Route path="/" element={<Navigate to="/dashboard" replace />} />
                    <Route path="/dashboard" element={<DashboardPage />} />
                    <Route path="/analysis-plan" element={<PlanningPage />} />
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
    </ErrorBoundary>
  );
}

export default App;
