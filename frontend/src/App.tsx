import { BrowserRouter, Routes, Route, Navigate, useLocation } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import { Component, type ReactNode } from 'react';
import { AppShell } from './components/layout';
import { ProtectedRoute } from './components/common';
import {
  LoginPage,
  ProgramPage,
  ExperimentsPage,
  PerformancePage,
  PortfolioPage,
  TeamPage,
  KnowledgePage,
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
        <div className="min-h-screen flex items-center justify-center bg-cream-50 p-8">
          <div className="max-w-lg w-full bg-white rounded-xl border border-rust-600/30 shadow p-6">
            <h1 className="text-lg font-semibold text-rust-700 mb-2 font-display">Something went wrong</h1>
            <pre className="text-sm text-ink-700 whitespace-pre-wrap bg-cream-100 rounded p-3 border border-line-200">
              {e.message}{'\n\n'}{e.stack}
            </pre>
            <button
              onClick={() => { this.setState({ error: null }); window.location.reload(); }}
              className="mt-4 px-4 py-2 bg-sage-700 text-white text-sm rounded-lg hover:bg-sage-800"
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

/** Permanent /chat → /workspace redirect that forwards query params
 * (deep links like /chat?session=... exist in saved bookmarks + old emails). */
function ChatRedirect() {
  const location = useLocation();
  return <Navigate to={`/workspace${location.search}`} replace />;
}

function App() {
  return (
    <ErrorBoundary>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          {/* Public route - Login */}
          <Route path="/login" element={<LoginPage />} />

          {/* Workspace (chat) — inside the AppShell (shared nav, header, and
              project switcher) with a full-bleed content area */}
          <Route
            path="/workspace"
            element={
              <ProtectedRoute>
                <AppShell fullBleed>
                  <AgentPage />
                </AppShell>
              </ProtectedRoute>
            }
          />
          <Route path="/chat" element={<ChatRedirect />} />

          {/* Protected routes — share AppShell */}
          <Route
            path="/*"
            element={
              <ProtectedRoute>
                <AppShell>
                  <Routes>
                    <Route path="/" element={<Navigate to="/program" replace />} />
                    <Route path="/program" element={<ProgramPage />} />
                    <Route path="/experiments" element={<ExperimentsPage />} />
                    <Route path="/experiments/:experimentId" element={<ExperimentsPage />} />
                    <Route path="/performance" element={<PerformancePage />} />
                    <Route path="/performance/:tab" element={<PerformancePage />} />
                    <Route path="/portfolio" element={<PortfolioPage />} />
                    <Route path="/knowledge" element={<KnowledgePage />} />
                    <Route path="/team" element={<TeamPage />} />
                    {/* Legacy paths */}
                    <Route path="/dashboard" element={<Navigate to="/program" replace />} />
                    <Route path="/runs" element={<Navigate to="/performance/runs" replace />} />
                    {/* The standalone Planning page was folded into the agent
                        workspace's Causal tab (CausalPlanner). */}
                    <Route path="/analysis-plan" element={<Navigate to="/workspace" replace />} />
                    <Route path="*" element={<Navigate to="/program" replace />} />
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
