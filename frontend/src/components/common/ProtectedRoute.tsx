import { Navigate, useLocation } from 'react-router-dom';
import { useAuthStore } from '../../stores/authStore';

interface ProtectedRouteProps {
  children: React.ReactNode;
}

export function ProtectedRoute({ children }: ProtectedRouteProps) {
  const { isAuthenticated, accessToken } = useAuthStore();
  const location = useLocation();

  // When JWT auth is explicitly enabled (VITE_AUTH_ENABLED), gate on an access
  // token; otherwise keep the existing isAuthenticated (API-key) gate so the
  // dev/single-user posture is unchanged.
  const authEnabled = import.meta.env.VITE_AUTH_ENABLED === 'true';
  const allowed = authEnabled ? !!accessToken : isAuthenticated;

  if (!allowed) {
    // Redirect to login, preserving the intended destination
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  return <>{children}</>;
}
