import { Navigate } from 'react-router-dom';

export function RootRedirect() {
  return <Navigate to="/dashboard" replace />;
}
