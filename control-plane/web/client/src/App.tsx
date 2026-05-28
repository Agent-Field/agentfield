import { Navigate, Route, BrowserRouter as Router, Routes, useParams } from "react-router-dom";
import { QueryClientProvider } from "@tanstack/react-query";
import { RootRedirect } from "./components/RootRedirect";
import { ModeProvider } from "./contexts/ModeContext";
import { ThemeProvider } from "./components/theme-provider";
import { AppLayout } from "./components/AppLayout";
import { NewDashboardPage } from "./pages/NewDashboardPage";
import { NewSettingsPage } from "./pages/NewSettingsPage";
import { AgentsPage } from "./pages/AgentsPage";
import { RunsPage } from "./pages/RunsPage";
import { RunDetailPage } from "./pages/RunDetailPage";
import { VerifyProvenancePage } from "./pages/VerifyProvenancePage";
import { ComparisonPage } from "./pages/ComparisonPage";
import { PlaygroundPage } from "./pages/PlaygroundPage";
import { AccessManagementPage } from "./pages/AccessManagementPage";
import { TriggersPage } from "./pages/TriggersPage";
import { IntegrationsPage } from "./pages/IntegrationsPage";
import { AuthProvider } from "./contexts/AuthContext";
import { AuthGuard } from "./components/AuthGuard";
import { queryClient } from "./lib/query-client";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { NotificationProvider } from "./components/ui/notification";

function getRouterBasename(): string {
  const configuredBase = import.meta.env.VITE_BASE_PATH;
  if (configuredBase && configuredBase !== "/") {
    return configuredBase;
  }

  if (typeof window !== "undefined" && window.location.pathname.startsWith("/ui")) {
    return "/ui";
  }

  return "/";
}

function NavigateToPlayground() {
  const { reasonerId } = useParams();
  return <Navigate to={`/playground/${reasonerId}`} replace />;
}

function AppContent() {
  return (
    <Routes>
      <Route element={<AppLayout />}>
        <Route path="/" element={<RootRedirect />} />
        <Route path="/dashboard" element={<NewDashboardPage />} />
        <Route path="/settings" element={<NewSettingsPage />} />
        <Route path="/settings/observability-webhook" element={<Navigate to="/settings" replace />} />
        <Route path="/agents" element={<AgentsPage />} />
        <Route path="/runs" element={<RunsPage />} />
        <Route path="/runs/compare" element={<ComparisonPage />} />
        <Route path="/runs/:runId" element={<RunDetailPage />} />
        <Route path="/verify" element={<VerifyProvenancePage />} />
        <Route path="/playground" element={<PlaygroundPage />} />
        <Route path="/playground/:reasonerId" element={<PlaygroundPage />} />
        <Route path="/access" element={<AccessManagementPage />} />
        <Route path="/triggers" element={<TriggersPage />} />
        <Route path="/integrations" element={<IntegrationsPage />} />

        {/* Old → New redirects */}
        <Route path="/executions" element={<Navigate to="/runs" replace />} />
        <Route path="/executions/:executionId" element={<Navigate to="/runs" replace />} />
        <Route path="/workflows" element={<Navigate to="/runs" replace />} />
        <Route path="/workflows/:workflowId" element={<Navigate to="/runs" replace />} />
        <Route path="/nodes" element={<Navigate to="/agents" replace />} />
        <Route path="/nodes/:nodeId" element={<Navigate to="/agents" replace />} />
        <Route path="/reasoners/all" element={<Navigate to="/agents" replace />} />
        <Route path="/reasoners/:reasonerId" element={<NavigateToPlayground />} />
        <Route path="/identity/dids" element={<Navigate to="/settings" replace />} />
        <Route path="/identity/credentials" element={<Navigate to="/settings" replace />} />
        <Route path="/governance" element={<Navigate to="/access" replace />} />
        <Route path="/authorization" element={<Navigate to="/access" replace />} />
        <Route path="/packages" element={<Navigate to="/settings" replace />} />
      </Route>
    </Routes>
  );
}

function App() {
  const basename = getRouterBasename();

  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider
        attribute="class"
        defaultTheme="dark"
        enableSystem
        disableTransitionOnChange
      >
        <ModeProvider>
          <NotificationProvider>
            <AuthProvider>
              <AuthGuard>
                <Router basename={basename}>
                  <ErrorBoundary>
                    <AppContent />
                  </ErrorBoundary>
                </Router>
              </AuthGuard>
            </AuthProvider>
          </NotificationProvider>
        </ModeProvider>
      </ThemeProvider>
    </QueryClientProvider>
  );
}

export default App;
