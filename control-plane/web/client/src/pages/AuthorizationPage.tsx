import { useCallback, useEffect, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Renew } from "@/components/ui/icon-bridge";
import {
  AnimatedTabs,
  AnimatedTabsContent,
  AnimatedTabsList,
  AnimatedTabsTrigger,
} from "@/components/ui/animated-tabs";
import { PageHeader } from "../components/PageHeader";
import { NotificationProvider } from "@/components/ui/notification";
import { AdminTokenPrompt } from "../components/AdminTokenPrompt";
import { AccessRulesTab } from "../components/authorization/AccessRulesTab";
import { AgentTagsTab } from "../components/authorization/AgentTagsTab";
import * as policiesApi from "../services/accessPoliciesApi";
import type { AccessPolicy } from "../services/accessPoliciesApi";

function AuthorizationPageContent() {
  const [policies, setPolicies] = useState<AccessPolicy[]>([]);
  const [policiesLoading, setPoliciesLoading] = useState(true);
  const [pendingCount, setPendingCount] = useState(0);
  const [refreshKey, setRefreshKey] = useState(0);

  const fetchPolicies = useCallback(async () => {
    try {
      setPoliciesLoading(true);
      const data = await policiesApi.listPolicies();
      setPolicies(data.policies || []);
    } catch {
      // Errors handled per-tab
    } finally {
      setPoliciesLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchPolicies();
  }, [fetchPolicies]);

  const handleRefreshAll = () => {
    fetchPolicies();
    setRefreshKey((k) => k + 1);
  };

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-6 overflow-hidden">
      {/* Header */}
      <PageHeader
        title="Authorization"
        description="Manage access policies and agent tag approvals"
        actions={[
          {
            label: "Refresh",
            onClick: handleRefreshAll,
            disabled: policiesLoading,
            icon: (
              <Renew
                className={`h-4 w-4 ${policiesLoading ? "animate-spin" : ""}`}
              />
            ),
          },
        ]}
      />

      {/* Admin token prompt — shared above tabs */}
      <AdminTokenPrompt onTokenSet={handleRefreshAll} />

      {/* Tabs */}
      <AnimatedTabs defaultValue="access-rules" className="flex-1 min-h-0 flex flex-col">
        <AnimatedTabsList className="border-b border-border/50 mb-2">
          <AnimatedTabsTrigger value="access-rules">
            Access Rules
          </AnimatedTabsTrigger>
          <AnimatedTabsTrigger value="agent-tags" className="gap-2">
            Agent Tags
            {pendingCount > 0 && (
              <Badge
                variant="count"
                size="sm"
              >
                {pendingCount}
              </Badge>
            )}
          </AnimatedTabsTrigger>
        </AnimatedTabsList>

        <AnimatedTabsContent value="access-rules" className="flex-1 min-h-0">
          <AccessRulesTab
            policies={policies}
            loading={policiesLoading}
            onRefresh={fetchPolicies}
          />
        </AnimatedTabsContent>

        <AnimatedTabsContent value="agent-tags" className="flex-1 min-h-0">
          <AgentTagsTab
            policies={policies}
            onPendingCountChange={setPendingCount}
            refreshKey={refreshKey}
          />
        </AnimatedTabsContent>
      </AnimatedTabs>
    </div>
  );
}

export function AuthorizationPage() {
  return (
    <NotificationProvider>
      <AuthorizationPageContent />
    </NotificationProvider>
  );
}
