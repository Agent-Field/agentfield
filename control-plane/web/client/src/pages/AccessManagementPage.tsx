import { useMemo } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { ExternalLink, RefreshCw } from "lucide-react";

import { AdminTokenPrompt } from "@/components/AdminTokenPrompt";
import { AccessRulesTab } from "@/components/authorization/AccessRulesTab";
import { AgentTagsTab } from "@/components/authorization/AgentTagsTab";
import { HintIcon } from "@/components/authorization/HintIcon";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
import {
  TooltipProvider,
} from "@/components/ui/tooltip";
import { Skeleton } from "@/components/ui/skeleton";
import { useAuth } from "@/contexts/AuthContext";
import {
  ACCESS_MANAGEMENT_QUERY_KEY,
  useAgentTagSummaries,
  useAccessAdminRoutesProbe,
  useAccessPolicies,
} from "@/hooks/queries";
import { countPendingAgentTags } from "@/lib/governanceUtils";
import { cn } from "@/lib/utils";

const DOCS_KB_PATH = "/api/v1/agentic/kb/articles/identity/tag-authorization";

function AccessManagementPageInner() {
  const queryClient = useQueryClient();
  const { adminToken } = useAuth();

  const probeQuery = useAccessAdminRoutesProbe(adminToken);
  const adminRoutesAvailable = probeQuery.data === true;

  const policiesQuery = useAccessPolicies(
    probeQuery.isSuccess && adminRoutesAvailable,
  );

  const tagsQuery = useAgentTagSummaries();

  const pendingCount = useMemo(
    () => countPendingAgentTags(tagsQuery.data ?? []),
    [tagsQuery.data],
  );

  const invalidateAccessQueries = () => {
    void queryClient.invalidateQueries({ queryKey: [ACCESS_MANAGEMENT_QUERY_KEY] });
  };

  const policiesLoading =
    probeQuery.isLoading ||
    (adminRoutesAvailable && policiesQuery.isLoading);

  const showProbeSkeleton = probeQuery.isLoading;

  return (
    <TooltipProvider delayDuration={200}>
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <div className="flex items-center gap-1">
            <h1 className="text-2xl font-semibold tracking-tight">Access management</h1>
            <HintIcon label="What this page does">
              Tag rules for cross-agent calls and registration-tag approvals. When
              the server expects it, use the browser admin token below—separate from
              your normal API key.
            </HintIcon>
          </div>
          <p className="text-muted-foreground mt-1 text-sm">
            Cross-agent rules and tag approvals.
          </p>
        </div>
        <Button
          variant="outline"
          size="sm"
          className="shrink-0 gap-2"
          disabled={
            probeQuery.isFetching ||
            policiesQuery.isFetching ||
            tagsQuery.isFetching
          }
          onClick={() => invalidateAccessQueries()}
        >
          <RefreshCw
            className={cn(
              "size-3.5",
              (probeQuery.isFetching ||
                policiesQuery.isFetching ||
                tagsQuery.isFetching) &&
                "animate-spin",
            )}
          />
          Refresh
        </Button>
      </div>

      {showProbeSkeleton ? (
        <Skeleton className="h-24 w-full rounded-lg" />
      ) : probeQuery.isError ? (
        <Alert variant="destructive">
          <AlertTitle>Could not reach the control plane</AlertTitle>
          <AlertDescription>
            {probeQuery.error instanceof Error
              ? probeQuery.error.message
              : "Try again after checking your API key and network."}
          </AlertDescription>
        </Alert>
      ) : !adminRoutesAvailable ? (
        <Alert>
          <AlertTitle>Authorization APIs are not enabled on this server</AlertTitle>
          <AlertDescription className="space-y-2 text-sm">
            <p>
              Policy and tag-approval admin routes are only registered when VC
              authorization is turned on. Enable it in control plane config,
              for example:
            </p>
            <ul className="list-disc pl-4 text-xs font-mono text-muted-foreground">
              <li>AGENTFIELD_AUTHORIZATION_ENABLED=true</li>
              <li>
                AGENTFIELD_AUTHORIZATION_ADMIN_TOKEN=… (recommended for
                production)
              </li>
            </ul>
            <p className="text-xs text-muted-foreground">
              See <code className="rounded bg-muted px-1">control-plane/.env.example</code>{" "}
              and your <code className="rounded bg-muted px-1">agentfield.yaml</code>{" "}
              <span className="font-sans">features.did.authorization</span> block.
            </p>
            <a
              href={DOCS_KB_PATH}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1 text-xs font-medium text-primary hover:underline"
            >
              Knowledge base: Tag-based authorization
              <ExternalLink className="size-3" />
            </a>
          </AlertDescription>
        </Alert>
      ) : null}

      <Card className="border-border/80">
        <CardHeader className="pb-3">
          <div className="flex items-center gap-1">
            <CardTitle className="text-sm font-medium">Browser admin token</CardTitle>
            <HintIcon label="About the admin token">
              Same secret as server config (<code className="font-mono">admin_token</code> or{" "}
              <code className="font-mono">AGENTFIELD_AUTHORIZATION_ADMIN_TOKEN</code>). Not in the DB
              or Settings—only this browser. Sends <code className="font-mono">X-Admin-Token</code> when
              set.
            </HintIcon>
          </div>
        </CardHeader>
        <CardContent className="pt-0">
          <AdminTokenPrompt onTokenSet={() => invalidateAccessQueries()} />
        </CardContent>
      </Card>

      <Tabs defaultValue="access-rules" className="flex flex-col gap-0">
        <TabsList variant="underline" className="w-full justify-start">
          <TabsTrigger value="access-rules" variant="underline">
            Access rules
          </TabsTrigger>
          <TabsTrigger value="agent-tags" variant="underline" className="gap-2">
            Agent tags
            {pendingCount > 0 && (
              <Badge variant="secondary" className="h-5 min-w-5 px-1.5 text-micro">
                {pendingCount}
              </Badge>
            )}
          </TabsTrigger>
        </TabsList>

        <TabsContent value="access-rules" className="mt-6">
          <Card>
            <CardHeader>
              <div className="flex items-center gap-1">
                <CardTitle className="text-sm font-medium">
                  Cross-agent access policies
                </CardTitle>
                <HintIcon label="How access rules work">
                  Caller tags → target tags for cross-agent calls. Higher priority first.
                  Optional allow/deny function patterns.
                </HintIcon>
              </div>
            </CardHeader>
            <CardContent>
              <AccessRulesTab
                policies={policiesQuery.data ?? []}
                loading={policiesLoading}
                onRefresh={() => void policiesQuery.refetch()}
                canMutate={adminRoutesAvailable}
                fetchError={
                  policiesQuery.isError
                    ? policiesQuery.error instanceof Error
                      ? policiesQuery.error
                      : new Error(String(policiesQuery.error))
                    : null
                }
              />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="agent-tags" className="mt-6">
          <Card>
            <CardHeader>
              <div className="flex items-center gap-1">
                <CardTitle className="text-sm font-medium">
                  Agent tag approvals
                </CardTitle>
                <HintIcon label="About agent tags">
                  One row per registered agent. Approve or reject tags from registration—
                  approved tags drive access checks. Pending needs a decision for gated traffic.
                </HintIcon>
              </div>
            </CardHeader>
            <CardContent>
              <AgentTagsTab
                policies={policiesQuery.data ?? []}
                agents={tagsQuery.data ?? []}
                agentsLoading={tagsQuery.isLoading}
                agentsError={
                  tagsQuery.isError
                    ? tagsQuery.error instanceof Error
                      ? tagsQuery.error
                      : new Error(String(tagsQuery.error))
                    : null
                }
                canMutate={adminRoutesAvailable}
                onRefresh={() => invalidateAccessQueries()}
              />
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
    </TooltipProvider>
  );
}

export function AccessManagementPage() {
  return <AccessManagementPageInner />;
}

export default AccessManagementPage;
