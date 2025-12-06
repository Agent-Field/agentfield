import type { Icon } from "@/components/ui/icon-bridge";
import {
  GitBranch,
  Database,
  BarChart3,
  ShieldCheck,
  RadioTower,
  FileText,
  Layers,
} from "@/components/ui/icon-bridge";
import { Badge } from "../ui/badge";
import { useIsMobile } from "@/hooks/use-mobile";
import {
  AnimatedTabs,
  AnimatedTabsList,
  AnimatedTabsTrigger,
} from "../ui/animated-tabs";
import { cn } from "../../lib/utils";
import { getStatusLabel, normalizeExecutionStatus } from "../../utils/status";
import type { WorkflowSummary, WorkflowTimelineNode } from "../../types/workflows";
import { summarizeWorkflowWebhook } from "../../utils/webhook";
import type { WorkflowVCChainResponse } from "../../types/did";

type WorkflowTabId = 'graph' | 'io' | 'webhooks' | 'notes' | 'identity' | 'insights';

interface EnhancedWorkflowTabsProps {
  activeTab: WorkflowTabId;
  onTabChange: (tab: WorkflowTabId) => void;
  workflow: WorkflowSummary;
  dagData?: { timeline?: WorkflowTimelineNode[] } | null;
  vcChain?: WorkflowVCChainResponse | null;
  className?: string;
}

export function EnhancedWorkflowTabs({
  activeTab,
  onTabChange,
  workflow,
  dagData,
  vcChain,
  className
}: EnhancedWorkflowTabsProps) {
  const isMobile = useIsMobile();
  const normalizedStatus = normalizeExecutionStatus(workflow.status);

  const timeline = dagData?.timeline ?? [];
  const webhookSummary = summarizeWorkflowWebhook(timeline);

  const getTabCount = (tabType: WorkflowTabId) => {
    switch (tabType) {
      case 'graph':
        return timeline.length || workflow.total_executions;
      case 'io':
        return timeline.filter((node) => node.input_data || node.output_data)?.length || 0;
      case 'webhooks':
        return webhookSummary.nodesWithWebhook;
      case 'notes':
        return timeline.reduce((count: number, node) => count + (node.notes?.length || 0), 0) || 0;
      case 'identity':
        return vcChain?.component_vcs?.length || 0;
      case 'insights':
        return timeline.filter((node) => node.duration_ms)?.length || 0;
      default:
        return undefined;
    }
  };

  const tabs: Array<{
    id: WorkflowTabId;
    label: string;
    icon: Icon;
    description: string;
    shortcut: string;
    count?: number;
  }> = [
    {
      id: 'graph',
      label: 'Graph',
      icon: GitBranch,
      description: 'Live workflow topology',
      shortcut: '1',
      count: getTabCount('graph'),
    },
    {
      id: 'io',
      label: 'Inputs & Outputs',
      icon: Database,
      description: 'Inspect node inputs and outputs',
      shortcut: '2',
      count: getTabCount('io'),
    },
    {
      id: 'webhooks',
      label: 'Webhooks',
      icon: RadioTower,
      description: 'Callback deliveries and status',
      shortcut: '3',
      count: getTabCount('webhooks'),
    },
    {
      id: 'notes',
      label: 'Notes',
      icon: FileText,
      description: 'Operator notes, annotations, and context',
      shortcut: '4',
      count: getTabCount('notes'),
    },
    {
      id: 'identity',
      label: 'Identity',
      icon: ShieldCheck,
      description: 'Trust, credentials, and verification chain',
      shortcut: '5',
      count: getTabCount('identity'),
    },
    {
      id: 'insights',
      label: 'Insights',
      icon: BarChart3,
      description: 'Performance and health analytics',
      shortcut: '6',
      count: getTabCount('insights'),
    },
  ];

  return (
    <div className={cn(
      "h-12 bg-background flex items-center overflow-x-auto scrollbar-hide",
      isMobile ? "px-4" : "px-6",
      className
    )}>
      <AnimatedTabs
        value={activeTab}
        onValueChange={(value) => onTabChange(value as WorkflowTabId)}
        className="h-full min-w-0"
      >
        <AnimatedTabsList className={cn(
          "h-full gap-1",
          isMobile ? "flex-nowrap" : ""
        )}>
          {tabs.map((tab) => {
            const Icon = tab.icon;

            return (
              <AnimatedTabsTrigger
                key={tab.id}
                value={tab.id}
                className="gap-2 px-3 py-2 flex-shrink-0"
                title={`${tab.description} (Cmd/Ctrl + ${tab.shortcut})`}
              >
                <Icon className="w-4 h-4" />
                <span>{tab.label}</span>

                {tab.count !== undefined && tab.count > 0 && (
                  <Badge variant="secondary" className="text-xs h-5 px-1.5 min-w-[20px]">
                    {tab.count > 999 ? '999+' : tab.count}
                  </Badge>
                )}
              </AnimatedTabsTrigger>
            );
          })}
        </AnimatedTabsList>
      </AnimatedTabs>

      {/* Tab Context Info */}
      <div className={cn(
        "flex items-center gap-4 ml-6 text-body-small",
        isMobile ? "hidden" : "flex"
      )}>
        {/* Current Tab Info */}
        {activeTab === 'graph' && timeline.length > 0 && (
          <div className="flex items-center gap-4">
            <span>Total nodes: {timeline.length}</span>
            <span>•</span>
            <span>Max depth: {workflow.max_depth}</span>
          </div>
        )}

        {activeTab === 'io' && timeline.length > 0 && (
          <div className="flex items-center gap-2">
            <Database className="w-3 h-3" />
            <span>Data available for {getTabCount('io')} nodes</span>
          </div>
        )}

        {activeTab === 'notes' && timeline.length > 0 && (
          <div className="flex items-center gap-2">
            <FileText className="w-3 h-3" />
            <span>{getTabCount('notes')} total notes</span>
          </div>
        )}

        {activeTab === 'webhooks' && (
          <div className="flex items-center gap-2">
            <RadioTower className="w-3 h-3" />
            <span>
              {webhookSummary.failedDeliveries > 0
                ? `${webhookSummary.failedDeliveries} failures across callbacks`
                : webhookSummary.successDeliveries > 0
                  ? `${webhookSummary.successDeliveries} successful deliveries`
                  : `${webhookSummary.nodesWithWebhook} nodes monitoring callbacks`}
            </span>
          </div>
        )}

        {activeTab === 'identity' && (
          <div className="flex items-center gap-2">
            <ShieldCheck className="w-3 h-3" />
            <span>
              {vcChain?.component_vcs?.length ? `${vcChain.component_vcs.length} credentials linked` : 'No credentials issued yet'}
            </span>
          </div>
        )}

        {activeTab === 'insights' && timeline.length > 0 && (
          <div className="flex items-center gap-2">
            <Layers className="w-3 h-3" />
            <span>Metrics for {getTabCount('insights')} nodes</span>
          </div>
        )}
      </div>

      {/* Right Side Actions */}
      <div className={cn(
        "ml-auto flex items-center gap-2",
        isMobile ? "hidden" : "flex"
      )}>
        {/* Workflow Status */}
        <div className="flex items-center gap-2 text-xs">
          <div
            className={cn(
              "w-1.5 h-1.5 rounded-full",
              normalizedStatus === 'running' && "bg-amber-500 animate-pulse",
              normalizedStatus === 'succeeded' && "bg-green-500",
              normalizedStatus === 'failed' && "bg-red-500",
              normalizedStatus === 'cancelled' && "bg-gray-500",
              normalizedStatus === 'timeout' && "bg-purple-500",
              (normalizedStatus === 'queued' || normalizedStatus === 'pending') && "bg-amber-400",
              normalizedStatus === 'unknown' && "bg-muted"
            )}
          />
          <span className="text-muted-foreground capitalize">
            {getStatusLabel(normalizedStatus)}
          </span>
        </div>
      </div>
    </div>
  );
}
