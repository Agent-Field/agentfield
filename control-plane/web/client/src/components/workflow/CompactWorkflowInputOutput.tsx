import { useState, useMemo } from "react";
import {
  ArrowsOutSimple,
  Code,
  DownloadSimple,
  Eye,
  UploadSimple,
} from "@/components/ui/icon-bridge";
import { cn } from "../../lib/utils";
import { Button } from "../ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../ui/tabs";
import { Badge } from "../ui/badge";
import { CopyButton } from "../ui/copy-button";
import { SegmentedControl } from "../ui/segmented-control";
import type { SegmentedControlOption } from "../ui/segmented-control";
import { UnifiedJsonViewer } from "@/components/ui/UnifiedJsonViewer";
import { DataModal } from "../execution/EnhancedModal";
import { useMainNodeExecution } from "../../hooks/useMainNodeExecution";

interface WorkflowDAGNode {
  workflow_id: string;
  execution_id: string;
  agent_node_id: string;
  reasoner_id: string;
  status: string;
  started_at: string;
  completed_at?: string;
  duration_ms?: number;
  parent_workflow_id?: string;
  parent_execution_id?: string;
  workflow_depth: number;
  children: WorkflowDAGNode[];
  notes: any[];
  notes_count: number;
  latest_note?: any;
}

interface WorkflowDAGResponse {
  root_workflow_id: string;
  session_id?: string;
  actor_id?: string;
  total_nodes: number;
  max_depth: number;
  dag: WorkflowDAGNode;
  timeline: WorkflowDAGNode[];
}

interface CompactWorkflowInputOutputProps {
  dagData: WorkflowDAGResponse | null;
  className?: string;
}

interface CompactDataViewProps {
  data: any;
  size?: number;
  type: "input" | "output";
  onViewFull: () => void;
  viewMode: "formatted" | "json";
  onViewModeChange: (mode: "formatted" | "json") => void;
}

function formatBytes(bytes?: number): string {
  if (!bytes) return "0 B";
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${sizes[i]}`;
}

const DATA_VIEW_OPTIONS: ReadonlyArray<SegmentedControlOption> = [
  { value: "formatted", label: "Formatted", icon: Eye },
  { value: "json", label: "JSON", icon: Code },
] as const;

function CompactDataView({
  data,
  size,
  type,
  onViewFull,
  viewMode,
  onViewModeChange,
}: CompactDataViewProps) {
  const jsonString = JSON.stringify(data, null, 2);
  const Icon = type === "input" ? DownloadSimple : UploadSimple;
  const title = type === "input" ? "Input Data" : "Output Data";

  if (!data) {
    return (
      <div className="flex items-center justify-center h-24 text-muted-foreground bg-muted/20 rounded-lg border border-border">
        <span className="text-sm">No {type} data available</span>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Icon className="w-4 h-4 text-foreground" />
          <span className="text-sm font-medium text-foreground">{title}</span>
          <Badge variant="secondary" className="text-xs font-mono">
            {formatBytes(size)}
          </Badge>
        </div>
        <div className="flex items-center gap-1">
          {/* View Mode Toggle */}
          <SegmentedControl
            value={viewMode}
            onValueChange={(mode) => onViewModeChange(mode as "formatted" | "json")}
            options={DATA_VIEW_OPTIONS}
            size="sm"
            optionClassName="min-w-[110px]"
          />
          <CopyButton
            value={jsonString}
            variant="ghost"
            size="icon"
            className="h-6 w-6 hover:bg-muted/80 [&_svg]:h-3 [&_svg]:w-3"
            tooltip="Copy data"
          />
          <Button
            variant="ghost"
            size="sm"
            onClick={onViewFull}
            className="h-7 w-7 p-0 hover:bg-muted"
          >
            <ArrowsOutSimple className="h-3 w-3" />
          </Button>
        </div>
      </div>

      {/* Compact Preview - Fixed scrollability */}
      <div className="border border-border rounded-md max-h-[200px] overflow-y-auto bg-card">
        <UnifiedJsonViewer
          data={data}
          className="p-3 text-xs"
        />
      </div>
    </div>
  );
}

export function CompactWorkflowInputOutput({
  dagData,
  className,
}: CompactWorkflowInputOutputProps) {
  const {
    execution,
    loading,
    error,
    hasInputData,
    hasOutputData,
    isCompleted,
    isRunning,
  } = useMainNodeExecution(dagData);

  const [inputViewMode, setInputViewMode] = useState<"formatted" | "json">(
    "formatted",
  );
  const [outputViewMode, setOutputViewMode] = useState<"formatted" | "json">(
    "formatted",
  );
  const [inputModalOpen, setInputModalOpen] = useState(false);
  const [outputModalOpen, setOutputModalOpen] = useState(false);

  // Smart tab selection based on workflow status
  const defaultTab = useMemo(() => {
    if (isRunning) return "input";
    if (isCompleted && hasOutputData) return "output";
    if (hasInputData) return "input";
    return "input";
  }, [isRunning, isCompleted, hasInputData, hasOutputData]);

  const [activeTab, setActiveTab] = useState(defaultTab);

  // Update active tab when default changes
  useMemo(() => {
    setActiveTab(defaultTab);
  }, [defaultTab]);

  // Don't render if no execution data or no relevant data
  if (loading || error || !execution || (!hasInputData && !hasOutputData)) {
    return null;
  }

  return (
    <Card className={cn("w-full", className)}>
      <CardHeader className="pb-3">
        <CardTitle className="text-base font-medium">
          Workflow Input/Output
        </CardTitle>
      </CardHeader>
      <CardContent className="pt-0">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList variant="segmented" className="grid w-full grid-cols-2">
            <TabsTrigger
              value="input"
              disabled={!hasInputData}
              variant="segmented"
              className="gap-2"
            >
              <DownloadSimple className="w-4 h-4" />
              Input
              {hasInputData && (
                <Badge variant="secondary" className="text-xs ml-1">
                  {formatBytes(execution?.input_size)}
                </Badge>
              )}
            </TabsTrigger>
            <TabsTrigger
              value="output"
              disabled={!hasOutputData}
              variant="segmented"
              className="gap-2"
            >
              <UploadSimple className="w-4 h-4" />
              Output
              {hasOutputData && (
                <Badge variant="secondary" className="text-xs ml-1">
                  {formatBytes(execution?.output_size)}
                </Badge>
              )}
            </TabsTrigger>
          </TabsList>

          <TabsContent value="input" className="mt-4 space-y-0">
            {hasInputData ? (
              <>
                <CompactDataView
                  data={execution?.input_data}
                  size={execution?.input_size}
                  type="input"
                  onViewFull={() => setInputModalOpen(true)}
                  viewMode={inputViewMode}
                  onViewModeChange={setInputViewMode}
                />

                {/* Input Modal */}
                {execution && (
                    <DataModal
                      isOpen={inputModalOpen}
                      onClose={() => setInputModalOpen(false)}
                      title="Input Data"
                      icon={DownloadSimple}
                      data={execution.input_data}
                    />
                )}
              </>
            ) : (
              <div className="flex items-center justify-center h-24 text-muted-foreground bg-muted/20 rounded-lg border border-border">
                <span className="text-sm">No input data available</span>
              </div>
            )}
          </TabsContent>

          <TabsContent value="output" className="mt-4 space-y-0">
            {hasOutputData ? (
              <>
                <CompactDataView
                  data={execution?.output_data}
                  size={execution?.output_size}
                  type="output"
                  onViewFull={() => setOutputModalOpen(true)}
                  viewMode={outputViewMode}
                  onViewModeChange={setOutputViewMode}
                />

                {/* Output Modal */}
                {execution && (
                    <DataModal
                      isOpen={outputModalOpen}
                      onClose={() => setOutputModalOpen(false)}
                      title="Output Data"
                      icon={UploadSimple}
                      data={execution.output_data}
                    />
                )}
              </>
            ) : (
              <div className="flex items-center justify-center h-24 text-muted-foreground bg-muted/20 rounded-lg border border-border">
                <span className="text-sm">No output data available</span>
              </div>
            )}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
