import { Database } from "@/components/ui/icon-bridge";
import { Card, CardContent, CardHeader, CardTitle } from "../../ui/card";
import { UnifiedDataPanel } from "../../ui/UnifiedDataPanel";

interface WorkflowNodeData {
  workflow_id: string;
  execution_id: string;
  agent_node_id: string;
  reasoner_id: string;
  status: string;
  started_at: string;
  completed_at?: string;
  duration_ms?: number;
  workflow_depth: number;
  task_name?: string;
  agent_name?: string;
}

interface NodeDetails {
  input?: any;
  output?: any;
  error_message?: string;
  cost?: number;
  memory_updates?: any[];
  performance_metrics?: {
    response_time_ms: number;
    tokens_used?: number;
  };
}

interface DataSectionProps {
  node: WorkflowNodeData;
  details?: NodeDetails;
}

const formatBytes = (bytes?: number): string => {
  if (!bytes) return "0 B";
  const thresholds = ["B", "KB", "MB", "GB"];
  const index = Math.min(
    thresholds.length - 1,
    Math.floor(Math.log(bytes) / Math.log(1024))
  );
  const value = bytes / Math.pow(1024, index);
  return `${value >= 10 || index === 0 ? value.toFixed(0) : value.toFixed(1)} ${
    thresholds[index]
  }`;
};

const calculateDataSize = (data: unknown): number | undefined => {
  if (data === null || data === undefined) {
    return undefined;
  }

  try {
    const payload = typeof data === "string" ? data : JSON.stringify(data);
    return new Blob([payload]).size;
  } catch {
    return undefined;
  }
};

export function DataSection({ details }: DataSectionProps) {
  const inputData = details?.input;
  const outputData = details?.output;

  const inputSize = calculateDataSize(inputData);
  const outputSize = calculateDataSize(outputData);

  return (
    <Card className="border-border bg-card">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-sm font-medium text-foreground">
          <Database className="h-4 w-4 text-muted-foreground" />
          Input & Output
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 pt-0">
        <UnifiedDataPanel
          data={inputData}
          title="Input Data"
          type="input"
          size={inputSize}
          className="rounded-lg border border-border/80 bg-background/60"
          showModalButton={false}
          maxHeight="200px"
        />
        <UnifiedDataPanel
          data={outputData}
          title="Output Data"
          type="output"
          size={outputSize}
          className="rounded-lg border border-border/80 bg-background/60"
          showModalButton={false}
          maxHeight="200px"
        />
        {(inputSize || outputSize) && (
          <div className="grid grid-cols-2 gap-4 rounded-lg border border-dashed border-border/70 bg-muted/40 p-3 text-sm text-muted-foreground text-muted-foreground">
            <div>
              <span className="block text-sm text-muted-foreground font-medium text-foreground/80">
                Input Size
              </span>
              <span className="font-mono text-xs">{formatBytes(inputSize)}</span>
            </div>
            <div>
              <span className="block text-sm text-muted-foreground font-medium text-foreground/80">
                Output Size
              </span>
              <span className="font-mono text-xs">
                {formatBytes(outputSize)}
              </span>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
