import { useState } from "react";
import {
  ArrowCounterClockwise,
  ArrowSquareOut,
  Check,
  Copy,
  SpinnerGap,
  Play,
  Terminal,
  Code,
  WarningCircle,
} from "@/components/ui/icon-bridge";
import { useNavigate } from "react-router-dom";
import type { WorkflowExecution } from "../../types/executions";
import { cn } from "../../lib/utils";
import { statusTone } from "../../lib/theme";
import { Button } from "../ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import { Badge } from "../ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../ui/tabs";
import { CopyButton } from "../ui/copy-button";

interface ExecutionRetryPanelProps {
  execution: WorkflowExecution;
}

function generateCurlCommand(execution: WorkflowExecution): string {
  // Use the correct API format: nodeid.reasonerid
  const baseUrl = window.location.origin.replace('/ui', ''); // Remove /ui if present
  const target = `${execution.agent_node_id}.${execution.reasoner_id}`;
  const apiUrl = `${baseUrl}/api/v1/execute/${target}`;

  const payload = {
    input: execution.input_data || {}
  };

  const curlCommand = `curl -X POST "${apiUrl}" \\
  -H "Content-Type: application/json" \\
  -d '${JSON.stringify(payload, null, 2).replace(/'/g, "'\\''")}' \\
  --silent \\
  --show-error`;

  return curlCommand;
}

function generatePythonCode(execution: WorkflowExecution): string {
  const baseUrl = window.location.origin.replace('/ui', ''); // Remove /ui if present
  const target = `${execution.agent_node_id}.${execution.reasoner_id}`;
  const apiUrl = `${baseUrl}/api/v1/execute/${target}`;

  const payload = {
    input: execution.input_data || {}
  };

  return `import requests
import json

url = "${apiUrl}"
payload = ${JSON.stringify(payload, null, 2)}

headers = {
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)

if response.status_code == 200:
    result = response.json()
    print(f"Execution ID: {result.get('execution_id', 'N/A')}")
    print(f"Status: {result.get('status', 'N/A')}")
else:
    print(f"Error: {response.status_code} - {response.text}")`;
}

export function ExecutionRetryPanel({ execution }: ExecutionRetryPanelProps) {
  const [isRetrying, setIsRetrying] = useState(false);
  const [retryResult, setRetryResult] = useState<{ success: boolean; message: string; executionId?: string } | null>(null);
  const navigate = useNavigate();

  const curlCommand = generateCurlCommand(execution);
  const pythonCode = generatePythonCode(execution);

  const renderCopyAction = (value: string, label: string, compact = false) => (
    <CopyButton
      value={value}
      variant="outline"
      size="sm"
      className={compact ? "h-6 px-2 text-xs" : "h-8 px-3 text-sm"}
      tooltip={`Copy ${label} snippet`}
    >
      {(copied) =>
        copied ? (
          <>
            <Check className="w-3 h-3 mr-1 text-status-success" />
            {compact ? "✓" : "Copied!"}
          </>
        ) : (
          <>
            <Copy className="w-3 h-3 mr-1" />
            {label}
          </>
        )
      }
    </CopyButton>
  );

  const handleRetry = async () => {
    setIsRetrying(true);
    setRetryResult(null);

    try {
      // Use the correct API format: nodeid.reasonerid
      const baseUrl = window.location.origin.replace('/ui', '');
      const target = `${execution.agent_node_id}.${execution.reasoner_id}`;
      const apiUrl = `${baseUrl}/api/v1/execute/${target}`;

      const payload = {
        input: execution.input_data || {}
      };

      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      });

      if (response.ok) {
        const result = await response.json();
        setRetryResult({
          success: true,
          message: `New execution started successfully`,
          executionId: result.execution_id
        });
      } else {
        const errorText = await response.text();
        setRetryResult({
          success: false,
          message: `Failed to start execution: ${response.status} - ${errorText}`
        });
      }
    } catch (error) {
      setRetryResult({
        success: false,
        message: `Network error: ${error instanceof Error ? error.message : 'Unknown error'}`
      });
    } finally {
      setIsRetrying(false);
    }
  };

  const hasInputData = execution.input_data && Object.keys(execution.input_data).length > 0;

  return (
    <Card className="border-border/60">
      <CardHeader className="pb-4">
        <CardTitle className="flex items-center gap-2">
          <ArrowCounterClockwise className="h-5 w-5" />
          Execution Retry
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Compact Action Bar */}
        <div className="flex flex-wrap items-center gap-3 p-4 bg-muted/30 rounded-lg border border-border/40">
          <Button
            onClick={handleRetry}
            disabled={isRetrying}
            className="flex items-center gap-2"
            size="sm"
          >
            {isRetrying ? (
              <>
                <SpinnerGap className="h-4 w-4 animate-spin" />
                Retrying...
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                Retry Now
              </>
            )}
          </Button>

          <div className="h-4 w-px bg-border" />

          {renderCopyAction(curlCommand, "cURL", true)}
          {renderCopyAction(pythonCode, "Python", true)}

          <div className="flex items-center gap-2 text-sm text-muted-foreground ml-auto">
            <Badge variant="secondary" className="text-xs">
              {execution.agent_node_id}.{execution.reasoner_id}
            </Badge>
            {!hasInputData && (
              <div
                className={cn(
                  "flex items-center gap-1",
                  statusTone.warning.fg
                )}
              >
                <WarningCircle className="h-3 w-3" />
                <span className="text-xs font-medium">Empty input</span>
              </div>
            )}
          </div>
        </div>

        {/* Status Feedback */}
        {retryResult && (
          <div
            className={cn(
              "flex items-start justify-between rounded-lg border p-3",
              retryResult.success
                ? [
                    statusTone.success.bg,
                    statusTone.success.border,
                    statusTone.success.fg,
                  ]
                : [
                    statusTone.error.bg,
                    statusTone.error.border,
                    statusTone.error.fg,
                  ]
            )}
          >
            <div className="flex-1">
              <p className="text-sm font-medium">
                {retryResult.success ? 'Success' : 'Error'}
              </p>
              <p className="text-sm mt-1">{retryResult.message}</p>
              {retryResult.executionId && (
                <p className="text-xs mt-2 font-mono">
                  Execution ID: {retryResult.executionId}
                </p>
              )}
            </div>
            {retryResult.success && retryResult.executionId && (
              <Button
                variant="outline"
                size="sm"
                onClick={() => navigate(`/executions/${retryResult.executionId}`)}
                className="ml-3 h-8"
              >
                <ArrowSquareOut className="mr-1 h-3 w-3" />
                View
              </Button>
            )}
          </div>
        )}

        {/* Code Examples - Compact Tabs */}
        <Tabs defaultValue="curl" className="w-full">
          <TabsList variant="segmented" className="grid w-full grid-cols-2">
            <TabsTrigger value="curl" variant="segmented" className="gap-2">
              <Terminal className="w-4 h-4" />
              cURL
            </TabsTrigger>
            <TabsTrigger value="python" variant="segmented" className="gap-2">
              <Code className="w-4 h-4" />
              Python
            </TabsTrigger>
          </TabsList>

          <TabsContent value="curl" className="mt-4">
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <p className="text-sm text-muted-foreground">
                  Terminal command to retry this execution
                </p>
                {renderCopyAction(curlCommand, "Copy")}
              </div>
              <div className="relative">
                <pre className="json-hl-root bg-muted/50 p-3 rounded-lg text-xs font-mono overflow-x-auto max-h-32">
                  <code>{curlCommand}</code>
                </pre>
              </div>
              <div className="text-sm text-muted-foreground">
                💡 Add <code>| jq</code> for pretty JSON output
              </div>
            </div>
          </TabsContent>

          <TabsContent value="python" className="mt-4">
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <p className="text-sm text-muted-foreground">
                  Python script using requests library
                </p>
                {renderCopyAction(pythonCode, "Copy")}
              </div>
              <div className="relative">
                <pre className="json-hl-root bg-muted/50 p-3 rounded-lg text-xs font-mono overflow-x-auto max-h-32">
                  <code>{pythonCode}</code>
                </pre>
              </div>
              <div className="text-sm text-muted-foreground">
                📦 Install: <code>pip install requests</code>
              </div>
            </div>
          </TabsContent>
        </Tabs>

        {/* Execution Context */}
        <div className="pt-4 border-t border-border/40">
          <div className="flex flex-wrap gap-2 text-xs">
            <span className="text-muted-foreground">Context:</span>
            <Badge variant="outline">Target: {execution.agent_node_id}.{execution.reasoner_id}</Badge>
            {execution.session_id && (
              <Badge variant="outline">Session: {execution.session_id.slice(0, 8)}...</Badge>
            )}
            {execution.actor_id && (
              <Badge variant="outline">Actor: {execution.actor_id.slice(0, 8)}...</Badge>
            )}
            {hasInputData && (
              <Badge variant="secondary">
                {Object.keys(execution.input_data || {}).length} input params
              </Badge>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
