import {
  Activity,
  Analytics,
  CheckmarkFilled,
  Code,
  Copy,
  InProgress,
  Play,
  Time,
  View,
} from "../components/ui/icon-bridge";
import { useCallback, useEffect, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import { DIDIdentityBadge } from "../components/did/DIDDisplay";
import { Badge } from "../components/ui/badge";
import { ExecutionForm, type ExecutionFormData } from "../components/reasoners/ExecutionForm";
import { ExecutionHistoryList } from "../components/reasoners/ExecutionHistoryList";
import {
  ExecutionQueue,
  type ExecutionQueueRef,
  type QueuedExecution,
} from "../components/reasoners/ExecutionQueue";
import { FormattedOutput } from "../components/reasoners/FormattedOutput";
import { PerformanceChart } from "../components/reasoners/PerformanceChart";
import { StatusIndicator } from "../components/reasoners/StatusIndicator";
import { Alert } from "../components/ui/alert";
import { Button } from "../components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "../components/ui/card";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "../components/ui/tabs";
import { SegmentedControl } from "../components/ui/segmented-control";
import type { SegmentedControlOption } from "../components/ui/segmented-control";
import { ResponsiveGrid } from "@/components/layout/ResponsiveGrid";
import { reasonersApi } from "../services/reasonersApi";
import { normalizeExecutionStatus } from "../utils/status";
import type { ExecutionHistory, PerformanceMetrics } from "../types/execution";
import type { ReasonerWithNode } from "../types/reasoners";
import { generateExampleData, validateFormData } from "../utils/schemaUtils";

const RESULT_VIEW_OPTIONS: ReadonlyArray<SegmentedControlOption> = [
  { value: "formatted", label: "Formatted", icon: View },
  { value: "json", label: "JSON", icon: Code },
] as const;

export function ReasonerDetailPage() {
  const { fullReasonerId } = useParams<{ fullReasonerId: string }>();

  const [reasoner, setReasoner] = useState<ReasonerWithNode | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Multiple execution state
  const [selectedExecution, setSelectedExecution] =
    useState<QueuedExecution | null>(null);
  const [resultViewMode, setResultViewMode] = useState<"formatted" | "json">(
    "formatted"
  );
  const executionQueueRef = useRef<ExecutionQueueRef | null>(null);

  // History and metrics
  const [history, setHistory] = useState<ExecutionHistory | null>(null);
  const [metrics, setMetrics] = useState<PerformanceMetrics | null>(null);

  // Form state
  const [formData, setFormData] = useState<ExecutionFormData>({});
  const [validationErrors, setValidationErrors] = useState<string[]>([]);
  const [isExecuting, setIsExecuting] = useState(false);

  const loadReasonerDetails = useCallback(async () => {
    if (!fullReasonerId) return;

    try {
      setLoading(true);
      const data = await reasonersApi.getReasonerDetails(fullReasonerId);
      setReasoner(data);

      // Initialize form with example data if schema is available
      if (data.input_schema) {
        const exampleData = generateExampleData(data.input_schema);
        setFormData({ input: exampleData });
      }
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to load reasoner details"
      );
    } finally {
      setLoading(false);
    }
  }, [fullReasonerId]);

  const loadMetrics = useCallback(async () => {
    if (!fullReasonerId) return;
    try {
      const data = await reasonersApi.getPerformanceMetrics(fullReasonerId);
      setMetrics(data);
    } catch (err) {
      console.error("Failed to load metrics:", err);
    }
  }, [fullReasonerId]);

  const loadHistory = useCallback(async () => {
    if (!fullReasonerId) return;
    try {
      const data = await reasonersApi.getExecutionHistory(
        fullReasonerId,
        1,
        10
      );
      setHistory(data);
    } catch (err) {
      console.error("Failed to load history:", err);
    }
  }, [fullReasonerId]);

  useEffect(() => {
    loadReasonerDetails();
    loadMetrics();
    loadHistory();
  }, [loadHistory, loadMetrics, loadReasonerDetails]);

  const handleExecute = () => {
    if (!reasoner || !fullReasonerId || !executionQueueRef.current) return;

    // Validate form data
    if (reasoner.input_schema) {
      const validation = validateFormData(
        formData.input,
        reasoner.input_schema
      );
      if (!validation.isValid) {
        setValidationErrors(validation.errors);
        return;
      }
    }

    setValidationErrors([]);
    setIsExecuting(true);

    // Add execution to queue
    const executionId = executionQueueRef.current.addExecution(
      formData.input || {}
    );
    console.log("Added execution to queue:", executionId);

    // Reset executing state after a brief delay
    setTimeout(() => setIsExecuting(false), 500);
  };

  const handleExecutionComplete = (execution: QueuedExecution) => {
    // Refresh history and metrics after execution
    loadHistory();
    loadMetrics();

    // Auto-select the completed execution to show results
    setSelectedExecution(execution);
  };

  const handleExecutionSelect = (execution: QueuedExecution | null) => {
    // Handle selection from the execution queue
    setSelectedExecution(execution);
  };

  const handleCopyCommand = () => {
    if (!reasoner || !fullReasonerId) return;

    const baseUrl = window.location.origin;
    const curlCommand = `curl -X POST ${baseUrl}/api/v1/execute/${encodeURIComponent(
      fullReasonerId
    )} \\
  -H "Content-Type: application/json" \\
  -d '${JSON.stringify({ input: formData.input || {} }, null, 2)}'`;

    navigator.clipboard.writeText(curlCommand);
  };

  const toggleResultView = () => {
    setResultViewMode((prev) => (prev === "formatted" ? "json" : "formatted"));
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="flex flex-col items-center gap-3">
          <InProgress className="h-8 w-8 animate-spin text-muted-foreground" />
          <p className="text-body-small">
            Loading reasoner details...
          </p>
        </div>
      </div>
    );
  }

  if (error || !reasoner) {
    return (
      <div className="container mx-auto px-4 py-8">
        <Alert className="max-w-md mx-auto border-red-200 bg-red-50">
          <div className="flex items-center gap-2">
            <div className="h-4 w-4 rounded-full bg-red-500" />
            <div>
              <h3 className="font-semibold text-red-900">Error</h3>
              <p className="text-sm text-red-700">
                {error || "Reasoner not found"}
              </p>
            </div>
          </div>
        </Alert>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-display">
              {reasoner.name}
            </h2>
            <p className="text-body">
              {reasoner.description || "No description available"}
            </p>
          </div>
          <div className="flex items-center gap-4">
            <StatusIndicator
              status={
                reasoner.node_status === "active"
                  ? "online"
                  : reasoner.node_status === "inactive"
                  ? "offline"
                  : "unknown"
              }
            />
            <Button
              variant="outline"
              size="sm"
              onClick={handleCopyCommand}
              className="flex items-center gap-2"
            >
              <Copy className="h-4 w-4" />
              Copy cURL
            </Button>
          </div>
        </div>
        <div className="flex items-center gap-4 text-body">
          <span>Node: {reasoner.node_id}</span>
          <span>•</span>
          <span>ID: {fullReasonerId}</span>
          <span>•</span>
          <DIDIdentityBadge nodeId={reasoner.node_id} showDID={true} />
        </div>
        {reasoner.tags && reasoner.tags.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-2">
            {reasoner.tags.map((tag) => (
              <Badge
                key={`${reasoner.reasoner_id}-${tag}`}
                variant="secondary"
                className="text-xs"
              >
                #{tag}
              </Badge>
            ))}
          </div>
        )}
      </div>

      {/* Quick Stats */}
      {metrics && (
        <ResponsiveGrid preset="quarters" gap="md" align="start">
          <Card className="card-elevated">
            <CardContent className="p-4">
              <div className="flex items-center gap-2 mb-2">
                <Time className="h-4 w-4 text-text-tertiary" />
                <span className="text-caption">Avg Response</span>
              </div>
              <p className="text-heading-3">{metrics.avg_response_time_ms}ms</p>
            </CardContent>
          </Card>

          <Card className="card-elevated">
            <CardContent className="p-4">
              <div className="flex items-center gap-2 mb-2">
                <CheckmarkFilled className="h-4 w-4 text-status-success" />
                <span className="text-caption">Success Rate</span>
              </div>
              <p className="text-heading-3">
                {(metrics.success_rate * 100).toFixed(1)}%
              </p>
            </CardContent>
          </Card>

          <Card className="card-elevated">
            <CardContent className="p-4">
              <div className="flex items-center gap-2 mb-2">
                <Analytics className="h-4 w-4 text-text-tertiary" />
                <span className="text-caption">Total Executions</span>
              </div>
              <p className="text-heading-3">{metrics.total_executions}</p>
            </CardContent>
          </Card>

          <Card className="card-elevated">
            <CardContent className="p-4">
              <div className="flex items-center gap-2 mb-2">
                <Activity className="h-4 w-4 text-text-tertiary" />
                <span className="text-caption">Last 24h</span>
              </div>
              <p className="text-heading-3">{metrics.executions_last_24h}</p>
            </CardContent>
          </Card>
        </ResponsiveGrid>
      )}

      {/* Responsive Layout */}
      <ResponsiveGrid columns={{ base: 1, lg: 12 }} gap="md" align="start">
        {/* Left Panel - Input & Configuration */}
        <div className="lg:col-span-5 space-y-6">
          {/* Input Form */}
          <Card className="card-elevated">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Play className="h-5 w-5" />
                Execute Reasoner
              </CardTitle>
              <CardDescription>
                Provide input data and execute the reasoner
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <ExecutionForm
                schema={reasoner.input_schema}
                formData={formData}
                onChange={setFormData}
                validationErrors={validationErrors}
              />

              <Button
                onClick={handleExecute}
                className="w-full"
                size="lg"
                disabled={isExecuting}
              >
                {isExecuting ? (
                  <>
                    <InProgress className="h-4 w-4 mr-2 animate-spin" />
                    Executing...
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4 mr-2" />
                    Execute Reasoner
                  </>
                )}
              </Button>
            </CardContent>
          </Card>

          {/* Schema Information */}
          <Tabs defaultValue="input" className="space-y-4">
            <TabsList variant="underline" className="grid w-full grid-cols-2">
              <TabsTrigger value="input" variant="underline">Input Schema</TabsTrigger>
              <TabsTrigger value="output" variant="underline">Output Schema</TabsTrigger>
            </TabsList>

            <TabsContent value="input">
              <Card className="card-elevated">
                <CardHeader>
                  <CardTitle className="text-sm">Input Schema</CardTitle>
                  <CardDescription>
                    Expected input format for this reasoner
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {reasoner.input_schema ? (
                    <pre className="bg-bg-secondary p-4 rounded-lg text-sm overflow-auto scrollbar-thin border border-border-secondary">
                      {JSON.stringify(reasoner.input_schema, null, 2)}
                    </pre>
                  ) : (
                    <p className="text-text-tertiary">
                      No input schema available
                    </p>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="output">
              <Card className="card-elevated">
                <CardHeader>
                  <CardTitle className="text-sm">Output Schema</CardTitle>
                  <CardDescription>
                    Expected output format from this reasoner
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {reasoner.output_schema ? (
                    <pre className="bg-bg-secondary p-4 rounded-lg text-sm overflow-auto scrollbar-thin border border-border-secondary">
                      {JSON.stringify(reasoner.output_schema, null, 2)}
                    </pre>
                  ) : (
                    <p className="text-text-tertiary">
                      No output schema available
                    </p>
                  )}
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>

        {/* Right Panel - Execution & Results */}
        <div className="lg:col-span-7 space-y-6">
          {/* Execution Queue */}
          <ExecutionQueue
            reasonerId={fullReasonerId!}
            onExecutionComplete={handleExecutionComplete}
            onExecutionSelect={handleExecutionSelect}
            ref={executionQueueRef}
          />

          {/* Selected Execution Result */}
          {selectedExecution && (
            <Card className="card-elevated">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle>Execution Result</CardTitle>
                    <CardDescription>
                      Result from execution: {selectedExecution.inputSummary}
                    </CardDescription>
                  </div>
                  {normalizeExecutionStatus(selectedExecution.status) === "succeeded" &&
                    selectedExecution.result !== undefined &&
                    selectedExecution.result !== null && (
                      <SegmentedControl
                        value={resultViewMode}
                        onValueChange={(mode) =>
                          setResultViewMode(mode as typeof resultViewMode)
                        }
                        options={RESULT_VIEW_OPTIONS}
                        size="sm"
                        optionClassName="min-w-[120px]"
                      />
                    )}
                </div>
              </CardHeader>
              <CardContent>
                {normalizeExecutionStatus(selectedExecution.status) === "succeeded" &&
                selectedExecution.result !== undefined &&
                selectedExecution.result !== null ? (
                  <FormattedOutput
                    data={selectedExecution.result}
                    showRaw={resultViewMode === "json"}
                    onToggleView={toggleResultView}
                    executionId={selectedExecution.id}
                    duration={selectedExecution.duration}
                    status={normalizeExecutionStatus(selectedExecution.status)}
                    hideHeader={true}
                  />
                ) : selectedExecution.status === "failed" ? (
                  <div className="text-center py-8">
                    <div className="h-8 w-8 mx-auto mb-3 rounded-full bg-status-error-bg flex items-center justify-center">
                      <div className="h-4 w-4 rounded-full bg-status-error" />
                    </div>
                    <p className="text-status-error font-medium mb-1">
                      Execution Failed
                    </p>
                    <p className="text-body-small">{selectedExecution.error}</p>
                  </div>
                ) : (
                  <div className="flex items-center justify-center py-8">
                    <InProgress className="h-6 w-6 animate-spin text-text-tertiary mr-3" />
                    <span className="text-text-secondary">Executing...</span>
                  </div>
                )}
              </CardContent>
            </Card>
          )}

          {/* Activity & Performance */}
          <Tabs defaultValue="activity" className="space-y-4">
            <TabsList variant="underline" className="grid w-full grid-cols-2">
              <TabsTrigger value="activity" variant="underline" className="gap-2">
                <Activity className="h-4 w-4" />
                Activity
              </TabsTrigger>
              <TabsTrigger
                value="performance"
                variant="underline"
                className="gap-2"
              >
                <Analytics className="h-4 w-4" />
                Performance
              </TabsTrigger>
            </TabsList>

            <TabsContent value="activity">
              <Card className="card-elevated">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Activity className="h-5 w-5" />
                    Recent Executions
                  </CardTitle>
                  <CardDescription>
                    Latest execution attempts and their results
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <ExecutionHistoryList
                    history={history}
                    onLoadMore={() => {
                      // TODO: Implement pagination
                    }}
                  />
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="performance">
              <Card className="card-elevated">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Analytics className="h-5 w-5" />
                    Performance Metrics
                  </CardTitle>
                  <CardDescription>
                    Response times and success rates over time
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <PerformanceChart metrics={metrics} />
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </ResponsiveGrid>
    </div>
  );
}
