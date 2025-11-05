import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { useMode } from "@/contexts/ModeContext";
import { cn } from "@/lib/utils";
import type {
  MCPTool,
  MCPToolTestRequest,
  MCPToolTestResponse,
} from "@/types/haxen";
import {
  CheckmarkFilled,
  Copy,
  ErrorFilled,
  Play,
  RecentlyViewed,
  Stop,
  Time,
  TrashCan,
} from "@/components/ui/icon-bridge";
import { useEffect, useState } from "react";

interface MCPToolTesterProps {
  tool: MCPTool;
  serverAlias: string;
  nodeId: string;
  onTestTool?: (request: MCPToolTestRequest) => Promise<MCPToolTestResponse>;
  className?: string;
}

interface TestHistory {
  id: string;
  timestamp: string;
  request: MCPToolTestRequest;
  response: MCPToolTestResponse;
}

/**
 * Interactive tool testing interface
 * Dynamic form generation based on tool schema with execution history
 */
export function MCPToolTester({
  tool,
  serverAlias,
  nodeId,
  onTestTool,
  className,
}: MCPToolTesterProps) {
  const { mode } = useMode();
  const [parameters, setParameters] = useState<Record<string, any>>({});
  const [isExecuting, setIsExecuting] = useState(false);
  const [testHistory, setTestHistory] = useState<TestHistory[]>([]);
  const [selectedHistoryId, setSelectedHistoryId] = useState<string | null>(
    null
  );
  const [timeoutMs, setTimeoutMs] = useState(30000);

  const isDeveloperMode = mode === "developer";

  // Parse tool schema to get parameter definitions
  const getSchemaProperties = (schema: any) => {
    if (!schema || typeof schema !== "object") return [];

    const properties = schema.properties || {};
    const required = schema.required || [];

    return Object.entries(properties).map(([name, prop]: [string, any]) => ({
      name,
      type: prop.type || "string",
      description: prop.description || "",
      required: required.includes(name),
      default: prop.default,
      enum: prop.enum,
      minimum: prop.minimum,
      maximum: prop.maximum,
      pattern: prop.pattern,
    }));
  };

  const schemaProperties = getSchemaProperties(tool.input_schema);

  // Initialize parameters with defaults
  useEffect(() => {
    const defaultParams: Record<string, any> = {};
    schemaProperties.forEach((param) => {
      if (param.default !== undefined) {
        defaultParams[param.name] = param.default;
      } else if (param.type === "boolean") {
        defaultParams[param.name] = false;
      } else if (param.type === "number" || param.type === "integer") {
        defaultParams[param.name] = param.minimum || 0;
      } else {
        defaultParams[param.name] = "";
      }
    });
    setParameters(defaultParams);
  }, [tool.name]);

  const handleParameterChange = (name: string, value: any) => {
    setParameters((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const validateParameters = () => {
    const errors: string[] = [];

    schemaProperties.forEach((param) => {
      const value = parameters[param.name];

      if (
        param.required &&
        (value === undefined || value === null || value === "")
      ) {
        errors.push(`${param.name} is required`);
      }

      if (param.type === "number" || param.type === "integer") {
        const numValue = Number(value);
        if (isNaN(numValue)) {
          errors.push(`${param.name} must be a valid number`);
        } else {
          if (param.minimum !== undefined && numValue < param.minimum) {
            errors.push(`${param.name} must be at least ${param.minimum}`);
          }
          if (param.maximum !== undefined && numValue > param.maximum) {
            errors.push(`${param.name} must be at most ${param.maximum}`);
          }
        }
      }

      if (param.pattern && typeof value === "string") {
        const regex = new RegExp(param.pattern);
        if (!regex.test(value)) {
          errors.push(`${param.name} does not match required pattern`);
        }
      }
    });

    return errors;
  };

  const handleExecute = async () => {
    if (!onTestTool || isExecuting) return;

    const validationErrors = validateParameters();
    if (validationErrors.length > 0) {
      alert("Validation errors:\n" + validationErrors.join("\n"));
      return;
    }

    setIsExecuting(true);

    try {
      const request: MCPToolTestRequest = {
        node_id: nodeId,
        server_alias: serverAlias,
        tool_name: tool.name,
        parameters,
        timeout_ms: timeoutMs,
      };

      const response = await onTestTool(request);

      const historyEntry: TestHistory = {
        id: Date.now().toString(),
        timestamp: new Date().toISOString(),
        request,
        response,
      };

      setTestHistory((prev) => [historyEntry, ...prev.slice(0, 9)]); // Keep last 10
      setSelectedHistoryId(historyEntry.id);
    } catch (error) {
      console.error("Tool execution failed:", error);
    } finally {
      setIsExecuting(false);
    }
  };

  const loadFromHistory = (historyEntry: TestHistory) => {
    setParameters(historyEntry.request.parameters);
    setTimeoutMs(historyEntry.request.timeout_ms || 30000);
    setSelectedHistoryId(historyEntry.id);
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  const clearHistory = () => {
    setTestHistory([]);
    setSelectedHistoryId(null);
  };

  const renderParameterInput = (param: any) => {
    const value = parameters[param.name];

    switch (param.type) {
      case "boolean":
        return (
          <div className="flex items-center space-x-2">
            <Checkbox
              id={param.name}
              checked={value || false}
              onCheckedChange={(checked) =>
                handleParameterChange(param.name, checked)
              }
            />
            <Label htmlFor={param.name} className="text-sm font-medium">
              {param.name}
              {param.required && <span className="text-red-500 ml-1">*</span>}
            </Label>
          </div>
        );

      case "number":
      case "integer":
        return (
          <div className="space-y-2">
            <Label htmlFor={param.name} className="text-sm font-medium">
              {param.name}
              {param.required && <span className="text-red-500 ml-1">*</span>}
            </Label>
            <Input
              id={param.name}
              type="number"
              value={value || ""}
              onChange={(e) =>
                handleParameterChange(param.name, e.target.value)
              }
              min={param.minimum}
              max={param.maximum}
              step={param.type === "integer" ? 1 : "any"}
              placeholder={param.description}
            />
          </div>
        );

      default:
        if (param.enum) {
          return (
            <div className="space-y-2">
              <Label htmlFor={param.name} className="text-sm font-medium">
                {param.name}
                {param.required && <span className="text-red-500 ml-1">*</span>}
              </Label>
              <Select
                value={value || ""}
                onValueChange={(newValue) =>
                  handleParameterChange(param.name, newValue)
                }
              >
                <SelectTrigger>
                  <SelectValue placeholder={`Select ${param.name}`} />
                </SelectTrigger>
                <SelectContent>
                  {param.enum.map((option: string) => (
                    <SelectItem key={option} value={option}>
                      {option}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          );
        }

        return (
          <div className="space-y-2">
            <Label htmlFor={param.name} className="text-sm font-medium">
              {param.name}
              {param.required && <span className="text-red-500 ml-1">*</span>}
            </Label>
            <Input
              id={param.name}
              value={value || ""}
              onChange={(e) =>
                handleParameterChange(param.name, e.target.value)
              }
              placeholder={param.description}
            />
          </div>
        );
    }
  };

  const selectedHistory = testHistory.find((h) => h.id === selectedHistoryId);

  return (
    <Card className={cn("w-full", className)}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Play className="w-5 h-5 text-muted-foreground" />
            <CardTitle>Test Tool: {tool.name}</CardTitle>
          </div>

          <div className="text-body-small">
            Server: <span className="font-medium">{serverAlias}</span>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Tool Description */}
        {tool.description && (
          <div className="p-3 bg-blue-50 border border-blue-200 rounded-md">
            <p className="text-sm text-blue-800">{tool.description}</p>
          </div>
        )}

        {/* Parameters Form */}
        {schemaProperties.length > 0 && (
          <div className="space-y-4">
            <h3 className="text-heading-3">Parameters</h3>
            <div className="grid gap-4">
              {schemaProperties.map((param) => (
                <div key={param.name} className="space-y-1">
                  {renderParameterInput(param)}
                  {param.description && (
                    <p className="text-body-small">
                      {param.description}
                    </p>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Execution Settings */}
        {isDeveloperMode && (
          <div className="space-y-4">
            <h3 className="text-heading-3">Execution Settings</h3>
            <div className="space-y-2">
              <Label htmlFor="timeout" className="text-sm font-medium">
                Timeout (ms)
              </Label>
              <Input
                id="timeout"
                type="number"
                value={timeoutMs}
                onChange={(e) => setTimeoutMs(Number(e.target.value))}
                min={1000}
                max={300000}
                step={1000}
              />
            </div>
          </div>
        )}

        {/* Execute Button */}
        <div className="flex gap-2">
          <Button
            onClick={handleExecute}
            disabled={!onTestTool || isExecuting}
            className="flex-1"
          >
            {isExecuting ? (
              <>
                <div className="w-4 h-4 animate-spin rounded-full border-2 border-current border-t-transparent mr-2" />
                Executing...
              </>
            ) : (
              <>
                <Play className="w-4 h-4 mr-2" />
                Execute Tool
              </>
            )}
          </Button>

          {isExecuting && (
            <Button
              variant="outline"
              onClick={() => setIsExecuting(false)}
              disabled
            >
              <Stop className="w-4 h-4" />
            </Button>
          )}
        </div>

        {/* Test Results */}
        {selectedHistory && (
          <div className="space-y-4">
            <Separator />
            <div className="flex items-center justify-between">
              <h3 className="text-heading-3">Test Result</h3>
              <div className="flex items-center gap-2">
                <Badge
                  variant={
                    selectedHistory.response.success ? "default" : "destructive"
                  }
                  className="flex items-center gap-1"
                >
                  {selectedHistory.response.success ? (
                    <CheckmarkFilled className="w-3 h-3" />
                  ) : (
                    <ErrorFilled className="w-3 h-3" />
                  )}
                  {selectedHistory.response.success ? "Success" : "Error"}
                </Badge>
                <Badge variant="secondary" className="flex items-center gap-1">
                  <Time className="w-3 h-3" />
                  {selectedHistory.response.execution_time_ms}ms
                </Badge>
              </div>
            </div>

            <div className="space-y-3">
              {selectedHistory.response.success ? (
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <Label className="text-sm font-medium">Result</Label>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() =>
                        copyToClipboard(
                          JSON.stringify(
                            selectedHistory.response.result,
                            null,
                            2
                          )
                        )
                      }
                    >
                      <Copy className="w-4 h-4" />
                    </Button>
                  </div>
                  <pre className="text-xs bg-gray-900 text-gray-100 p-3 rounded-md overflow-x-auto max-h-64">
                    {JSON.stringify(selectedHistory.response.result, null, 2)}
                  </pre>
                </div>
              ) : (
                <div>
                  <Label className="text-sm font-medium text-red-600">
                    Error
                  </Label>
                  <div className="mt-2 p-3 bg-red-50 border border-red-200 rounded-md">
                    <p className="text-sm text-red-800">
                      {selectedHistory.response.error}
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Test History */}
        {testHistory.length > 0 && (
          <div className="space-y-4">
            <Separator />
            <div className="flex items-center justify-between">
              <h3 className="text-heading-3 flex items-center gap-2">
                <RecentlyViewed className="w-5 h-5" />
                Test History
              </h3>
              <Button variant="ghost" size="sm" onClick={clearHistory}>
                <TrashCan className="w-4 h-4" />
                Clear
              </Button>
            </div>

            <div className="space-y-2 max-h-64 overflow-y-auto">
              {testHistory.map((entry) => (
                <div
                  key={entry.id}
                  className={cn(
                    "p-3 border rounded-md cursor-pointer transition-colors",
                    selectedHistoryId === entry.id
                      ? "border-blue-500 bg-blue-50"
                      : "border-gray-200 hover:border-gray-300"
                  )}
                  onClick={() => loadFromHistory(entry)}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      {entry.response.success ? (
                        <CheckmarkFilled className="w-4 h-4 text-green-600" />
                      ) : (
                        <ErrorFilled className="w-4 h-4 text-red-600" />
                      )}
                      <span className="text-sm font-medium">
                        {new Date(entry.timestamp).toLocaleTimeString()}
                      </span>
                    </div>
                    <Badge variant="secondary" className="text-xs">
                      {entry.response.execution_time_ms}ms
                    </Badge>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
