import { Code, Copy, Time, View } from "@/components/ui/icon-bridge";
import { useState } from "react";
import { Badge } from "../ui/badge";
import { Button } from "../ui/button";
import { Card, CardContent } from "../ui/card";
import { UnifiedJsonViewer } from "@/components/ui/UnifiedJsonViewer";
import { JsonHighlightedPre } from "@/components/ui/json-syntax-highlight";
import type { CanonicalStatus } from "../../utils/status";
import { getStatusLabel } from "../../utils/status";

interface FormattedOutputProps {
  data: any;
  showRaw?: boolean;
  onToggleView?: () => void;
  executionId?: string;
  duration?: number;
  status?: CanonicalStatus;
  hideHeader?: boolean;
}

export function FormattedOutput({
  data,
  showRaw = false,
  onToggleView,
  executionId,
  duration,
  status = "succeeded",
  hideHeader = false,
}: FormattedOutputProps) {
  // When hideHeader is true, use showRaw prop to determine view mode
  // When hideHeader is false, use internal state
  const [internalViewMode, setInternalViewMode] = useState<
    "formatted" | "json"
  >("formatted");
  const viewMode = hideHeader
    ? showRaw
      ? "json"
      : "formatted"
    : internalViewMode;

  if (!data) {
    return (
      <div className="text-center py-8 text-muted-foreground">
        <Code className="h-8 w-8 mx-auto mb-2 opacity-50" />
        <p>No output data available</p>
      </div>
    );
  }

  const copyAllData = () => {
    navigator.clipboard.writeText(JSON.stringify(data, null, 2));
  };

  if (showRaw) {
    return (
      <div className="space-y-4">
        {/* Header with toggle */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <h3 className="font-medium">Raw JSON Output</h3>
            {status === "succeeded" && (
              <Badge
                variant="outline"
                className="bg-green-50 text-green-700 border-green-200"
              >
                {getStatusLabel(status)}
              </Badge>
            )}
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" onClick={copyAllData}>
              <Copy className="h-4 w-4 mr-2" />
              Copy
            </Button>
          </div>
        </div>

        {/* Raw JSON Display */}
        <div className="relative">
          <JsonHighlightedPre
            data={data}
            className="max-h-96 overflow-auto rounded-lg border bg-muted p-4 text-sm"
          />
        </div>

        {/* Execution Info */}
        {(executionId || duration) && (
          <div className="text-sm text-muted-foreground flex items-center gap-4">
            {duration && (
              <span className="flex items-center gap-1">
                <Time className="h-3 w-3" />
                Completed in {duration}ms
              </span>
            )}
            {executionId && (
              <span className="flex items-center gap-1">
                <Copy className="h-3 w-3" />
                {executionId.slice(0, 12)}...
              </span>
            )}
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Header with toggle - only show if not hidden */}
      {!hideHeader && (
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <h3 className="font-medium">Execution Result</h3>
            {status === "succeeded" && (
              <Badge
                variant="outline"
                className="bg-green-50 text-green-700 border-green-200"
              >
                {getStatusLabel(status)}
              </Badge>
            )}
          </div>
          <div className="flex items-center gap-2">
            {/* View Mode Toggle */}
            <div className="flex items-center gap-1 bg-muted rounded-lg p-1">
              <Button
                variant={viewMode === "formatted" ? "default" : "ghost"}
                size="sm"
                onClick={() => setInternalViewMode("formatted")}
                className="h-7 px-3 text-xs"
              >
                <View className="h-3 w-3 mr-1" />
                Formatted
              </Button>
              <Button
                variant={viewMode === "json" ? "default" : "ghost"}
                size="sm"
                onClick={() => setInternalViewMode("json")}
                className="h-7 px-3 text-xs"
              >
                <Code className="h-3 w-3 mr-1" />
                JSON
              </Button>
            </div>

            {onToggleView && (
              <Button variant="outline" size="sm" onClick={onToggleView}>
                <Code className="h-4 w-4 mr-2" />
                Raw JSON
              </Button>
            )}
            <Button variant="outline" size="sm" onClick={copyAllData}>
              <Copy className="h-4 w-4 mr-2" />
              Copy All
            </Button>
          </div>
        </div>
      )}

      {/* Content based on view mode */}
      {viewMode === "json" ? (
        <Card>
          <CardContent className="p-4">
            <JsonHighlightedPre
              data={data}
              className="max-h-96 overflow-auto rounded-lg border bg-muted p-4 text-sm"
            />
          </CardContent>
        </Card>
      ) : (
        <Card>
          <CardContent className="p-4">
            <UnifiedJsonViewer data={data} />
          </CardContent>
        </Card>
      )}

      {/* Execution Info */}
      {(executionId || duration) && (
        <div className="text-sm text-muted-foreground flex items-center gap-4 pt-2 border-t">
          {duration && (
            <span className="flex items-center gap-1">
              <Time className="h-3 w-3" />
              Completed in {duration}ms
              {duration < 500 && <span className="text-green-600">(Fast)</span>}
              {duration >= 500 && duration < 2000 && (
                <span className="text-yellow-600">(Normal)</span>
              )}
              {duration >= 2000 && <span className="text-red-600">(Slow)</span>}
            </span>
          )}
          {executionId && (
            <span className="flex items-center gap-1">
              <Copy className="h-3 w-3" />
              {executionId.slice(0, 12)}...
            </span>
          )}
        </div>
      )}
    </div>
  );
}
