import { AlertTriangle } from "@/components/ui/icon-bridge";
import type { WorkflowExecution } from "../../types/executions";
import { CollapsibleSection } from "./CollapsibleSection";
import { CopyButton } from "../ui/copy-button";
import { JsonHighlightedPre } from "../ui/json-syntax-highlight";

interface RedesignedErrorPanelProps {
  execution: WorkflowExecution;
}
export function RedesignedErrorPanel({ execution }: RedesignedErrorPanelProps) {
  if (!execution.error_message) {
    return null;
  }

  // Try to parse the error message as JSON to extract structured info
  let errorData: any = execution.error_message;
  let isStructuredError = false;

  try {
    errorData = JSON.parse(execution.error_message);
    isStructuredError = true;
  } catch {
    // Keep as string if not valid JSON
  }

  const badge = (
    <span className="text-xs text-red-600 bg-red-50 dark:bg-red-950/50 px-2 py-0.5 rounded">
      Error
    </span>
  );

  return (
    <CollapsibleSection
      title="Error Details"
      icon={AlertTriangle}
      badge={badge}
      defaultOpen={true}
      className="border-red-200 dark:border-red-800"
      headerClassName="bg-red-50/50 dark:bg-red-950/20"
    >
      <div className="p-4 space-y-4">
        {/* Error Summary */}
        <div className="flex items-start justify-between gap-4">
          <div className="flex-1">
            <h4 className="text-sm font-medium text-red-900 dark:text-red-100 mb-2">
              Execution Failed
            </h4>
            {isStructuredError && errorData.message ? (
              <p className="text-sm text-red-700 dark:text-red-300 leading-relaxed">
                {errorData.message}
              </p>
            ) : (
              <p className="text-sm text-red-700 dark:text-red-300 leading-relaxed">
                {typeof errorData === "string" ? errorData : "An unknown error occurred"}
              </p>
            )}
          </div>
          <CopyButton
            value={execution.error_message}
            variant="ghost"
            size="icon"
            className="h-6 w-6 p-0 hover:bg-muted/80 [&_svg]:h-3 [&_svg]:w-3"
            tooltip="Copy error message"
          />
        </div>

        {/* Structured Error Details */}
        {isStructuredError && (
          <div className="space-y-3">
            {errorData.type && (
              <div>
                <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                  Error Type
                </span>
                <p className="text-sm text-foreground mt-1 font-mono">
                  {errorData.type}
                </p>
              </div>
            )}

            {errorData.code && (
              <div>
                <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                  Error Code
                </span>
                <p className="text-sm text-foreground mt-1 font-mono">
                  {errorData.code}
                </p>
              </div>
            )}

            {errorData.stack && (
              <div>
                <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                  Stack Trace
                </span>
                <pre className="text-xs font-mono text-muted-foreground mt-2 p-3 bg-muted/50 rounded-md overflow-x-auto whitespace-pre-wrap">
                  {errorData.stack}
                </pre>
              </div>
            )}

            {errorData.context && (
              <div>
                <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                  Context
                </span>
                <JsonHighlightedPre
                  data={errorData.context}
                  className="mt-2 overflow-x-auto rounded-md bg-muted/50 p-3 text-xs"
                />
              </div>
            )}
          </div>
        )}

        {/* Raw Error Message (if not structured) */}
        {!isStructuredError && execution.error_message.length > 200 && (
          <div>
            <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
              Full Error Message
            </span>
            <pre className="text-xs font-mono text-muted-foreground mt-2 p-3 bg-muted/50 rounded-md overflow-x-auto whitespace-pre-wrap max-h-60 overflow-y-auto">
              {execution.error_message}
            </pre>
          </div>
        )}

        {/* Debugging Tips */}
        <div className="border-t border-border/50 pt-4">
          <h5 className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2">
            Debugging Tips
          </h5>
          <ul className="text-sm text-muted-foreground space-y-1">
            <li>• Check the input data format and validation</li>
            <li>• Review the reasoner implementation for this error type</li>
            <li>• Verify agent node connectivity and dependencies</li>
            <li>• Check system logs for additional context</li>
          </ul>
        </div>
      </div>
    </CollapsibleSection>
  );
}
