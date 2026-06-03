import {
  CheckmarkFilled,
  ChevronDown,
  ChevronRight,
  Chip,
  Code,
  Copy,
  Settings,
} from "@/components/ui/icon-bridge";
import React, { useState } from "react";
import { Badge } from "../../ui/badge";
import { Button } from "../../ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "../../ui/card";
import { JsonHighlightedPre } from "@/components/ui/json-syntax-highlight";

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

interface TechnicalSectionProps {
  node: WorkflowNodeData;
  details?: NodeDetails;
  onCopy: (text: string, label: string) => void;
  copySuccess: string | null;
}

export function TechnicalSection({
  node,
  details,
  onCopy,
  copySuccess,
}: TechnicalSectionProps) {
  const [expandedSections, setExpandedSections] = useState<Set<string>>(
    new Set()
  );

  const toggleSection = (section: string) => {
    const newExpanded = new Set(expandedSections);
    if (newExpanded.has(section)) {
      newExpanded.delete(section);
    } else {
      newExpanded.add(section);
    }
    setExpandedSections(newExpanded);
  };

  const technicalDetails = [
    {
      label: "Workflow ID",
      value: node.workflow_id,
      copyable: true,
      icon: <Code size={14} className="text-muted-foreground" />,
    },
    {
      label: "Agent Node ID",
      value: node.agent_node_id,
      copyable: true,
      icon: <Chip size={14} className="text-muted-foreground" />,
    },
    {
      label: "Reasoner ID",
      value: node.reasoner_id,
      copyable: true,
      icon: <Settings size={14} className="text-muted-foreground" />,
    },
  ];

  const renderTechnicalDetail = (detail: (typeof technicalDetails)[0]) => (
    <div key={detail.label} className="flex items-center justify-between py-2">
      <div className="flex items-center gap-2 flex-1 min-w-0">
        {detail.icon}
        <div className="min-w-0 flex-1">
          <span className="text-sm text-muted-foreground/70 block">
            {detail.label}
          </span>
          <span className="text-sm text-muted-foreground font-mono truncate block">
            {detail.value}
          </span>
        </div>
      </div>
      {detail.copyable && (
        <Button
          variant="ghost"
          size="sm"
          onClick={() => onCopy(detail.value, detail.label)}
          className="h-6 w-6 p-0 hover:bg-muted flex-shrink-0 ml-2"
          title={`Copy ${detail.label}`}
        >
          {copySuccess === detail.label ? (
            <CheckmarkFilled
              size={12}
              className="text-status-success"
            />
          ) : (
            <Copy size={12} className="text-muted-foreground" />
          )}
        </Button>
      )}
    </div>
  );

  const renderExpandableSection = (
    title: string,
    content: React.ReactNode,
    sectionKey: string,
    badge?: string
  ) => {
    const isExpanded = expandedSections.has(sectionKey);

    return (
      <div className="border border-border rounded-lg">
        <button
          onClick={() => toggleSection(sectionKey)}
          className="w-full p-3 flex items-center justify-between hover:bg-muted transition-colors"
        >
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-foreground">
              {title}
            </span>
            {badge && (
              <Badge variant="secondary" className="text-xs">
                {badge}
              </Badge>
            )}
          </div>
          {isExpanded ? (
            <ChevronDown
              size={16}
              className="text-muted-foreground"
            />
          ) : (
            <ChevronRight
              size={16}
              className="text-muted-foreground"
            />
          )}
        </button>
        {isExpanded && (
          <div className="border-t border-border p-3 bg-muted/80">
            {content}
          </div>
        )}
      </div>
    );
  };

  return (
    <Card className="bg-muted border-border">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-medium text-foreground flex items-center gap-2">
          <Settings size={16} className="text-muted-foreground" />
          Technical Details
        </CardTitle>
      </CardHeader>

      <CardContent className="pt-0 space-y-4">
        {/* Basic Technical Information */}
        <div className="space-y-1">
          {technicalDetails.map(renderTechnicalDetail)}
        </div>

        {/* Performance Metrics */}
        {details?.performance_metrics && (
          <div className="pt-4 border-t border-border">
            <h5 className="text-xs font-medium text-foreground mb-3 flex items-center gap-2">
              <Chip size={14} className="text-muted-foreground" />
              Performance Metrics
            </h5>
            <div className="grid grid-cols-2 gap-4 text-xs">
              <div>
                <span className="text-muted-foreground/70 block mb-1">
                  Response Time
                </span>
                <span className="text-muted-foreground font-mono">
                  {details.performance_metrics.response_time_ms}ms
                </span>
              </div>
              {details.performance_metrics.tokens_used && (
                <div>
                  <span className="text-muted-foreground/70 block mb-1">
                    Tokens Used
                  </span>
                  <span className="text-muted-foreground font-mono">
                    {details.performance_metrics.tokens_used.toLocaleString()}
                  </span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Cost Information */}
        {details?.cost && (
          <div className="pt-4 border-t border-border">
            <div className="flex items-center justify-between">
              <h5 className="text-xs font-medium text-foreground">
                Execution Cost
              </h5>
              <span className="text-sm font-mono font-semibold text-foreground">
                ${details.cost.toFixed(4)}
              </span>
            </div>
          </div>
        )}

        {/* Memory Updates */}
        {details?.memory_updates &&
          details.memory_updates.length > 0 &&
          renderExpandableSection(
            "Memory Updates",
            <div className="space-y-2">
              {details.memory_updates.map((update: any, index: number) => (
                <div
                  key={index}
                  className="p-2 bg-muted rounded border border-border"
                >
                  <div className="flex items-center gap-2 mb-1">
                    <Badge variant="outline" className="text-xs">
                      {update.action}
                    </Badge>
                    <span className="text-sm text-muted-foreground font-mono">
                      {update.scope}/{update.key}
                    </span>
                  </div>
                  {update.value != null && (
                    <JsonHighlightedPre
                      data={update.value}
                      className="mt-1 text-sm"
                    />
                  )}
                </div>
              ))}
            </div>,
            "memory-updates",
            details.memory_updates.length.toString()
          )}

        {/* Raw Node Data */}
        {renderExpandableSection(
          "Raw Node Data",
          <JsonHighlightedPre data={node} className="text-sm overflow-x-auto" />,
          "raw-node-data"
        )}

        {/* Raw Details Data */}
        {details &&
          renderExpandableSection(
            "Raw Details Data",
            <JsonHighlightedPre data={details} className="text-sm overflow-x-auto" />,
            "raw-details-data"
          )}

        {/* Debug Information */}
        <div className="pt-4 border-t border-border">
          <h5 className="text-xs font-medium text-foreground mb-3">
            Debug Information
          </h5>
          <div className="grid grid-cols-1 gap-2 text-xs">
            <div className="flex justify-between">
              <span className="text-muted-foreground/70">
                Node Type
              </span>
              <span className="text-muted-foreground font-mono">
                Reasoner Execution
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground/70">
                Workflow Depth
              </span>
              <span className="text-muted-foreground font-mono">
                Level {node.workflow_depth}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground/70">
                Has Input Data
              </span>
              <span className="text-muted-foreground font-mono">
                {details?.input ? "Yes" : "No"}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground/70">
                Has Output Data
              </span>
              <span className="text-muted-foreground font-mono">
                {details?.output ? "Yes" : "No"}
              </span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
