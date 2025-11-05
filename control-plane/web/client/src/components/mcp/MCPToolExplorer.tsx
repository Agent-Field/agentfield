import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import { useMode } from "@/contexts/ModeContext";
import { cn } from "@/lib/utils";
import type { MCPTool } from "@/types/agentfield";
import {
  ChevronDown,
  ChevronUp,
  Code,
  Information,
  Play,
  Search,
  Tools,
} from "@/components/ui/icon-bridge";
import { useMemo, useState } from "react";

interface MCPToolExplorerProps {
  tools: MCPTool[];
  serverAlias: string;
  nodeId: string;
  onTestTool?: (toolName: string) => void;
  isLoading?: boolean;
  className?: string;
}

/**
 * Display available tools for an MCP server
 * Shows tool schemas and descriptions with mode-aware information display
 */
export function MCPToolExplorer({
  tools,
  serverAlias,
  nodeId: _nodeId,
  onTestTool,
  isLoading = false,
  className,
}: MCPToolExplorerProps) {
  const { mode } = useMode();
  const [searchQuery, setSearchQuery] = useState("");
  const [expandedTools, setExpandedTools] = useState<Set<string>>(new Set());

  const isDeveloperMode = mode === "developer";

  // Filter tools based on search query
  const filteredTools = useMemo(() => {
    if (!searchQuery) return tools;

    const query = searchQuery.toLowerCase();
    return tools.filter((tool) => {
      const description = tool.description?.toLowerCase() ?? '';
      return (
        tool.name.toLowerCase().includes(query) ||
        description.includes(query)
      );
    });
  }, [tools, searchQuery]);

  const toggleToolExpansion = (toolName: string) => {
    const newExpanded = new Set(expandedTools);
    if (newExpanded.has(toolName)) {
      newExpanded.delete(toolName);
    } else {
      newExpanded.add(toolName);
    }
    setExpandedTools(newExpanded);
  };

  const formatSchema = (schema: unknown) => {
    try {
      return JSON.stringify(schema, null, 2);
    } catch {
      return "Invalid schema format";
    }
  };

  const getSchemaProperties = (schema: any) => {
    if (!schema || typeof schema !== "object") return [];

    const properties = schema.properties || {};
    const required = schema.required || [];

    return Object.entries(properties).map(([name, prop]: [string, any]) => ({
      name,
      type: prop.type || "unknown",
      description: prop.description || "",
      required: required.includes(name),
      default: prop.default,
      enum: prop.enum,
    }));
  };

  const renderToolCard = (tool: MCPTool) => {
    const isExpanded = expandedTools.has(tool.name);
    const schemaProperties = getSchemaProperties(tool.input_schema);

    return (
      <Card
        key={tool.name}
        className="transition-all duration-200 hover:shadow-md"
      >
        <Collapsible
          open={isExpanded}
          onOpenChange={() => toggleToolExpansion(tool.name)}
        >
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3 flex-1">
                <Tools className="w-5 h-5 text-muted-foreground" />
                <div className="flex-1">
                  <CardTitle className="text-heading-3">
                    {tool.name}
                  </CardTitle>
                  <p className="text-body-small mt-1 line-clamp-2">
                    {tool.description ?? 'No description provided.'}
                  </p>
                </div>
              </div>

              <div className="flex items-center gap-2">
                {schemaProperties.length > 0 && (
                  <Badge variant="secondary" className="text-xs">
                    {schemaProperties.length} params
                  </Badge>
                )}

                {isDeveloperMode && onTestTool && (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={(e) => {
                      e.stopPropagation();
                      onTestTool(tool.name);
                    }}
                    disabled={isLoading}
                  >
                    <Play className="w-4 h-4" />
                    Test
                  </Button>
                )}

                <CollapsibleTrigger asChild>
                  <Button variant="ghost" size="sm" className="p-1">
                    {isExpanded ? (
                      <ChevronUp className="w-4 h-4" />
                    ) : (
                      <ChevronDown className="w-4 h-4" />
                    )}
                  </Button>
                </CollapsibleTrigger>
              </div>
            </div>
          </CardHeader>

          <CollapsibleContent>
            <CardContent className="pt-0">
              {/* Tool Description - Expanded */}
              {tool.description && (
                <div className="mb-4">
                  <h4 className="text-sm font-medium mb-2 flex items-center gap-2">
                    <Information className="w-4 h-4" />
                    Description
                  </h4>
                  <p className="text-body-small bg-gray-50 p-3 rounded-md">
                    {tool.description}
                  </p>
                </div>
              )}

              {/* Parameters */}
              {schemaProperties.length > 0 && (
                <div className="mb-4">
                  <h4 className="text-sm font-medium mb-3">Parameters</h4>
                  <div className="space-y-3">
                    {schemaProperties.map((param) => (
                      <div
                        key={param.name}
                        className="border rounded-md p-3 bg-gray-50"
                      >
                        <div className="flex items-center gap-2 mb-1">
                          <code className="text-sm font-mono bg-white px-2 py-1 rounded border">
                            {param.name}
                          </code>
                          <Badge
                            variant={
                              param.required ? "destructive" : "secondary"
                            }
                            className="text-xs"
                          >
                            {param.type}
                          </Badge>
                          {param.required && (
                            <Badge variant="outline" className="text-xs">
                              required
                            </Badge>
                          )}
                        </div>

                        {param.description && (
                          <p className="text-body-small mb-2">
                            {param.description}
                          </p>
                        )}

                        <div className="flex gap-4 text-body-small">
                          {param.default !== undefined && (
                            <span>
                              Default:{" "}
                              <code>{JSON.stringify(param.default)}</code>
                            </span>
                          )}
                          {param.enum && (
                            <span>
                              Options: <code>{param.enum.join(", ")}</code>
                            </span>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Raw Schema - Developer Mode Only */}
              {isDeveloperMode && (
                <div>
                  <h4 className="text-sm font-medium mb-2 flex items-center gap-2">
                    <Code className="w-4 h-4" />
                    Input Schema
                  </h4>
                  <pre className="text-xs bg-gray-900 text-gray-100 p-3 rounded-md overflow-x-auto">
                    {formatSchema(tool.input_schema ?? {})}
                  </pre>
                </div>
              )}
            </CardContent>
          </CollapsibleContent>
        </Collapsible>
      </Card>
    );
  };

  return (
    <Card className={cn("w-full", className)}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Tools className="w-5 h-5 text-muted-foreground" />
            <CardTitle>Available Tools</CardTitle>
            <Badge variant="secondary">{tools.length} tools</Badge>
          </div>

          <div className="text-body-small">
            Server: <span className="font-medium">{serverAlias}</span>
          </div>
        </div>

        {/* Search */}
        {tools.length > 0 && (
          <div className="relative mt-4">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <Input
              placeholder="Search tools..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10"
            />
          </div>
        )}
      </CardHeader>

      <CardContent>
        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <div className="w-6 h-6 animate-spin rounded-full border-2 border-current border-t-transparent" />
            <span className="ml-2 text-muted-foreground">Loading tools...</span>
          </div>
        ) : filteredTools.length === 0 ? (
          <div className="text-center py-8 text-muted-foreground">
            {searchQuery
              ? "No tools match your search"
              : tools.length === 0
              ? "No tools available for this server"
              : "No tools found"}
          </div>
        ) : (
          <div className="space-y-4">
            {filteredTools.map(renderToolCard)}

            {/* Expand/Collapse All - Developer Mode */}
            {isDeveloperMode && filteredTools.length > 1 && (
              <div className="flex justify-center pt-4 border-t">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => {
                    if (expandedTools.size === filteredTools.length) {
                      setExpandedTools(new Set());
                    } else {
                      setExpandedTools(
                        new Set(filteredTools.map((t) => t.name))
                      );
                    }
                  }}
                >
                  {expandedTools.size === filteredTools.length
                    ? "Collapse All"
                    : "Expand All"}
                </Button>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
