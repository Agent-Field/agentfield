import { useState, useMemo } from "react";
import {
  ChevronRight,
  ChevronDown,
  FileText,
  Hash,
  Type,
  List,
  Braces,
  Quote,
  Eye,
  EyeOff,
  Search,
  Maximize2,
  Minimize2,
} from "@/components/ui/icon-bridge";
import { Button } from "./button";
import { Input } from "./input";
import { CopyButton } from "./copy-button";
import { cn } from "../../lib/utils";

interface UnifiedJsonViewerProps {
  data: any;
  title?: string;
  className?: string;
  maxHeight?: string;
  collapsible?: boolean;
  showCopyButton?: boolean;
  searchable?: boolean;
  showHeader?: boolean;
}

interface JsonNodeProps {
  data: any;
  keyName?: string;
  level: number;
  path: string[];
  searchTerm?: string;
  onCopy?: (value: string, path: string[]) => void;
}

function detectContentType(
  value: any
):
  | "markdown"
  | "json"
  | "url"
  | "email"
  | "code"
  | "args"
  | "kwargs"
  | "string"
  | "number"
  | "boolean"
  | "null" {
  if (value === null) return "null";
  if (typeof value === "boolean") return "boolean";
  if (typeof value === "number") return "number";

  if (typeof value === "string") {
    // Check for URLs
    if (/^https?:\/\//.test(value)) return "url";

    // Check for email
    if (/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value)) return "email";

    // Check for markdown patterns
    if (
      value.includes("**") ||
      value.includes("*") ||
      value.includes("`") ||
      value.includes("#") ||
      value.includes("[") ||
      value.includes("](")
    ) {
      return "markdown";
    }

    // Check for code-like content
    if (
      value.includes("function") ||
      value.includes("def ") ||
      value.includes("class ") ||
      value.includes("import ") ||
      value.includes("from ")
    ) {
      return "code";
    }

    return "string";
  }

  if (typeof value === "object") {
    // Check for args/kwargs patterns
    if (Array.isArray(value)) return "args";
    if (value && typeof value === "object") {
      const keys = Object.keys(value);
      if (
        keys.some(
          (key) =>
            key.startsWith("_") || key.includes("arg") || key.includes("param")
        )
      ) {
        return "kwargs";
      }
    }
    return "json";
  }

  return "string";
}

function getTypeIcon(type: string) {
  const iconClass = "w-3 h-3";
  switch (type) {
    case "markdown":
      return <FileText className={cn(iconClass, "json-hl-key")} />;
    case "json":
      return <Braces className={cn(iconClass, "json-hl-key")} />;
    case "args":
      return <List className={cn(iconClass, "json-hl-punct")} />;
    case "kwargs":
      return <Hash className={cn(iconClass, "json-hl-number")} />;
    case "string":
      return <Quote className={cn(iconClass, "json-hl-string")} />;
    case "number":
      return <Type className={cn(iconClass, "json-hl-number")} />;
    case "code":
      return <FileText className={cn(iconClass, "json-hl-null")} />;
    case "url":
      return <FileText className={cn(iconClass, "json-hl-key")} />;
    case "email":
      return <Type className={cn(iconClass, "json-hl-string")} />;
    default:
      return null;
  }
}

function MarkdownPreview({ content }: { content: string }) {
  const formattedContent = useMemo(() => {
    return content
      .replace(/\*\*(.*?)\*\*/g, '<strong class="font-semibold">$1</strong>')
      .replace(/\*(.*?)\*/g, '<em class="italic">$1</em>')
      .replace(
        /`(.*?)`/g,
        '<code class="bg-muted px-1 rounded text-sm font-mono text-foreground">$1</code>'
      )
      .replace(/\n\n/g, '</p><p class="mt-2">')
      .replace(/\n/g, "<br>")
      .replace(
        /\[(.*?)\]\((.*?)\)/g,
        '<a href="$2" class="text-blue-500 hover:underline" target="_blank" rel="noopener">$1</a>'
      );
  }, [content]);

  return (
    <div
      className="prose prose-sm max-w-none text-foreground bg-blue-50/30 dark:bg-blue-950/20 p-3 rounded-md border border-blue-200/50 dark:border-blue-800/50 mt-2"
      dangerouslySetInnerHTML={{ __html: `<p>${formattedContent}</p>` }}
    />
  );
}

function JsonNode({
  data,
  keyName,
  level,
  path,
  searchTerm,
  onCopy,
}: JsonNodeProps) {
  const [isExpanded, setIsExpanded] = useState(level < 2); // Auto-expand first 2 levels
  const [showPreview, setShowPreview] = useState(false);

  const contentType = detectContentType(data);
  const isExpandable = typeof data === "object" && data !== null;
  const currentPath = keyName ? [...path, keyName] : path;

  // Search functionality
  const matchesSearch = useMemo(() => {
    if (!searchTerm) return true;
    const searchLower = searchTerm.toLowerCase();
    const keyMatches = keyName?.toLowerCase().includes(searchLower);
    const valueMatches =
      typeof data === "string" && data.toLowerCase().includes(searchLower);
    return keyMatches || valueMatches;
  }, [searchTerm, keyName, data]);

  if (!matchesSearch && typeof data !== "object") return null;

  const renderValue = () => {
    if (data === null) {
      return <span className="json-hl-null italic">null</span>;
    }

    if (typeof data === "boolean") {
      return (
        <span className="json-hl-boolean font-medium">{String(data)}</span>
      );
    }

    if (typeof data === "number") {
      return <span className="json-hl-number font-medium">{data}</span>;
    }

    if (typeof data === "string") {
      const typeIcon = getTypeIcon(contentType);
      const isLongString = data.length > 100;
      const displayValue =
        isLongString && !showPreview ? `${data.slice(0, 100)}...` : data;

      return (
        <div className="space-y-2">
          <div className="flex items-center gap-2 flex-wrap">
            {typeIcon}
            <span className="json-hl-string break-all">"{displayValue}"</span>
            {isLongString && (
              <Button
                variant="ghost"
                size="sm"
                onClick={(e) => {
                  e.stopPropagation();
                  setShowPreview(!showPreview);
                }}
                className="h-5 w-5 p-0 ml-1"
                title={showPreview ? "Hide preview" : "Show preview"}
              >
                {showPreview ? (
                  <EyeOff className="h-3 w-3" />
                ) : (
                  <Eye className="h-3 w-3" />
                )}
              </Button>
            )}
            {contentType === "url" && (
              <Button
                variant="ghost"
                size="sm"
                onClick={(e) => {
                  e.stopPropagation();
                  window.open(data, "_blank", "noopener,noreferrer");
                }}
                className="h-5 w-5 p-0 ml-1"
                title="Open URL"
              >
                <FileText className="h-3 w-3" />
              </Button>
            )}
          </div>

          {showPreview && contentType === "markdown" && (
            <MarkdownPreview content={data} />
          )}

          {showPreview && contentType === "code" && (
            <pre className="json-hl-root bg-muted p-3 rounded-md text-sm font-mono overflow-x-auto mt-2 border border-border">
              <code>{data}</code>
            </pre>
          )}

          {showPreview &&
            isLongString &&
            !["markdown", "code"].includes(contentType) && (
              <div className="bg-muted p-3 rounded-md text-sm mt-2 border break-all">
                {data}
              </div>
            )}
        </div>
      );
    }

    return null;
  };

  const getCollectionInfo = () => {
    if (Array.isArray(data)) {
      return `Array(${data.length})`;
    }
    if (typeof data === "object" && data !== null) {
      const keys = Object.keys(data);
      return `Object(${keys.length})`;
    }
    return "";
  };

  const toggleExpanded = () => {
    setIsExpanded(!isExpanded);
  };

  if (!isExpandable) {
    return (
      <div
        className="flex items-start gap-2 group py-1 hover:bg-muted/20 rounded px-1"
        style={{ paddingLeft: `${level * 16}px` }}
      >
        <div className="w-4" /> {/* Spacer for alignment */}
        {keyName && (
          <span className="json-hl-key font-medium shrink-0">"{keyName}":</span>
        )}
        <div className="flex-1 min-w-0">{renderValue()}</div>
        <CopyButton
          value={String(data)}
          variant="ghost"
          size="icon"
          className="h-5 w-5 p-0 opacity-0 transition-opacity group-hover:opacity-100 [&_svg]:h-3 [&_svg]:w-3"
          tooltip={`Copy ${currentPath.length > 0 ? currentPath.join(".") : "value"}`}
          onClick={(event) => event.stopPropagation()}
        />
      </div>
    );
  }

  const entries = Array.isArray(data)
    ? data.map((item, index) => [index.toString(), item])
    : Object.entries(data);

  return (
    <div className="space-y-1">
      <div
        className="flex items-center gap-2 group py-1 cursor-pointer hover:bg-muted/30 rounded px-1"
        style={{ paddingLeft: `${level * 16}px` }}
        onClick={toggleExpanded}
      >
        <button className="flex items-center justify-center w-4 h-4 hover:bg-muted rounded transition-colors">
          {isExpanded ? (
            <ChevronDown className="w-3 h-3 text-muted-foreground" />
          ) : (
            <ChevronRight className="w-3 h-3 text-muted-foreground" />
          )}
        </button>

        {keyName && (
          <span className="json-hl-key font-medium">"{keyName}":</span>
        )}

        <div className="flex items-center gap-2">
          {getTypeIcon(contentType)}
          <span className="json-hl-punct">
            {Array.isArray(data) ? "[" : "{"}
          </span>

          {!isExpanded && (
            <>
              <span className="text-sm text-muted-foreground italic">
                {getCollectionInfo()}
              </span>
              <span className="json-hl-punct">
                {Array.isArray(data) ? "]" : "}"}
              </span>
            </>
          )}
        </div>

        <CopyButton
          value={JSON.stringify(data, null, 2)}
          variant="ghost"
          size="icon"
          className="h-5 w-5 p-0 opacity-0 transition-opacity group-hover:opacity-100 [&_svg]:h-3 [&_svg]:w-3"
          tooltip={`Copy ${currentPath.length > 0 ? currentPath.join(".") : "JSON"}`}
          onClick={(event) => event.stopPropagation()}
        />
      </div>

      {isExpanded && (
        <div className="space-y-1">
          {entries.map(([key, value]) => (
            <JsonNode
              key={key}
              data={value}
              keyName={Array.isArray(data) ? undefined : key}
              level={level + 1}
              path={currentPath}
              searchTerm={searchTerm}
              onCopy={onCopy}
            />
          ))}

          <div
            className="flex items-center"
            style={{ paddingLeft: `${(level + 1) * 16}px` }}
          >
            <div className="w-4" />
            <span className="json-hl-punct">
              {Array.isArray(data) ? "]" : "}"}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

export function UnifiedJsonViewer({
  data,
  title,
  className,
  maxHeight = "400px",
  collapsible = true,
  showCopyButton = true,
  searchable = true,
  showHeader = true,
}: UnifiedJsonViewerProps) {
  const [searchTerm, setSearchTerm] = useState("");
  const [isExpanded, setIsExpanded] = useState(false);
  const jsonString = JSON.stringify(data, null, 2);

  const isEmpty =
    !data || (typeof data === "object" && Object.keys(data).length === 0);

  const isFlexibleHeight = maxHeight === "none";

  return (
    <div
      className={cn(
        "border border-border rounded-lg overflow-hidden bg-background flex flex-col",
        isFlexibleHeight && "h-full",
        className
      )}
    >
      {/* Header */}
      {showHeader && (title || showCopyButton || (searchable && !isEmpty)) && (
        <div className="flex-shrink-0 border-b border-border bg-muted/20">
          {(title || showCopyButton) && (
            <div className="flex items-center justify-between p-3">
              {title && (
                <h4 className="text-sm font-medium text-foreground flex items-center gap-2">
                  {getTypeIcon(detectContentType(data))}
                  {title}
                </h4>
              )}
              <div className="flex items-center gap-2">
                {showCopyButton && !isEmpty && (
                  <CopyButton
                    value={jsonString}
                    variant="ghost"
                    size="icon"
                    className="h-5 w-5 p-0 opacity-100 [&_svg]:h-3 [&_svg]:w-3"
                    tooltip="Copy JSON"
                    onClick={(event) => event.stopPropagation()}
                  />
                )}
                {collapsible && !isEmpty && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setIsExpanded(!isExpanded)}
                    className="h-5 w-5 p-0"
                    title={isExpanded ? "Collapse" : "Expand"}
                  >
                    {isExpanded ? (
                      <Minimize2 className="h-3 w-3 text-muted-foreground" />
                    ) : (
                      <Maximize2 className="h-3 w-3 text-muted-foreground" />
                    )}
                  </Button>
                )}
              </div>
            </div>
          )}

          {searchable && !isEmpty && (
            <div className="px-3 pb-3">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <Input
                  placeholder="Search keys and values..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-10 h-8"
                />
              </div>
            </div>
          )}
        </div>
      )}

      {/* Content */}
      <div
        className={cn(
          "bg-background scrollbar-thin scrollbar-track-transparent scrollbar-thumb-border hover:scrollbar-thumb-muted-foreground",
          isEmpty ? "p-6" : "p-4",
          isFlexibleHeight ? "flex-1 min-h-0 overflow-auto" : "overflow-auto"
        )}
        style={!isExpanded && !isFlexibleHeight ? { maxHeight } : undefined}
      >
        {isEmpty ? (
          <div className="text-center text-muted-foreground">
            <div className="flex flex-col items-center gap-3">
              <Braces className="h-8 w-8 text-muted-foreground/50" />
              <p className="text-sm">No data available</p>
            </div>
          </div>
        ) : (
          <div className="font-mono text-sm">
            <JsonNode
              data={data}
              level={0}
              path={[]}
              searchTerm={searchTerm}
            />
          </div>
        )}
      </div>
    </div>
  );
}
