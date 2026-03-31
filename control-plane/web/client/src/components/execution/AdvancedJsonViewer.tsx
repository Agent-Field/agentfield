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
  Search,
} from "@/components/ui/icon-bridge";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { CopyButton } from "../ui/copy-button";

interface AdvancedJsonViewerProps {
  data: any;
  maxHeight?: string;
  className?: string;
  searchable?: boolean;
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
  switch (type) {
    case "markdown":
      return <FileText className="w-3 h-3 text-blue-500" />;
    case "json":
      return <Braces className="w-3 h-3 text-purple-500" />;
    case "args":
      return <List className="w-3 h-3 text-green-500" />;
    case "kwargs":
      return <Hash className="w-3 h-3 text-orange-500" />;
    case "string":
      return <Quote className="w-3 h-3 text-green-600" />;
    case "number":
      return <Type className="w-3 h-3 text-blue-600" />;
    case "code":
      return <FileText className="w-3 h-3 text-red-500" />;
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
      className="prose prose-sm max-w-none text-foreground bg-blue-50/30 dark:bg-blue-950/20 p-2 rounded border border-blue-200/50 dark:border-blue-800/50"
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
      return <span className="text-slate-500 italic">null</span>;
    }

    if (typeof data === "boolean") {
      return (
        <span className="text-blue-600 dark:text-blue-400 font-medium">
          {String(data)}
        </span>
      );
    }

    if (typeof data === "number") {
      return (
        <span className="text-purple-600 dark:text-purple-400 font-medium">
          {data}
        </span>
      );
    }

    if (typeof data === "string") {
      const typeIcon = getTypeIcon(contentType);
      const isLongString = data.length > 100;
      const displayValue =
        isLongString && !showPreview ? `${data.slice(0, 100)}...` : data;

      return (
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            {typeIcon}
            <span className="text-green-600 dark:text-green-400">
              "{displayValue}"
            </span>
            {isLongString && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setShowPreview(!showPreview)}
                className="h-5 w-5 p-0"
                title={showPreview ? "Hide preview" : "Show preview"}
              >
                <Eye className="h-3 w-3" />
              </Button>
            )}
          </div>

          {showPreview && contentType === "markdown" && (
            <MarkdownPreview content={data} />
          )}

          {showPreview && contentType === "code" && (
            <pre className="bg-gray-100 dark:bg-gray-800 p-2 rounded text-sm font-mono overflow-x-auto">
              <code>{data}</code>
            </pre>
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
        className="flex items-start gap-2 group py-0.5"
        style={{ paddingLeft: `${level * 16}px` }}
      >
        <div className="w-4" /> {/* Spacer for alignment */}
        {keyName && (
          <span className="text-blue-700 dark:text-blue-300 font-medium shrink-0">
            "{keyName}":
          </span>
        )}
        <div className="flex-1 min-w-0">{renderValue()}</div>
        <CopyButton
          value={String(data)}
          variant="ghost"
          size="icon"
          className="h-5 w-5 p-0 opacity-0 transition-opacity group-hover:opacity-100 [&_svg]:h-3 [&_svg]:w-3"
          tooltip={`Copy ${currentPath.join(".")}`}
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
        className="flex items-center gap-2 group py-0.5 cursor-pointer hover:bg-muted/30 rounded px-1"
        style={{ paddingLeft: `${level * 16}px` }}
        onClick={toggleExpanded}
      >
        <button className="flex items-center justify-center w-4 h-4 hover:bg-muted rounded">
          {isExpanded ? (
            <ChevronDown className="w-3 h-3 text-muted-foreground" />
          ) : (
            <ChevronRight className="w-3 h-3 text-muted-foreground" />
          )}
        </button>

        {keyName && (
          <span className="text-blue-700 dark:text-blue-300 font-medium">
            "{keyName}":
          </span>
        )}

        <div className="flex items-center gap-2">
          {getTypeIcon(contentType)}
          <span className="text-muted-foreground">
            {Array.isArray(data) ? "[" : "{"}
          </span>

          {!isExpanded && (
            <>
              <span className="text-body-small italic">
                {getCollectionInfo()}
              </span>
              <span className="text-muted-foreground">
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
          tooltip={`Copy ${currentPath.join(".") || "JSON"}`}
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
            <span className="text-muted-foreground">
              {Array.isArray(data) ? "]" : "}"}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

export function AdvancedJsonViewer({
  data,
  maxHeight = "600px",
  className = "",
  searchable = true,
}: AdvancedJsonViewerProps) {
  const [searchTerm, setSearchTerm] = useState("");

  return (
    <div
      className={`border border-border rounded-lg bg-background ${className}`}
    >
      {searchable && (
        <div className="border-b border-border p-3">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <Input
              placeholder="Search keys and values..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-10"
            />
          </div>
        </div>
      )}

      <div
        className="overflow-auto p-4 font-mono text-sm scrollbar-thin scrollbar-track-transparent scrollbar-thumb-border hover:scrollbar-thumb-muted-foreground"
        style={{ maxHeight }}
      >
        <JsonNode
          data={data}
          level={0}
          path={[]}
          searchTerm={searchTerm}
        />
      </div>
    </div>
  );
}
