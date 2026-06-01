import { useState } from 'react';
import {
  ChevronDown,
  ChevronRight,
  Copy,
  Code,
  Document,
  Maximize,
  Launch
} from '@/components/ui/icon-bridge';
import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '../ui/collapsible';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface SmartStringRendererProps {
  content: string;
  label: string;
  path: string[];
  onOpenModal?: () => void;
  maxInlineHeight?: number;
  className?: string;
}

export function SmartStringRenderer({
  content,
  onOpenModal,
  maxInlineHeight = 200,
  className = ""
}: SmartStringRendererProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(content);
  };

  // Content type detection
  const isMarkdown = content.includes('#') || content.includes('*') || content.includes('`') ||
                    content.includes('[') && content.includes('](') || content.includes('```');
  const isLongText = content.length > 150;
  const hasMultipleLines = content.includes('\n');
  const isUrl = /^https?:\/\//.test(content.trim());
  const isCode = content.includes('{') && content.includes('}') ||
                content.includes('function') || content.includes('const ') ||
                content.includes('import ') || content.includes('export ');

  // Determine display mode
  const shouldShowCompact = isLongText || hasMultipleLines;
  const previewLength = 100;
  const preview = content.length > previewLength ? content.substring(0, previewLength) + '...' : content;

  // Render URL
  if (isUrl) {
    return (
      <div className={`flex items-center gap-2 ${className}`}>
        <Launch className="h-4 w-4 text-muted-foreground flex-shrink-0" />
        <div className="flex-1 min-w-0">
          <a
            href={content}
            target="_blank"
            rel="noopener noreferrer"
            className="text-accent-secondary hover:text-accent-primary underline decoration-accent-secondary/30 hover:decoration-accent-primary transition-colors text-sm truncate block"
            title={content}
          >
            {content}
          </a>
        </div>
        <Button
          variant="ghost"
          size="sm"
          onClick={handleCopy}
          className="h-6 w-6 p-0 flex-shrink-0"
        >
          <Copy className="h-3 w-3" />
        </Button>
      </div>
    );
  }

  // Render compact view for long content
  if (shouldShowCompact) {
    return (
      <div className={`space-y-2 ${className}`}>
        <Collapsible open={isExpanded} onOpenChange={setIsExpanded}>
          <div className="flex items-start gap-2">
            <CollapsibleTrigger asChild>
              <Button
                variant="ghost"
                size="sm"
                className="h-6 w-6 p-0 flex-shrink-0 mt-0.5"
              >
                {isExpanded ? (
                  <ChevronDown className="h-3 w-3" />
                ) : (
                  <ChevronRight className="h-3 w-3" />
                )}
              </Button>
            </CollapsibleTrigger>

            <div className="flex-1 min-w-0">
              {/* Preview */}
              <div className="flex items-start gap-2">
                <div className="flex-1 min-w-0">
                  <div className="text-sm">
                    {isExpanded ? (
                      <span className="font-medium text-foreground">Expanded</span>
                    ) : (
                      <div className="truncate">{preview}</div>
                    )}
                  </div>
                  <div className="flex items-center gap-2 mt-1">
                    <Badge variant="secondary" className="text-xs">
                      {content.length.toLocaleString()} chars
                    </Badge>
                    {isMarkdown && (
                      <Badge variant="secondary" className="text-xs">
                        <Document className="h-3 w-3 mr-1" />
                        Markdown
                      </Badge>
                    )}
                    {isCode && (
                      <Badge variant="secondary" className="text-xs">
                        <Code className="h-3 w-3 mr-1" />
                        Code
                      </Badge>
                    )}
                  </div>
                </div>

                <div className="flex items-center gap-1 flex-shrink-0">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={handleCopy}
                    className="h-6 w-6 p-0"
                  >
                    <Copy className="h-3 w-3" />
                  </Button>
                  {onOpenModal && (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={onOpenModal}
                      className="h-6 w-6 p-0"
                    >
                      <Maximize className="h-3 w-3" />
                    </Button>
                  )}
                </div>
              </div>

              {/* Expanded content */}
              <CollapsibleContent>
                <div
                  className="mt-3 border border-border rounded-lg bg-muted/30 overflow-hidden"
                  style={{ maxHeight: maxInlineHeight }}
                >
                  <div className="p-4 overflow-auto">
                    {isMarkdown ? (
                      <div className="prose prose-sm max-w-none prose-invert prose-headings:text-foreground prose-p:text-muted-foreground prose-strong:text-foreground prose-code:text-foreground prose-code:bg-muted prose-code:px-1 prose-code:py-0.5 prose-code:rounded prose-pre:bg-muted prose-pre:border prose-pre:border-border">
                        <ReactMarkdown
                          remarkPlugins={[remarkGfm]}
                          components={{
                            h1: ({children}) => <h1 className="text-xl font-semibold mb-3 text-foreground">{children}</h1>,
                            h2: ({children}) => <h2 className="text-base font-semibold mb-2 text-foreground">{children}</h2>,
                            h3: ({children}) => <h3 className="text-sm font-medium mb-2 text-foreground">{children}</h3>,
                            p: ({children}) => <p className="mb-2 text-sm leading-relaxed">{children}</p>,
                            ul: ({children}) => <ul className="list-disc list-inside mb-2 text-sm space-y-1">{children}</ul>,
                            ol: ({children}) => <ol className="list-decimal list-inside mb-2 text-sm space-y-1">{children}</ol>,
                            li: ({children}) => <li className="leading-relaxed">{children}</li>,
                            code: ({children, className}) => {
                              const isInline = !className;
                              return isInline ? (
                                <code className="bg-muted px-1 py-0.5 rounded text-xs font-mono text-foreground border border-border">
                                  {children}
                                </code>
                              ) : (
                                <code className={className}>{children}</code>
                              );
                            },
                            pre: ({children}) => (
                              <pre className="json-hl-root bg-muted p-3 rounded text-xs overflow-auto border border-border mb-2 font-mono">
                                {children}
                              </pre>
                            ),
                            blockquote: ({children}) => (
                              <blockquote className="border-l-4 border-accent-primary pl-3 italic text-muted-foreground mb-2 bg-muted/30 py-1 rounded-r">
                                {children}
                              </blockquote>
                            ),
                            a: ({href, children}) => (
                              <a
                                href={href}
                                className="text-accent-secondary hover:text-accent-primary underline decoration-accent-secondary/30 hover:decoration-accent-primary transition-colors"
                                target="_blank"
                                rel="noopener noreferrer"
                              >
                                {children}
                              </a>
                            ),
                            table: ({children}) => (
                              <div className="overflow-auto mb-2 border border-border rounded">
                                <table className="min-w-full text-xs">{children}</table>
                              </div>
                            ),
                            th: ({children}) => (
                              <th className="border-b border-border px-2 py-1 text-left text-xs font-medium text-foreground bg-muted">
                                {children}
                              </th>
                            ),
                            td: ({children}) => (
                              <td className="border-b border-border px-2 py-1 text-sm text-muted-foreground">
                                {children}
                              </td>
                            ),
                          }}
                        >
                          {content}
                        </ReactMarkdown>
                      </div>
                    ) : isCode ? (
                      <pre className="json-hl-root text-xs font-mono whitespace-pre-wrap">
                        {content}
                      </pre>
                    ) : (
                      <div className="text-sm whitespace-pre-wrap leading-relaxed">
                        {content}
                      </div>
                    )}
                  </div>
                </div>
              </CollapsibleContent>
            </div>
          </div>
        </Collapsible>
      </div>
    );
  }

  // Render simple inline content
  return (
    <div className={`flex items-center gap-2 ${className}`}>
      <div className="flex-1 min-w-0">
        {isMarkdown ? (
          <div className="prose prose-sm max-w-none prose-invert prose-p:text-muted-foreground prose-p:mb-0 prose-strong:text-foreground prose-code:text-foreground prose-code:bg-muted prose-code:px-1 prose-code:py-0.5 prose-code:rounded">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                p: ({children}) => <span className="text-sm">{children}</span>,
                code: ({children}) => (
                  <code className="bg-muted px-1 py-0.5 rounded text-xs font-mono text-foreground border border-border">
                    {children}
                  </code>
                ),
                strong: ({children}) => <strong className="text-foreground">{children}</strong>,
                em: ({children}) => <em className="text-foreground">{children}</em>,
              }}
            >
              {content}
            </ReactMarkdown>
          </div>
        ) : (
          <span className="text-sm truncate">
            {content}
          </span>
        )}
      </div>

      <div className="flex items-center gap-1 flex-shrink-0">
        <Button
          variant="ghost"
          size="sm"
          onClick={handleCopy}
          className="h-6 w-6 p-0"
        >
          <Copy className="h-3 w-3" />
        </Button>
        {onOpenModal && (
          <Button
            variant="ghost"
            size="sm"
            onClick={onOpenModal}
            className="h-6 w-6 p-0"
          >
            <Maximize className="h-3 w-3" />
          </Button>
        )}
      </div>
    </div>
  );
}
