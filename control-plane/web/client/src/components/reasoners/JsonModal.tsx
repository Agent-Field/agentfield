import { useState } from 'react';
import {
  Close,
  Copy,
  Download,
  Code,
  View,
  ChevronRight
} from '@/components/ui/icon-bridge';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '../ui/dialog';
import { Button } from '../ui/button';
import { SegmentedControl, type SegmentedControlOption } from '../ui/segmented-control';
import { Badge } from '../ui/badge';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { JsonHighlightedPre } from '@/components/ui/json-syntax-highlight';

interface JsonModalProps {
  isOpen: boolean;
  onClose: () => void;
  content: any;
  path: string[];
  title: string;
}

export function JsonModal({ isOpen, onClose, content, path, title }: JsonModalProps) {
  const [viewMode, setViewMode] = useState<'formatted' | 'raw'>('formatted');
  const viewOptions: ReadonlyArray<SegmentedControlOption> = [
    { value: 'formatted', label: 'Formatted', icon: View },
    { value: 'raw', label: 'Raw', icon: Code },
  ];

  const handleCopy = () => {
    const textToCopy = typeof content === 'string' ? content : JSON.stringify(content, null, 2);
    navigator.clipboard.writeText(textToCopy);
  };

  const handleDownload = () => {
    const textToDownload = typeof content === 'string' ? content : JSON.stringify(content, null, 2);
    const blob = new Blob([textToDownload], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${path.join('_')}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const renderContent = () => {
    if (viewMode === 'raw') {
      return (
        <JsonHighlightedPre
          text={
            typeof content === 'string'
              ? content
              : JSON.stringify(content, null, 2)
          }
          className="bg-muted p-6 rounded-lg text-sm overflow-auto font-mono border border-border max-h-[70vh]"
        />
      );
    }

    if (typeof content === 'string') {
      // Check if it's markdown-like content
      if (content.includes('#') || content.includes('*') || content.includes('`') || content.includes('\n')) {
        return (
          <div className="max-w-none text-muted-foreground">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                h1: ({children}) => <h1 className="text-2xl font-semibold tracking-tight mb-4 border-b border-border pb-2">{children}</h1>,
                h2: ({children}) => <h2 className="text-xl font-semibold mb-3">{children}</h2>,
                h3: ({children}) => <h3 className="text-base font-semibold mb-2">{children}</h3>,
                h4: ({children}) => <h4 className="text-base font-medium text-foreground mb-2">{children}</h4>,
                p: ({children}) => <p className="mb-4 text-sm leading-relaxed">{children}</p>,
                ul: ({children}) => <ul className="list-disc list-inside mb-4 text-sm space-y-1">{children}</ul>,
                ol: ({children}) => <ol className="list-decimal list-inside mb-4 text-sm space-y-1">{children}</ol>,
                li: ({children}) => <li className="leading-relaxed text-sm">{children}</li>,
                code: ({children, className}) => {
                  const isInline = !className;
                  return isInline ? (
                    <code className="bg-muted px-1.5 py-0.5 rounded text-sm font-mono text-foreground border border-border">
                      {children}
                    </code>
                  ) : (
                    <code className={className}>{children}</code>
                  );
                },
                pre: ({children}) => (
                  <pre className="json-hl-root bg-muted p-4 rounded-lg text-sm overflow-auto border border-border mb-4 font-mono">
                    {children}
                  </pre>
                ),
                blockquote: ({children}) => (
                  <blockquote className="border-l-4 border-accent-primary pl-4 italic text-muted-foreground mb-4 bg-muted/30 py-2 rounded-r">
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
                  <div className="overflow-auto mb-4 border border-border rounded-lg">
                    <table className="min-w-full">{children}</table>
                  </div>
                ),
                thead: ({children}) => (
                  <thead className="bg-muted">{children}</thead>
                ),
                th: ({children}) => (
                  <th className="border-b border-border px-4 py-3 text-left text-sm font-medium text-foreground">
                    {children}
                  </th>
                ),
                td: ({children}) => (
                  <td className="border-b border-border px-4 py-3 text-sm">
                    {children}
                  </td>
                ),
                hr: () => (
                  <hr className="my-6 border-border" />
                ),
              }}
            >
              {content}
            </ReactMarkdown>
          </div>
        );
      } else {
        // Plain text with proper formatting
        return (
          <div className="text-muted-foreground leading-relaxed whitespace-pre-wrap font-sans">
            {content}
          </div>
        );
      }
    }

    // For non-string content, show formatted JSON
    return (
      <JsonHighlightedPre
        data={content}
        className="bg-muted p-6 rounded-lg text-sm overflow-auto font-mono border border-border max-h-[70vh] break-words"
      />
    );
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-4xl max-h-[90vh] flex flex-col">
        <DialogHeader className="flex-shrink-0 border-b border-border pb-4">
          <div className="flex items-center justify-between">
            <div className="flex-1 min-w-0">
              <DialogTitle className="text-base font-semibold text-foreground mb-2">
                {title}
              </DialogTitle>

              {/* Breadcrumb */}
              <div className="flex items-center gap-1 text-sm text-muted-foreground">
                <span>Result</span>
                {path.map((segment, index) => (
                  <div key={index} className="flex items-center gap-1">
                    <ChevronRight className="h-3 w-3" />
                    <span className="font-mono">{segment}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="flex items-center gap-2 ml-4">
              {/* View Mode Toggle */}
              <SegmentedControl
                value={viewMode}
                onValueChange={(mode) => setViewMode(mode as 'formatted' | 'raw')}
                options={viewOptions}
                size="sm"
                optionClassName="min-w-[110px]"
              />

              {/* Actions */}
              <Button
                variant="outline"
                size="sm"
                onClick={handleCopy}
                className="h-8 px-3"
              >
                <Copy className="h-3 w-3 mr-1" />
                Copy
              </Button>

              <Button
                variant="outline"
                size="sm"
                onClick={handleDownload}
                className="h-8 px-3"
              >
                <Download className="h-3 w-3 mr-1" />
                Download
              </Button>

              <Button
                variant="ghost"
                size="sm"
                onClick={onClose}
                className="h-8 w-8 p-0"
              >
                <Close className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </DialogHeader>

        <div className="flex-1 overflow-auto p-6 min-h-0">
          {renderContent()}
        </div>

        {/* Footer with metadata */}
        <div className="flex-shrink-0 border-t border-border pt-3 px-6 pb-4">
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <div className="flex items-center gap-4">
              <span>Type: {typeof content}</span>
              {typeof content === 'string' && (
                <span>Length: {content.length.toLocaleString()} characters</span>
              )}
              {Array.isArray(content) && (
                <span>Items: {content.length.toLocaleString()}</span>
              )}
              {typeof content === 'object' && content !== null && !Array.isArray(content) && (
                <span>Keys: {Object.keys(content).length}</span>
              )}
            </div>
            <Badge variant="secondary" className="text-xs">
              {path.join('.')}
            </Badge>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
