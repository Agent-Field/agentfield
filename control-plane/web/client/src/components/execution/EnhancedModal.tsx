import React from "react";
import { ArrowsOutSimple, CornersIn, X } from "@/components/ui/icon-bridge";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "../ui/dialog";
import { Button } from "../ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../ui/tabs";
import { UnifiedJsonViewer } from "@/components/ui/UnifiedJsonViewer";
import { JsonHighlightedPre } from "@/components/ui/json-syntax-highlight";

interface EnhancedModalProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  icon?: React.ComponentType<{ className?: string }>;
  children: React.ReactNode;
  maxWidth?: string;
  maxHeight?: string;
  resizable?: boolean;
}

interface DataModalProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  icon?: React.ComponentType<{ className?: string }>;
  data: unknown;
}

function EnhancedModal({
  isOpen,
  onClose,
  title,
  icon: Icon,
  children,
  maxWidth = "90vw",
  maxHeight = "90vh",
  resizable = true
}: EnhancedModalProps) {
  const [isMaximized, setIsMaximized] = React.useState(false);

  const modalWidth = isMaximized ? "100vw" : maxWidth;
  const modalHeight = isMaximized ? "100vh" : maxHeight;

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent
        className="p-0 overflow-hidden flex flex-col"
        style={{
          maxWidth: modalWidth,
          maxHeight: modalHeight,
          width: modalWidth,
          height: modalHeight
        }}
      >
        {/* Header - Fixed */}
        <DialogHeader className="flex-shrink-0 border-b border-border bg-background/95 backdrop-blur-sm">
          <div className="flex items-center justify-between p-4">
            <DialogTitle className="flex items-center gap-3 text-base font-semibold">
              {Icon && <Icon className="w-5 h-5" />}
              {title}
            </DialogTitle>

            <div className="flex items-center gap-2">
              {resizable && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setIsMaximized(!isMaximized)}
                  className="h-8 w-8 p-0"
                  title={isMaximized ? "Restore" : "Maximize"}
                >
                  {isMaximized ? (
                    <CornersIn className="h-4 w-4" />
                  ) : (
                    <ArrowsOutSimple className="h-4 w-4" />
                  )}
                </Button>
              )}

              <Button
                variant="ghost"
                size="sm"
                onClick={onClose}
                className="h-8 w-8 p-0"
                title="Close"
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </DialogHeader>

        {/* Content - Scrollable */}
        <div className="flex-1 overflow-auto scrollbar-thin scrollbar-track-transparent scrollbar-thumb-border">
          {children}
        </div>
      </DialogContent>
    </Dialog>
  );
}

export function DataModal({ isOpen, onClose, title, icon, data }: DataModalProps) {
  const [viewMode, setViewMode] = React.useState<"formatted" | "raw" | "markdown">("formatted");
  const handleViewModeChange = (value: string) => {
    if (value === "formatted" || value === "raw" || value === "markdown") {
      setViewMode(value);
    }
  };

  const jsonString = React.useMemo(() => {
    try {
      return JSON.stringify(data, null, 2);
    } catch {
      return String(data);
    }
  }, [data]);

  // Detect if data might contain markdown
  const hasMarkdownLikeContent = React.useMemo(() => {
    if (typeof data === "string") {
      return data.includes("**") || data.includes("*") || data.includes("`") || data.includes("#");
    }
    return false;
  }, [data]);

  const MarkdownRenderer = ({ content }: { content: string }) => {
    const formattedContent = React.useMemo(() => {
      return content
        .replace(/\*\*(.*?)\*\*/g, '<strong class="font-semibold">$1</strong>')
        .replace(/\*(.*?)\*/g, '<em class="italic">$1</em>')
        .replace(/`(.*?)`/g, '<code class="bg-muted px-1 rounded text-sm font-mono">$1</code>')
        .replace(/\n\n/g, '</p><p class="mt-2">')
        .replace(/\n/g, '<br>')
        .replace(/\[(.*?)\]\((.*?)\)/g, '<a href="$2" class="text-blue-500 hover:underline" target="_blank" rel="noopener">$1</a>');
    }, [content]);

    return (
      <div
        className="prose prose-sm max-w-none text-foreground"
        dangerouslySetInnerHTML={{ __html: `<p>${formattedContent}</p>` }}
      />
    );
  };

  return (
    <EnhancedModal
      isOpen={isOpen}
      onClose={onClose}
      title={`${title} - Full View`}
      icon={icon}
      maxWidth="90vw"
      maxHeight="90vh"
    >
      <div className="flex flex-col h-full">
        {/* Tab Navigation - Fixed */}
        <div className="flex-shrink-0 border-b border-border bg-background/95">
          <Tabs
            value={viewMode}
            onValueChange={handleViewModeChange}
            className="w-full"
          >
            <TabsList variant="underline" className="grid w-full grid-cols-3 h-12">
              <TabsTrigger
                value="formatted"
                variant="underline"
                className="justify-center"
              >
                Formatted View
              </TabsTrigger>
              <TabsTrigger
                value="raw"
                variant="underline"
                className="justify-center"
              >
                Raw JSON
              </TabsTrigger>
              {hasMarkdownLikeContent && (
                <TabsTrigger
                  value="markdown"
                  variant="underline"
                  className="justify-center"
                >
                  Markdown Preview
                </TabsTrigger>
              )}
            </TabsList>
          </Tabs>
        </div>

        {/* Tab Content - Scrollable */}
        <div className="flex-1 overflow-auto scrollbar-thin scrollbar-track-transparent scrollbar-thumb-border">
          <Tabs value={viewMode} className="h-full">
            <TabsContent value="formatted" className="h-full m-0 p-4 overflow-auto scrollbar-thin scrollbar-track-transparent scrollbar-thumb-border">
              <div className="h-full overflow-auto scrollbar-thin scrollbar-track-transparent scrollbar-thumb-border">
                <UnifiedJsonViewer
                  data={data}
                  className="h-full border-0"
                />
              </div>
            </TabsContent>

            <TabsContent value="raw" className="h-full m-0 p-4 overflow-auto scrollbar-thin scrollbar-track-transparent scrollbar-thumb-border">
              <div className="border border-border rounded-lg h-full overflow-auto bg-background scrollbar-thin scrollbar-track-transparent scrollbar-thumb-border">
                <div className="p-4 h-full overflow-auto scrollbar-thin scrollbar-track-transparent scrollbar-thumb-border hover:scrollbar-thumb-muted-foreground">
                  <JsonHighlightedPre
                    text={jsonString}
                    className="text-sm font-mono leading-relaxed"
                  />
                </div>
              </div>
            </TabsContent>

            {hasMarkdownLikeContent && (
              <TabsContent value="markdown" className="h-full m-0 p-4 overflow-auto scrollbar-thin scrollbar-track-transparent scrollbar-thumb-border">
                <div className="border border-border rounded-lg h-full overflow-auto bg-background scrollbar-thin scrollbar-track-transparent scrollbar-thumb-border">
                  <div className="p-4 h-full overflow-auto scrollbar-thin scrollbar-track-transparent scrollbar-thumb-border hover:scrollbar-thumb-muted-foreground">
                    <MarkdownRenderer content={typeof data === "string" ? data : jsonString} />
                  </div>
                </div>
              </TabsContent>
            )}
          </Tabs>
        </div>
      </div>
    </EnhancedModal>
  );
}

export { EnhancedModal };
