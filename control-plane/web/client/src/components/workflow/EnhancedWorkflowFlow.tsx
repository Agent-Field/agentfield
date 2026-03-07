import { useState, useCallback, useEffect, useMemo, useRef } from "react";
import { GitBranch, Search, X, Scan, LocateFixed, Loader2, Eye, Zap, Bug, Focus, EyeOff } from "@/components/ui/icon-bridge";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { WorkflowDAGViewer } from "../WorkflowDAGViewer";
import type { WorkflowDAGControls, WorkflowDAGResponse } from "../WorkflowDAG";
import { Badge } from "../ui/badge";
import { cn } from "@/lib/utils";
import type { WorkflowSummary, WorkflowTimelineNode } from "../../types/workflows";

type ViewMode = 'standard' | 'performance' | 'debug';

const VIEW_MODE_CONFIG: ReadonlyArray<{ value: ViewMode; icon: typeof Eye; label: string }> = [
  { value: "standard", icon: Eye, label: "Standard view" },
  { value: "performance", icon: Zap, label: "Performance view" },
  { value: "debug", icon: Bug, label: "Debug view" },
];

interface EnhancedWorkflowFlowProps {
  workflow: WorkflowSummary;
  dagData?: { timeline?: WorkflowTimelineNode[] } | null;
  loading?: boolean;
  isRefreshing?: boolean;
  error?: string | null;
  selectedNodeIds: string[];
  onNodeSelection: (nodeIds: string[], replace?: boolean) => void;
  viewMode: ViewMode;
  onViewModeChange: (mode: ViewMode) => void;
  focusMode: boolean;
  isFullscreen: boolean;
  onFocusModeChange?: (enabled: boolean) => void;
}

export function EnhancedWorkflowFlow({
  workflow,
  dagData,
  loading,
  isRefreshing = false,
  error,
  selectedNodeIds,
  onNodeSelection,
  viewMode,
  onViewModeChange,
  focusMode,
  onFocusModeChange,
}: EnhancedWorkflowFlowProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [showSearch, setShowSearch] = useState(false);
  const [searchSummary, setSearchSummary] = useState<{ total: number; firstMatchId?: string }>({ total: 0 });
  const dagControlsRef = useRef<WorkflowDAGControls | null>(null);
  const pendingSearchFocusRef = useRef(false);

  const safeSelectedNodeIds = useMemo(() => selectedNodeIds ?? [], [selectedNodeIds]);

  const handleRegisterControls = useCallback((controls: WorkflowDAGControls) => {
    dagControlsRef.current = controls;
  }, []);

  const handleSearchSummaryUpdate = useCallback(({
    totalMatches,
    firstMatchId,
  }: {
    totalMatches: number;
    firstMatchId?: string;
  }) => {
    setSearchSummary({ total: totalMatches, firstMatchId });
  }, []);

  const handleExecutionClick = useCallback((execution: WorkflowTimelineNode) => {
    onNodeSelection([execution.execution_id]);
  }, [onNodeSelection]);

  const handleSearchToggle = useCallback(() => {
    setShowSearch(!showSearch);
    if (showSearch) {
      setSearchQuery("");
    }
  }, [showSearch]);

  const clearSearch = useCallback(() => {
    setSearchQuery("");
    setShowSearch(false);
    pendingSearchFocusRef.current = false;
  }, []);

  const handleFitView = useCallback(() => {
    dagControlsRef.current?.fitToView({ padding: 0.2 });
  }, []);

  const handleCenterSelection = useCallback(() => {
    if (!safeSelectedNodeIds.length) {
      handleFitView();
      return;
    }
    dagControlsRef.current?.focusOnNodes(safeSelectedNodeIds, { padding: 0.3 });
  }, [handleFitView, safeSelectedNodeIds]);

  useEffect(() => {
    if (focusMode && safeSelectedNodeIds.length > 0) {
      dagControlsRef.current?.focusOnNodes(safeSelectedNodeIds, { padding: 0.35 });
    }
  }, [focusMode, safeSelectedNodeIds]);

  useEffect(() => {
    if (!searchQuery) {
      pendingSearchFocusRef.current = false;
      return;
    }
    if (pendingSearchFocusRef.current && searchSummary.firstMatchId) {
      dagControlsRef.current?.focusOnNodes([searchSummary.firstMatchId], { padding: 0.25 });
      onNodeSelection([searchSummary.firstMatchId], true);
      pendingSearchFocusRef.current = false;
    }
  }, [searchSummary.firstMatchId, searchQuery, onNodeSelection]);

  useEffect(() => {
    if (!searchQuery && searchSummary.total !== 0) {
      setSearchSummary({ total: 0 });
    }
  }, [searchQuery, searchSummary.total]);

  const hasDagContent = Boolean(dagData);
  const shouldShowInitialLoader = Boolean(loading && !hasDagContent);

  if (shouldShowInitialLoader) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center space-y-4">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto" />
          <p className="text-muted-foreground">Loading workflow visualization...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center space-y-4">
          <GitBranch className="h-12 w-12 text-muted-foreground mx-auto" />
          <div>
            <h3 className="text-heading-3 text-foreground">Unable to load workflow</h3>
            <p className="text-muted-foreground">{error}</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full min-h-0 flex-col relative">
      {showSearch && (
        <div className="absolute top-4 left-4 z-10 flex items-center gap-2 bg-background/95 backdrop-blur-sm border border-border rounded-lg shadow-lg p-2">
          <Input
            placeholder="Search by agent, reasoner, or execution ID..."
            value={searchQuery}
            onChange={(e) => {
              setSearchQuery(e.target.value);
              pendingSearchFocusRef.current = false;
            }}
            className="w-80"
            autoFocus
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                pendingSearchFocusRef.current = true;
              }
            }}
          />
          {searchQuery && (
            <Badge variant="outline" className="text-xs whitespace-nowrap">
              {searchSummary.total} match{searchSummary.total === 1 ? '' : 'es'}
            </Badge>
          )}
          <Button variant="ghost" size="sm" onClick={clearSearch} className="h-8 w-8 p-0">
            <X className="w-4 h-4" />
          </Button>
        </div>
      )}

      {/* Floating navigation — bottom right */}
      <div className="absolute bottom-6 right-6 z-10 flex flex-col gap-2">
        <Button
          variant="secondary"
          size="sm"
          onClick={handleSearchToggle}
          className="h-10 w-10 p-0 shadow-lg"
          title="Search nodes"
        >
          <Search className="w-4 h-4" />
        </Button>
        <Button
          variant="secondary"
          size="sm"
          onClick={handleCenterSelection}
          disabled={!dagControlsRef.current}
          className="h-10 w-10 p-0 shadow-lg"
          title="Center selection"
        >
          <LocateFixed className="w-4 h-4" />
        </Button>
        <Button
          variant="secondary"
          size="sm"
          onClick={handleFitView}
          disabled={!dagControlsRef.current}
          className="h-10 w-10 p-0 shadow-lg"
          title="Fit view"
        >
          <Scan className="w-4 h-4" />
        </Button>
      </div>

      {/* View controls — bottom left */}
      <div className="absolute bottom-6 left-6 z-10 flex items-center gap-1 bg-background/95 backdrop-blur-sm border border-border rounded-lg shadow-lg p-1">
        {VIEW_MODE_CONFIG.map((mode) => {
          const Icon = mode.icon;
          const isActive = viewMode === mode.value;
          return (
            <Button
              key={mode.value}
              variant={isActive ? "default" : "ghost"}
              size="sm"
              onClick={() => onViewModeChange(mode.value)}
              className={cn(
                "h-8 w-8 p-0",
                isActive && "shadow-sm"
              )}
              title={mode.label}
            >
              <Icon className="w-3.5 h-3.5" />
            </Button>
          );
        })}

        <div className="w-px h-5 bg-border mx-0.5" />

        <Button
          variant={focusMode ? "default" : "ghost"}
          size="sm"
          onClick={() => onFocusModeChange?.(!focusMode)}
          className={cn(
            "h-8 w-8 p-0",
            focusMode && "shadow-sm"
          )}
          title={focusMode ? "Exit focus mode (Cmd/Ctrl + F)" : "Focus mode (Cmd/Ctrl + F)"}
        >
          {focusMode ? <EyeOff className="w-3.5 h-3.5" /> : <Focus className="w-3.5 h-3.5" />}
        </Button>
      </div>

      {/* Status overlay — top left */}
      {(isRefreshing || selectedNodeIds.length > 0 || focusMode || viewMode !== 'standard') && (
        <div className="absolute top-4 left-4 z-10 flex items-center gap-2 bg-background/95 backdrop-blur-sm border border-border rounded-lg shadow-sm px-3 py-2" style={{ marginTop: showSearch ? '60px' : '0' }}>
          {isRefreshing && hasDagContent && (
            <span className="flex items-center gap-2 text-body-small">
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
              Updating
            </span>
          )}
          {viewMode !== 'standard' && (
            <Badge variant="secondary" className="bg-primary/10 text-primary border-primary/30 text-xs">
              {viewMode === 'performance' ? 'Performance' : 'Debug'}
            </Badge>
          )}
          {focusMode && (
            <Badge variant="secondary" className="text-xs">
              Focus {safeSelectedNodeIds.length > 0 ? `(${safeSelectedNodeIds.length})` : ''}
            </Badge>
          )}
          {selectedNodeIds.length > 0 && !focusMode && (
            <span className="text-body-small">
              {selectedNodeIds.length} selected
            </span>
          )}
        </div>
      )}

      <div className="flex flex-1 min-h-0 overflow-hidden">
        <WorkflowDAGViewer
          workflowId={workflow.workflow_id}
          dagData={dagData as WorkflowDAGResponse}
          loading={loading}
          error={error}
          onExecutionClick={handleExecutionClick}
          className="flex-1 min-h-0"
          searchQuery={searchQuery}
          focusMode={focusMode}
          focusedNodeIds={safeSelectedNodeIds}
          selectedNodeIds={safeSelectedNodeIds}
          onReady={handleRegisterControls}
          onSearchResultsChange={handleSearchSummaryUpdate}
          viewMode={viewMode}
        />
      </div>
    </div>
  );
}
