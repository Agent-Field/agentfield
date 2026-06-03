import { useCallback, useMemo } from "react";
import { Button } from "../ui/button";
import {
  TreeStructure,
  FlowArrow,
  SquaresFour,
  GridFour,
  Layers,
  Search,
  Scan,
  Eye,
  Zap,
  Bug,
  Focus,
  EyeOff,
  Loader2,
  Check,
} from "@/components/ui/icon-bridge";
import type { IconComponent } from "@/components/ui/icon-bridge";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "../ui/tooltip";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "../ui/dropdown-menu";
import { cn } from "@/lib/utils";
import type { AllLayoutType } from "./layouts/LayoutManager";

type ViewMode = "standard" | "performance" | "debug";

interface GraphToolbarProps {
  // Layout controls
  availableLayouts: AllLayoutType[];
  currentLayout: AllLayoutType;
  onLayoutChange: (layout: AllLayoutType) => void;
  isSlowLayout: (layout: AllLayoutType) => boolean;
  isLargeGraph: boolean;
  isApplyingLayout?: boolean;

  // View mode controls
  viewMode: ViewMode;
  onViewModeChange: (mode: ViewMode) => void;

  // Focus mode
  focusMode: boolean;
  onFocusModeChange?: (enabled: boolean) => void;

  // Navigation
  onSearchToggle: () => void;
  showSearch: boolean;
  onSmartCenter: () => void;
  hasSelection: boolean;
  controlsReady: boolean;
}

// ── Icon / label maps ──────────────────────────────────────────────

const LAYOUT_META: Record<
  AllLayoutType,
  { icon: IconComponent; label: string; shortDesc: string }
> = {
  tree: {
    icon: TreeStructure,
    label: "Tree",
    shortDesc: "Top-to-bottom hierarchy",
  },
  flow: {
    icon: FlowArrow,
    label: "Flow",
    shortDesc: "Left-to-right flow",
  },
  box: {
    icon: SquaresFour,
    label: "Box",
    shortDesc: "Fast packed layout",
  },
  rectpacking: {
    icon: GridFour,
    label: "Rectangle",
    shortDesc: "Rectangle packing",
  },
  layered: {
    icon: Layers,
    label: "Layered",
    shortDesc: "Hierarchical layers",
  },
};

const VIEW_MODE_META: Record<
  ViewMode,
  { icon: IconComponent; label: string; shortDesc: string }
> = {
  standard: { icon: Eye, label: "Standard", shortDesc: "Default view" },
  performance: {
    icon: Zap,
    label: "Performance",
    shortDesc: "Timing heatmap",
  },
  debug: { icon: Bug, label: "Debug", shortDesc: "Technical details" },
};

// ── Sub-components ─────────────────────────────────────────────────

function ToolbarIconButton({
  icon: Icon,
  tooltip,
  onClick,
  active = false,
  disabled = false,
  className,
}: {
  icon: IconComponent;
  tooltip: string;
  onClick: () => void;
  active?: boolean;
  disabled?: boolean;
  className?: string;
}) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          variant={active ? "default" : "ghost"}
          size="sm"
          onClick={onClick}
          disabled={disabled}
          className={cn(
            "h-8 w-8 p-0 transition-colors",
            active && "shadow-sm",
            className
          )}
        >
          <Icon className="w-4 h-4" />
        </Button>
      </TooltipTrigger>
      <TooltipContent side="top" className="text-xs">
        {tooltip}
      </TooltipContent>
    </Tooltip>
  );
}

function ToolbarDivider() {
  return <div className="w-px h-5 bg-border mx-0.5 shrink-0" />;
}

// ── Main component ─────────────────────────────────────────────────

export function GraphToolbar({
  availableLayouts,
  currentLayout,
  onLayoutChange,
  isSlowLayout,
  isLargeGraph,
  isApplyingLayout = false,
  viewMode,
  onViewModeChange,
  focusMode,
  onFocusModeChange,
  onSearchToggle,
  showSearch,
  onSmartCenter,
  hasSelection,
  controlsReady,
}: GraphToolbarProps) {
  const currentLayoutMeta = LAYOUT_META[currentLayout] ?? LAYOUT_META.tree;
  const currentViewMeta = VIEW_MODE_META[viewMode];
  const LayoutIcon = currentLayoutMeta.icon;
  const ViewIcon = currentViewMeta.icon;

  const handleLayoutSelect = useCallback(
    (layout: AllLayoutType) => {
      if (layout !== currentLayout) {
        onLayoutChange(layout);
      }
    },
    [currentLayout, onLayoutChange]
  );

  const handleViewModeSelect = useCallback(
    (mode: ViewMode) => {
      if (mode !== viewMode) {
        onViewModeChange(mode);
      }
    },
    [viewMode, onViewModeChange]
  );

  const centerTooltip = hasSelection ? "Center on selection" : "Fit to view";

  // Memoize the list of slow layouts for the note
  const hasSlowLayouts = useMemo(
    () =>
      isLargeGraph && availableLayouts.some((l) => isSlowLayout(l)),
    [isLargeGraph, availableLayouts, isSlowLayout]
  );

  return (
    <TooltipProvider delayDuration={300}>
      <div className="flex items-center gap-0.5 bg-background/95 backdrop-blur-sm border border-border rounded-lg shadow-lg p-1">
        {/* ── Search toggle ─────────────────────────────── */}
        <ToolbarIconButton
          icon={Search}
          tooltip={showSearch ? "Close search" : "Search nodes"}
          onClick={onSearchToggle}
          active={showSearch}
        />

        <ToolbarDivider />

        {/* ── Layout dropdown ───────────────────────────── */}
        <DropdownMenu>
          <Tooltip>
            <TooltipTrigger asChild>
              <DropdownMenuTrigger asChild>
                <Button
                  variant="ghost"
                  size="sm"
                  disabled={isApplyingLayout || availableLayouts.length === 0}
                  className="h-8 gap-1.5 px-2 text-xs font-medium transition-colors"
                >
                  {isApplyingLayout ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <LayoutIcon className="w-4 h-4" />
                  )}
                  <span className="hidden sm:inline">
                    {currentLayoutMeta.label}
                  </span>
                </Button>
              </DropdownMenuTrigger>
            </TooltipTrigger>
            <TooltipContent side="top" className="text-xs">
              Layout: {currentLayoutMeta.label}
            </TooltipContent>
          </Tooltip>

          <DropdownMenuContent align="end" className="w-52">
            <DropdownMenuLabel className="text-xs text-muted-foreground font-normal">
              Graph Layout
            </DropdownMenuLabel>
            <DropdownMenuSeparator />

            {availableLayouts.map((layout) => {
              const meta = LAYOUT_META[layout] ?? {
                icon: SquaresFour,
                label: layout,
                shortDesc: "",
              };
              const Icon = meta.icon;
              const isActive = currentLayout === layout;
              const isSlow = isSlowLayout(layout) && isLargeGraph;

              return (
                <DropdownMenuItem
                  key={layout}
                  onClick={() => handleLayoutSelect(layout)}
                  className="flex items-center gap-2 cursor-pointer"
                >
                  <Icon className="w-4 h-4 shrink-0" />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-1.5">
                      <span className={cn("text-sm", isActive && "font-medium")}>
                        {meta.label}
                      </span>
                      {isSlow && (
                        <span className="text-micro text-muted-foreground">
                          slower
                        </span>
                      )}
                    </div>
                    <span className="text-xs text-muted-foreground">
                      {meta.shortDesc}
                    </span>
                  </div>
                  {isActive && (
                    <Check className="w-3.5 h-3.5 shrink-0 text-primary" />
                  )}
                </DropdownMenuItem>
              );
            })}

            {hasSlowLayouts && (
              <>
                <DropdownMenuSeparator />
                <div className="px-2 py-1.5 text-micro text-muted-foreground">
                  Some layouts may be slower for large graphs
                </div>
              </>
            )}
          </DropdownMenuContent>
        </DropdownMenu>

        {/* ── View mode dropdown ────────────────────────── */}
        <DropdownMenu>
          <Tooltip>
            <TooltipTrigger asChild>
              <DropdownMenuTrigger asChild>
                <Button
                  variant={viewMode !== "standard" ? "default" : "ghost"}
                  size="sm"
                  className={cn(
                    "h-8 gap-1.5 px-2 text-xs font-medium transition-colors",
                    viewMode !== "standard" && "shadow-sm"
                  )}
                >
                  <ViewIcon className="w-4 h-4" />
                  <span className="hidden sm:inline">
                    {currentViewMeta.label}
                  </span>
                </Button>
              </DropdownMenuTrigger>
            </TooltipTrigger>
            <TooltipContent side="top" className="text-xs">
              View: {currentViewMeta.label}
            </TooltipContent>
          </Tooltip>

          <DropdownMenuContent align="end" className="w-48">
            <DropdownMenuLabel className="text-xs text-muted-foreground font-normal">
              View Mode
            </DropdownMenuLabel>
            <DropdownMenuSeparator />

            {(Object.entries(VIEW_MODE_META) as [ViewMode, typeof VIEW_MODE_META.standard][]).map(
              ([mode, meta]) => {
                const Icon = meta.icon;
                const isActive = viewMode === mode;

                return (
                  <DropdownMenuItem
                    key={mode}
                    onClick={() => handleViewModeSelect(mode)}
                    className="flex items-center gap-2 cursor-pointer"
                  >
                    <Icon className="w-4 h-4 shrink-0" />
                    <div className="flex-1 min-w-0">
                      <span className={cn("text-sm", isActive && "font-medium")}>
                        {meta.label}
                      </span>
                      <span className="text-xs text-muted-foreground ml-1.5">
                        {meta.shortDesc}
                      </span>
                    </div>
                    {isActive && (
                      <Check className="w-3.5 h-3.5 shrink-0 text-primary" />
                    )}
                  </DropdownMenuItem>
                );
              }
            )}
          </DropdownMenuContent>
        </DropdownMenu>

        <ToolbarDivider />

        {/* ── Focus mode toggle ─────────────────────────── */}
        <ToolbarIconButton
          icon={focusMode ? EyeOff : Focus}
          tooltip={
            focusMode
              ? "Exit focus mode (Cmd/Ctrl+F)"
              : "Focus mode (Cmd/Ctrl+F)"
          }
          onClick={() => onFocusModeChange?.(!focusMode)}
          active={focusMode}
        />

        {/* ── Smart center / fit ────────────────────────── */}
        <ToolbarIconButton
          icon={Scan}
          tooltip={centerTooltip}
          onClick={onSmartCenter}
          disabled={!controlsReady}
        />
      </div>
    </TooltipProvider>
  );
}
