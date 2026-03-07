import { Button } from '../ui/button';
import { Badge } from '../ui/badge';
import { AlertTriangle, Loader2 } from '@/components/ui/icon-bridge';
import { HoverCard, HoverCardContent, HoverCardTrigger } from '../ui/hover-card';
import type { AllLayoutType } from './layouts/LayoutManager';

interface LayoutControlsProps {
  availableLayouts: AllLayoutType[];
  currentLayout: AllLayoutType;
  onLayoutChange: (layout: AllLayoutType) => void;
  isSlowLayout: (layout: AllLayoutType) => boolean;
  getLayoutDescription: (layout: AllLayoutType) => string;
  isLargeGraph: boolean;
  isApplyingLayout?: boolean;
  layoutProgress?: number;
}

const LAYOUT_LABELS: Record<AllLayoutType, string> = {
  tree: 'Tree View',
  flow: 'Flow View',
  box: 'Box',
  rectpacking: 'Rectangle',
  layered: 'Layered',
};

export function LayoutControls({
  availableLayouts,
  currentLayout,
  onLayoutChange,
  isSlowLayout,
  getLayoutDescription,
  isLargeGraph,
  isApplyingLayout = false,
  layoutProgress = 0,
}: LayoutControlsProps) {
  return (
    <div className="bg-background/90 backdrop-blur-sm border rounded-lg p-1 shadow-sm">
      <div className="flex gap-1">
        {availableLayouts.map((layout) => {
          const isSlow = isSlowLayout(layout);
          const isActive = currentLayout === layout;
          const label = LAYOUT_LABELS[layout];
          const description = getLayoutDescription(layout);

          return (
            <HoverCard key={layout}>
              <HoverCardTrigger asChild>
                <div className="relative">
                  <Button
                    variant={isActive ? 'default' : 'ghost'}
                    size="sm"
                    onClick={() => onLayoutChange(layout)}
                    disabled={isApplyingLayout}
                    className="h-8 px-3 text-xs font-medium relative"
                    title={description}
                  >
                    {isApplyingLayout && isActive && (
                      <Loader2 className="w-3 h-3 mr-1 animate-spin" />
                    )}
                    {label}
                    {isSlow && isLargeGraph && (
                      <AlertTriangle className="w-3 h-3 ml-1 text-amber-500" />
                    )}
                  </Button>

                  {/* Slow layout warning badge */}
                  {isSlow && isLargeGraph && (
                    <Badge
                      variant="secondary"
                      className="absolute -top-1 -right-1 h-4 px-1 text-[10px] bg-amber-100 text-amber-800 border-amber-200"
                    >
                      SLOW
                    </Badge>
                  )}
                </div>
              </HoverCardTrigger>
              <HoverCardContent side="bottom" className="max-w-xs">
                <div className="space-y-1">
                  <div className="font-medium">{label}</div>
                  <div className="text-body-small">{description}</div>
                  {isSlow && isLargeGraph && (
                    <div className="text-xs text-amber-600 flex items-center gap-1">
                      <AlertTriangle className="w-3 h-3" />
                      May be slow for large graphs
                    </div>
                  )}
                </div>
              </HoverCardContent>
            </HoverCard>
          );
        })}
      </div>

      {/* Progress bar for layout application */}
      {isApplyingLayout && layoutProgress > 0 && layoutProgress < 100 && (
        <div className="mt-2 w-full bg-muted rounded-full h-1">
          <div
            className="bg-primary h-1 rounded-full transition-all duration-300 ease-out"
            style={{ width: `${layoutProgress}%` }}
          />
        </div>
      )}

      {/* Graph size indicator */}
      <div className="mt-1 text-[10px] text-muted-foreground text-center">
        {isLargeGraph ? 'Large Graph Mode' : 'Small Graph Mode'}
      </div>
    </div>
  );
}
