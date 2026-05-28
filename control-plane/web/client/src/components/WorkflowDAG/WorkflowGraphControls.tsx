"use client";

import { Minus, Plus, Scan } from "@/components/ui/icon-bridge";
import { Panel, useReactFlow } from "@xyflow/react";
import { Button } from "../ui/button";
import { Card, CardContent } from "../ui/card";
import { Separator } from "../ui/separator";
import { cn } from "../../lib/utils";

const FIT_OPTIONS = {
  padding: 0.2,
  includeHiddenNodes: false as const,
  duration: 220,
};

const ZOOM_OPTIONS = { duration: 200 };

interface WorkflowGraphControlsProps {
  className?: string;
  /** When false, controls are omitted (e.g. embedded graph uses the compact agent bar). */
  show?: boolean;
}

/**
 * React Flow viewport controls — matches workflow graph panel styling (card, blur, border).
 */
export function WorkflowGraphControls({
  className,
  show = true,
}: WorkflowGraphControlsProps) {
  const { fitView, zoomIn, zoomOut } = useReactFlow();

  if (!show) {
    return null;
  }

  return (
    <Panel position="bottom-right" className={cn("z-30 m-3", className)}>
      <Card className="border-border/80 bg-card/95 shadow-md backdrop-blur-sm">
        <CardContent className="flex flex-col gap-0.5 p-1.5">
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className="h-8 w-8 shrink-0 text-muted-foreground hover:text-foreground"
            onClick={() => void fitView(FIT_OPTIONS)}
            aria-label="Fit graph to view"
            title="Fit graph to view"
          >
            <Scan className="size-4" />
          </Button>
          <Separator className="bg-border/60" />
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className="h-8 w-8 shrink-0 text-muted-foreground hover:text-foreground"
            onClick={() => void zoomIn(ZOOM_OPTIONS)}
            aria-label="Zoom in"
            title="Zoom in"
          >
            <Plus className="size-4" />
          </Button>
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className="h-8 w-8 shrink-0 text-muted-foreground hover:text-foreground"
            onClick={() => void zoomOut(ZOOM_OPTIONS)}
            aria-label="Zoom out"
            title="Zoom out"
          >
            <Minus className="size-4" />
          </Button>
        </CardContent>
      </Card>
    </Panel>
  );
}
