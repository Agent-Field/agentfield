import { useState } from "react";
import {
  MoreHorizontal,
  PauseCircle,
  Play,
  XCircle,
  Activity,
} from "lucide-react";

import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { isTerminalStatus } from "@/utils/status";
import type { WorkflowSummary } from "@/types/workflows";

/**
 * Honest copy for the cancel confirmation dialog. Exported so the bulk bar
 * can reuse the exact same language.
 *
 * Cancel semantics reality (from control-plane/internal/handlers/execute.go):
 * the control plane cannot kill a node that is already executing on an
 * agent worker — it can only refuse to dispatch the NEXT node. In-flight
 * work finishes and its output is discarded. We tell the user this up front.
 */
export const CANCEL_RUN_COPY = {
  title: (count: number) =>
    count > 1 ? `Cancel ${count} runs?` : "Cancel this run?",
  description:
    "Nodes currently executing will finish their current step — only pending nodes will be stopped. Any in-flight work will be discarded. This cannot be undone.",
  confirmLabel: (count: number) =>
    count > 1 ? `Cancel ${count} runs` : "Cancel run",
  keepLabel: "Keep running",
} as const;

/* ═══════════════════════════════════════════════════════════════
   Per-row kebab menu
   ═══════════════════════════════════════════════════════════════ */

interface RunLifecycleMenuProps {
  run: WorkflowSummary;
  isPending: boolean;
  onPause: (run: WorkflowSummary) => void;
  onResume: (run: WorkflowSummary) => void;
  onCancel: (run: WorkflowSummary) => void;
}

/**
 * Small inline kebab for run rows. Shows Pause / Resume / Cancel based on
 * the run's current status. Cancel opens an AlertDialog confirmation.
 *
 * Visually muted at rest, brightens on parent row hover (via
 * group-hover) so it stays discoverable without adding noise.
 */
export function RunLifecycleMenu({
  run,
  isPending,
  onPause,
  onResume,
  onCancel,
}: RunLifecycleMenuProps) {
  const [menuOpen, setMenuOpen] = useState(false);
  const [confirmOpen, setConfirmOpen] = useState(false);

  // Prefer the root execution status when available — that's the row the
  // user actually controls. The aggregate `run.status` can stay 'running'
  // even after the user paused the root because in-flight children keep
  // going (see backend execute.go dispatch-time guard). Falling back to
  // the aggregate keeps things working for older API responses.
  const effectiveStatus = run.root_execution_status ?? run.status;
  const isRunning = effectiveStatus === "running";
  const isPaused = effectiveStatus === "paused";
  const isTerminal = isTerminalStatus(effectiveStatus);
  const canPause = isRunning && Boolean(run.root_execution_id);
  const canResume = isPaused && Boolean(run.root_execution_id);
  const canCancel = !isTerminal && Boolean(run.root_execution_id);
  const hasAnyAction = canPause || canResume || canCancel;

  // Render an inert placeholder with the same footprint so the column
  // stays aligned across rows even when no action is available.
  if (!hasAnyAction && !isPending) {
    return <span className="inline-block size-9" aria-hidden />;
  }

  return (
    <>
      <DropdownMenu open={menuOpen} onOpenChange={setMenuOpen}>
        <DropdownMenuTrigger asChild>
          <Button
            variant="ghost"
            size="icon"
            className={cn(
              "size-9 shrink-0 text-muted-foreground/70 transition-colors",
              "group-hover/run-row:text-foreground",
              "hover:bg-muted hover:text-foreground",
              "data-[state=open]:bg-muted data-[state=open]:text-foreground",
              "focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1",
              isPending && "text-foreground",
            )}
            disabled={isPending}
            aria-busy={isPending}
            onClick={(e) => e.stopPropagation()}
            aria-label={`Run actions for ${run.run_id}`}
          >
            {isPending ? (
              <Activity className="size-3.5 animate-spin" aria-hidden />
            ) : (
              <MoreHorizontal className="size-3.5" aria-hidden />
            )}
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent
          align="end"
          side="bottom"
          className="w-44"
          onClick={(e) => e.stopPropagation()}
        >
          <DropdownMenuLabel className="text-xs font-normal text-muted-foreground">
            Lifecycle
          </DropdownMenuLabel>
          <DropdownMenuSeparator />
          {canPause ? (
            <DropdownMenuItem
              className="gap-2 text-xs"
              onClick={() => {
                setMenuOpen(false);
                onPause(run);
              }}
            >
              <PauseCircle className="size-3.5 text-amber-500" aria-hidden />
              Pause run
            </DropdownMenuItem>
          ) : null}
          {canResume ? (
            <DropdownMenuItem
              className="gap-2 text-xs"
              onClick={() => {
                setMenuOpen(false);
                onResume(run);
              }}
            >
              <Play className="size-3.5 text-emerald-500" aria-hidden />
              Resume run
            </DropdownMenuItem>
          ) : null}
          {canCancel ? (
            <DropdownMenuItem
              className="gap-2 text-xs text-destructive focus:text-destructive"
              onClick={() => {
                setMenuOpen(false);
                setConfirmOpen(true);
              }}
            >
              <XCircle className="size-3.5" aria-hidden />
              Cancel run
            </DropdownMenuItem>
          ) : null}
        </DropdownMenuContent>
      </DropdownMenu>

      <AlertDialog open={confirmOpen} onOpenChange={setConfirmOpen}>
        <AlertDialogContent onClick={(e) => e.stopPropagation()}>
          <AlertDialogHeader>
            <AlertDialogTitle>{CANCEL_RUN_COPY.title(1)}</AlertDialogTitle>
            <AlertDialogDescription>
              {CANCEL_RUN_COPY.description}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={isPending}>
              {CANCEL_RUN_COPY.keepLabel}
            </AlertDialogCancel>
            <AlertDialogAction
              disabled={isPending}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              onClick={() => {
                setConfirmOpen(false);
                onCancel(run);
              }}
            >
              {CANCEL_RUN_COPY.confirmLabel(1)}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}
