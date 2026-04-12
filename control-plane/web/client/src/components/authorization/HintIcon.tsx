import type { ReactNode } from "react";
import { Info } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";

/** Inline info control (Lucide `Info`) — details in tooltip on hover/focus. */
export function HintIcon({
  label,
  children,
}: {
  label: string;
  children: ReactNode;
}) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          className="inline-flex size-6 shrink-0 items-center justify-center rounded-md text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
          aria-label={label}
        >
          <Info className="size-3.5" strokeWidth={2.25} aria-hidden />
        </button>
      </TooltipTrigger>
      <TooltipContent side="top" className="max-w-[min(280px,calc(100vw-2rem))] text-xs leading-snug">
        {children}
      </TooltipContent>
    </Tooltip>
  );
}
