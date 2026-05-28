import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "@/lib/utils";

/**
 * Whisper-quiet type hint (e.g. Reasoner / Skill). Use on a *secondary* line
 * under the title, or `density="inline"` when it must sit beside the title.
 */
const entityTagVariants = cva(
  "inline-flex max-w-full shrink-0 items-baseline font-mono font-normal tabular-nums leading-none text-muted-foreground/45",
  {
    variants: {
      /** Default: same scale as subtitle / meta rows (10px). */
      density: {
        meta: "text-[0.625rem] leading-[0.875rem]",
        /** Next to a title — forced smaller than text-xs names */
        inline: "!text-[9px] !leading-3",
      },
      tone: {
        neutral: "",
        accent: "text-muted-foreground/50",
      },
    },
    defaultVariants: {
      density: "meta",
      tone: "neutral",
    },
  }
);

export interface EntityTagProps
  extends React.HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof entityTagVariants> {}

export function EntityTag({ className, tone, density, ...props }: EntityTagProps) {
  return (
    <span
      className={cn(entityTagVariants({ tone, density }), className)}
      {...props}
    />
  );
}
