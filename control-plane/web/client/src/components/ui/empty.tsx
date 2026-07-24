import * as React from "react";
import { cn } from "@/lib/utils";

type EmptyMediaVariant = "default" | "icon";

const MEDIA_VARIANTS: Record<EmptyMediaVariant, string> = {
  default:
    "rounded-xl border border-dashed border-border/60 bg-muted/20 p-6 text-foreground",
  icon: "rounded-full border border-border/40 bg-muted/30 p-3 text-muted-foreground",
};

const BASE_WRAPPER_CLASSES =
  "flex w-full flex-col items-center justify-center gap-8 rounded-xl border border-dashed border-border/60 bg-muted/10 px-6 py-12 text-center shadow-sm backdrop-blur";

const Empty = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  function Empty({ className, ...props }, ref) {
    return (
      <div
        ref={ref}
        className={cn(BASE_WRAPPER_CLASSES, className)}
        {...props}
      />
    );
  }
);
Empty.displayName = "Empty";

const EmptyHeader = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(function EmptyHeader({ className, ...props }, ref) {
  return (
    <div
      ref={ref}
      className={cn(
        "flex flex-col items-center gap-3 text-center sm:max-w-md",
        className
      )}
      {...props}
    />
  );
});
EmptyHeader.displayName = "EmptyHeader";

interface EmptyMediaProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: EmptyMediaVariant;
}

const EmptyMedia = React.forwardRef<HTMLDivElement, EmptyMediaProps>(
  function EmptyMedia({ className, variant = "default", ...props }, ref) {
    return (
      <div
        ref={ref}
        className={cn(MEDIA_VARIANTS[variant], className)}
        {...props}
      />
    );
  }
);
EmptyMedia.displayName = "EmptyMedia";

const EmptyTitle = React.forwardRef<
  HTMLHeadingElement,
  React.HTMLAttributes<HTMLHeadingElement>
>(function EmptyTitle({ className, ...props }, ref) {
  return (
    <h3
      ref={ref}
      className={cn("text-xl font-semibold text-foreground", className)}
      {...props}
    />
  );
});
EmptyTitle.displayName = "EmptyTitle";

const EmptyDescription = React.forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLParagraphElement>
>(function EmptyDescription({ className, ...props }, ref) {
  return (
    <p
      ref={ref}
      className={cn("text-sm text-muted-foreground", className)}
      {...props}
    />
  );
});
EmptyDescription.displayName = "EmptyDescription";

const EmptyContent = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(function EmptyContent({ className, ...props }, ref) {
  return (
    <div
      ref={ref}
      className={cn(
        "flex flex-col items-center gap-3 sm:flex-row sm:justify-center",
        className
      )}
      {...props}
    />
  );
});
EmptyContent.displayName = "EmptyContent";

export {
  Empty,
  EmptyContent,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
};
