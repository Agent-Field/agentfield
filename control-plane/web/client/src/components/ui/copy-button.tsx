"use client";

import * as React from "react";
import { Copy, Check } from "@/components/ui/icon-bridge";

import { Button, type ButtonProps } from "./button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "./tooltip";
import { cn } from "@/lib/utils";

export interface CopyButtonProps
  extends Omit<ButtonProps, "onClick" | "children"> {
  value: string;
  tooltip?: string;
  copiedTooltip?: string;
  onCopied?: (value: string) => void;
  successDuration?: number;
  onClick?: (event: React.MouseEvent<HTMLButtonElement>) => void;
  children?: React.ReactNode | ((copied: boolean) => React.ReactNode);
}

export const CopyButton = React.forwardRef<HTMLButtonElement, CopyButtonProps>(
  (
    {
      value,
      tooltip = "Copy to clipboard",
      copiedTooltip = "Copied!",
      onCopied,
      successDuration = 2000,
      className,
      children,
      variant = "ghost",
      size = "icon",
      onClick,
      ...props
    },
    ref
  ) => {
    const [copied, setCopied] = React.useState(false);
    const timeoutRef = React.useRef<number | null>(null);

    const handleClick = async (
      event: React.MouseEvent<HTMLButtonElement>
    ) => {
      onClick?.(event);
      if (event.defaultPrevented) {
        return;
      }

      try {
        await navigator.clipboard.writeText(value);
        if (timeoutRef.current !== null) {
          window.clearTimeout(timeoutRef.current);
        }
        setCopied(true);
        onCopied?.(value);
        timeoutRef.current = window.setTimeout(() => {
          setCopied(false);
          timeoutRef.current = null;
        }, successDuration);
      } catch (error) {
        // eslint-disable-next-line no-console
        console.error("Failed to copy text:", error);
      }
    };

    React.useEffect(() => {
      return () => {
        if (timeoutRef.current !== null) {
          window.clearTimeout(timeoutRef.current);
        }
      };
    }, []);

    const content =
      typeof children === "function" ? children(copied) : children;

    const button = (
      <Button
        ref={ref}
        variant={variant}
        size={size}
        className={cn("transition-all", className)}
        aria-label={copied ? copiedTooltip : tooltip}
        onClick={handleClick}
        {...props}
      >
        {content ?? (copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />)}
      </Button>
    );

    return (
      <Tooltip>
        <TooltipTrigger asChild>{button}</TooltipTrigger>
        <TooltipContent sideOffset={6}>
          {copied ? copiedTooltip : tooltip}
        </TooltipContent>
      </Tooltip>
    );
  }
);
CopyButton.displayName = "CopyButton";
