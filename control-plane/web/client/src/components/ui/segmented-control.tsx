"use client";

import * as React from "react";
import { cn } from "@/lib/utils";

import {
  AnimatedTabs,
  AnimatedTabsList,
  AnimatedTabsTrigger,
} from "./tabs";

export interface SegmentedControlOption {
  value: string;
  label: string;
  icon?: React.ComponentType<{ className?: string }>;
}

export interface SegmentedControlProps {
  value: string;
  onValueChange: (value: string) => void;
  options: ReadonlyArray<SegmentedControlOption>;
  className?: string;
  optionClassName?: string;
  size?: "default" | "sm";
  hideLabel?: boolean;
}

export function SegmentedControl({
  value,
  onValueChange,
  options,
  className,
  optionClassName,
  size = "default",
  hideLabel = false,
}: SegmentedControlProps) {
  const listHeight = size === "sm" ? "h-9" : "h-10";
  const triggerPadding =
    size === "sm" ? "px-3 py-1.5 text-xs" : "px-3.5 py-2 text-sm";

  return (
    <AnimatedTabs
      value={value}
      onValueChange={onValueChange}
      className={cn("h-full", className)}
    >
      <AnimatedTabsList
        className={cn(
          "gap-1 rounded-lg bg-muted/80 p-1",
          listHeight
        )}
      >
        {options.map((option) => {
          const Icon = option.icon;
          return (
            <AnimatedTabsTrigger
              key={option.value}
              value={option.value}
              className={cn(
                "gap-1.5 px-3 py-2 text-sm font-medium",
                triggerPadding,
                optionClassName
              )}
            >
              {Icon && <Icon className="h-4 w-4" />}
              {hideLabel ? (
                <span className="sr-only">{option.label}</span>
              ) : (
                <span>{option.label}</span>
              )}
            </AnimatedTabsTrigger>
          );
        })}
      </AnimatedTabsList>
    </AnimatedTabs>
  );
}
