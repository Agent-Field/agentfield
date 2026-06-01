import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import type { TimeRangePreset } from "@/types/dashboard";

interface TimeRangeOption {
  value: TimeRangePreset;
  label: string;
}

const TIME_RANGE_OPTIONS: TimeRangeOption[] = [
  { value: "1h", label: "1h" },
  { value: "24h", label: "24h" },
  { value: "7d", label: "7d" },
  { value: "30d", label: "30d" },
];

interface TimeRangeSelectorProps {
  /** Current selected preset */
  value: TimeRangePreset;
  /** Callback when preset changes */
  onChange: (preset: TimeRangePreset) => void;
  /** Whether comparison mode is enabled */
  compare?: boolean;
  /** Callback when comparison toggle changes */
  onCompareChange?: (compare: boolean) => void;
  /** Additional class name */
  className?: string;
  /** Show compact version without compare toggle */
  compact?: boolean;
}

/**
 * Time range selector with preset pills and optional comparison toggle.
 * Uses theme tokens for colors and spacing.
 */
export function TimeRangeSelector({
  value,
  onChange,
  compare = false,
  onCompareChange,
  className,
  compact = false,
}: TimeRangeSelectorProps) {
  return (
    <div className={cn("flex flex-wrap items-center gap-3", className)}>
      {/* Preset Pills */}
      <div className="flex items-center gap-1 rounded-lg border border-border/60 bg-muted/30 p-1">
        {TIME_RANGE_OPTIONS.map((option) => {
          const isActive = value === option.value;

          return (
            <Button
              key={option.value}
              variant={isActive ? "default" : "ghost"}
              size="sm"
              onClick={() => onChange(option.value)}
              className={cn(
                "h-7 px-3 text-xs font-medium transition-all",
                isActive
                  ? "shadow-sm"
                  : "hover:bg-muted text-muted-foreground hover:text-foreground"
              )}
            >
              {option.label}
            </Button>
          );
        })}
      </div>

      {/* Compare Toggle */}
      {!compact && onCompareChange && (
        <div className="flex items-center gap-2">
          <Switch
            id="compare-toggle"
            checked={compare}
            onCheckedChange={onCompareChange}
            className="data-[state=checked]:bg-primary"
          />
          <Label
            htmlFor="compare-toggle"
            className="text-xs text-muted-foreground cursor-pointer select-none"
          >
            Compare
          </Label>
        </div>
      )}
    </div>
  );
}
