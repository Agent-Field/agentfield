import { cn } from "@/lib/utils";
import { Button } from "./button";
import type { IconComponent } from "./icon-bridge";

interface SegmentedStatusOption {
    value: string;
    label: string;
    icon?: IconComponent;
    count?: number;
}

interface SegmentedStatusFilterProps {
    value: string;
    onChange: (value: string) => void;
    options: SegmentedStatusOption[];
    className?: string;
}

export function SegmentedStatusFilter({
    value,
    onChange,
    options,
    className,
}: SegmentedStatusFilterProps) {
    return (
        <div className={cn("flex flex-wrap items-center gap-2", className)}>
            {options.map((option) => {
                const isActive = value === option.value;
                const Icon = option.icon;

                return (
                    <Button
                        key={option.value}
                        variant={isActive ? "default" : "ghost"}
                        size="sm"
                        onClick={() => onChange(option.value)}
                        className={cn(
                            "gap-2 transition-all",
                            isActive && "shadow-sm",
                        )}
                    >
                        {Icon && (
                            <Icon
                                size={14}
                            />
                        )}
                        <span>{option.label}</span>
                        {option.count !== undefined && (
                            <span
                                className={cn(
                                    "ml-1 rounded-full px-2 py-0.5 text-xs font-semibold",
                                    isActive
                                        ? "bg-primary-foreground/20 text-primary-foreground"
                                        : "bg-muted text-muted-foreground",
                                )}
                            >
                                {option.count > 999999
                                    ? `${Math.floor(option.count / 1000000)}M`
                                    : option.count.toLocaleString()}
                            </span>
                        )}
                    </Button>
                );
            })}
        </div>
    );
}
