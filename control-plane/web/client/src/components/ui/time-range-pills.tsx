import { cn } from "@/lib/utils";
import { Button } from "./button";

interface TimeRangeOption {
    value: string;
    label: string;
}

interface TimeRangePillsProps {
    value: string;
    onChange: (value: string) => void;
    options: TimeRangeOption[];
    className?: string;
}

export function TimeRangePills({
    value,
    onChange,
    options,
    className,
}: TimeRangePillsProps) {
    return (
        <div className={cn("flex flex-wrap items-center gap-2", className)}>
            {options.map((option) => {
                const isActive = value === option.value;

                return (
                    <Button
                        key={option.value}
                        variant={isActive ? "default" : "ghost"}
                        size="sm"
                        onClick={() => onChange(option.value)}
                        className={cn(
                            "transition-all",
                            isActive && "shadow-sm",
                        )}
                    >
                        {option.label}
                    </Button>
                );
            })}
        </div>
    );
}
