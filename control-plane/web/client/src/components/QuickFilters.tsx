import { TrashCan } from "@/components/ui/icon-bridge";
import { cn } from "../lib/utils";
import type { FilterTag as FilterTagType } from "../types/filters";
import { createFilterTag } from "../utils/filterUtils";
import { Button } from "./ui/button";

interface QuickFiltersProps {
    tags: FilterTagType[];
    onTagsChange: (tags: FilterTagType[]) => void;
    className?: string;
}

const QUICK_FILTERS = [
    { type: "status", value: "running", label: "Running", color: "blue" },
    { type: "status", value: "succeeded", label: "Succeeded", color: "green" },
    { type: "status", value: "failed", label: "Failed", color: "red" },
    { type: "time", value: "last-24h", label: "Last 24h", color: "indigo" },
] as const;

export function QuickFilters({
    tags,
    onTagsChange,
    className,
}: QuickFiltersProps) {
    const handleQuickFilterClick = (
        filterType: string,
        filterValue: string,
        label: string,
    ) => {
        // Check if filter already exists
        const existingIndex = tags.findIndex(
            (tag) => tag.type === filterType && tag.value === filterValue,
        );

        if (existingIndex >= 0) {
            // Remove existing filter
            const newTags = [...tags];
            newTags.splice(existingIndex, 1);
            onTagsChange(newTags);
        } else {
            // Add new filter
            const newTag = createFilterTag(
                filterType as any,
                filterValue,
                label,
            );
            onTagsChange([...tags, newTag]);
        }
    };

    const handleClearAll = () => {
        onTagsChange([]);
    };

    const isFilterActive = (filterType: string, filterValue: string) => {
        return tags.some(
            (tag) => tag.type === filterType && tag.value === filterValue,
        );
    };

    return (
        <div className={cn("flex flex-wrap items-center gap-3", className)}>
            {/* Quick filter buttons */}
            <div className="flex flex-wrap items-center gap-2">
                {QUICK_FILTERS.map((filter) => {
                    const isActive = isFilterActive(filter.type, filter.value);
                    return (
                        <Button
                            key={`${filter.type}-${filter.value}`}
                            variant={isActive ? "default" : "outline"}
                            size="sm"
                            aria-pressed={isActive}
                            onClick={() =>
                                handleQuickFilterClick(
                                    filter.type,
                                    filter.value,
                                    filter.label,
                                )
                            }
                            className={cn(
                                "h-8 text-xs font-medium transition-all duration-200 border-border",
                                isActive &&
                                    "shadow-sm bg-primary text-primary-foreground border-primary",
                                !isActive &&
                                    "bg-background hover:bg-accent hover:text-accent-foreground",
                            )}
                        >
                            {filter.label}
                        </Button>
                    );
                })}
            </div>

            {/* Clear all button */}
            {tags.length > 0 && (
                <>
                    <div className="h-4 w-px bg-border" />
                    <Button
                        variant="ghost"
                        size="sm"
                        onClick={handleClearAll}
                        className="h-8 text-body-small hover:text-foreground hover:bg-accent"
                    >
                        <TrashCan size={14} className="mr-1.5" />
                        Clear all
                    </Button>
                </>
            )}
        </div>
    );
}
