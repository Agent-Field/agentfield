import React, { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { ResponsiveGrid } from "@/components/layout/ResponsiveGrid";
import { FilterSelect } from "./ui/FilterSelect";
import { TextInput } from "./ui/TextInput";
import {
    Collapsible,
    CollapsibleContent,
    CollapsibleTrigger,
} from "./ui/collapsible";
import type { ExecutionFilters as ExecutionFiltersType } from "../types/executions";

interface ExecutionFiltersProps {
    filters: Partial<ExecutionFiltersType>;
    onFiltersChange: (filters: Partial<ExecutionFiltersType>) => void;
    onSearch: (searchTerm: string) => void;
    searchTerm: string;
}

export function ExecutionFilters({
    filters,
    onFiltersChange,
    onSearch,
    searchTerm,
}: ExecutionFiltersProps) {
    const [isOpen, setIsOpen] = useState(false);
    const [localSearchTerm, setLocalSearchTerm] = useState(searchTerm);
    const [localFilters, setLocalFilters] = useState(filters);
    const [activeTimeFilter, setActiveTimeFilter] = useState<
        "last-hour" | "last-24h" | "last-week" | "all-time" | null
    >(null);

    useEffect(() => {
        setLocalSearchTerm(searchTerm);
    }, [searchTerm]);

    useEffect(() => {
        setLocalFilters(filters);
    }, [filters]);

    const handleSearchSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        onSearch(localSearchTerm);
    };

    const handleFilterChange = (
        key: keyof ExecutionFiltersType,
        value: any,
    ) => {
        const newFilters = { ...localFilters, [key]: value };
        setLocalFilters(newFilters);
        onFiltersChange(newFilters);
    };

    const clearFilters = () => {
        const clearedFilters = {
            page: 1,
            page_size: filters.page_size || 20,
        };
        setLocalFilters(clearedFilters);
        setLocalSearchTerm("");
        setActiveTimeFilter(null);
        onFiltersChange(clearedFilters);
        onSearch("");
    };

    const getActiveFiltersCount = () => {
        let count = 0;
        if (localFilters.agent_node_id) count++;
        if (localFilters.workflow_id) count++;
        if (localFilters.session_id) count++;
        if (localFilters.actor_id) count++;
        if (localFilters.status) count++;
        if (localFilters.start_time) count++;
        if (localFilters.end_time) count++;
        if (localSearchTerm) count++;
        return count;
    };

    const statusOptions = [
        { value: "", label: "All Statuses" },
        { value: "running", label: "Running" },
        { value: "completed", label: "Completed" },
        { value: "failed", label: "Failed" },
        { value: "pending", label: "Pending" },
    ];

    const pageSizeOptions = [
        { value: 10, label: "10 per page" },
        { value: 20, label: "20 per page" },
        { value: 50, label: "50 per page" },
        { value: 100, label: "100 per page" },
    ];

    return (
        <Card>
            <Collapsible open={isOpen} onOpenChange={setIsOpen}>
                <CollapsibleTrigger asChild>
                    <CardHeader
                        className="cursor-pointer hover:bg-gray-50 transition-colors"
                        aria-pressed={isOpen}
                    >
                        <div className="flex items-center justify-between">
                            <CardTitle>Filters & Search</CardTitle>
                            <div className="flex items-center space-x-2">
                                {getActiveFiltersCount() > 0 && (
                                    <Badge variant="count" size="sm">
                                        {getActiveFiltersCount()} active
                                    </Badge>
                                )}
                                <span className="text-body">
                                    {isOpen ? "▲" : "▼"}
                                </span>
                            </div>
                        </div>
                    </CardHeader>
                </CollapsibleTrigger>

                <CollapsibleContent>
                    <CardContent className="space-y-6">
                        {/* Search */}
                        <div>
                            <label className="block text-body font-medium mb-2">
                                Search Executions
                            </label>
                            <form
                                onSubmit={handleSearchSubmit}
                                className="flex space-x-2"
                            >
                                <input
                                    type="text"
                                    value={localSearchTerm}
                                    onChange={(e) =>
                                        setLocalSearchTerm(e.target.value)
                                    }
                                    placeholder="Search by workflow name, execution ID, or error message..."
                                    className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                />
                                <Button type="submit" size="sm">
                                    Search
                                </Button>
                                {localSearchTerm && (
                                    <Button
                                        type="button"
                                        variant="outline"
                                        size="sm"
                                        onClick={() => {
                                            setLocalSearchTerm("");
                                            onSearch("");
                                        }}
                                    >
                                        Clear
                                    </Button>
                                )}
                            </form>
                        </div>

                        {/* Filter Grid */}
                        <ResponsiveGrid
                            columns={{ base: 1, md: 2, lg: 3 }}
                            gap="md"
                            align="start"
                        >
                            {/* Status Filter */}
                            <FilterSelect
                                label="Status"
                                orientation="stacked"
                                value={localFilters.status || ""}
                                onValueChange={(value) =>
                                    handleFilterChange(
                                        "status",
                                        value || undefined,
                                    )
                                }
                                options={statusOptions}
                            />

                            {/* Agent Node Filter */}
                            <TextInput
                                label="Agent Node"
                                value={localFilters.agent_node_id || ""}
                                onChange={(e) =>
                                    handleFilterChange(
                                        "agent_node_id",
                                        e.target.value || undefined,
                                    )
                                }
                                placeholder="Filter by agent node ID..."
                            />

                            {/* Workflow Filter */}
                            <TextInput
                                label="Workflow"
                                value={localFilters.workflow_id || ""}
                                onChange={(e) =>
                                    handleFilterChange(
                                        "workflow_id",
                                        e.target.value || undefined,
                                    )
                                }
                                placeholder="Filter by workflow ID..."
                            />

                            {/* Session Filter */}
                            <TextInput
                                label="Session"
                                value={localFilters.session_id || ""}
                                onChange={(e) =>
                                    handleFilterChange(
                                        "session_id",
                                        e.target.value || undefined,
                                    )
                                }
                                placeholder="Filter by session ID..."
                            />

                            {/* Actor Filter */}
                            <TextInput
                                label="Actor"
                                value={localFilters.actor_id || ""}
                                onChange={(e) =>
                                    handleFilterChange(
                                        "actor_id",
                                        e.target.value || undefined,
                                    )
                                }
                                placeholder="Filter by actor ID..."
                            />

                            {/* Page Size */}
                            <FilterSelect
                                label="Results per page"
                                orientation="stacked"
                                value={String(localFilters.page_size || 20)}
                                onValueChange={(value) =>
                                    handleFilterChange(
                                        "page_size",
                                        parseInt(value, 10),
                                    )
                                }
                                options={pageSizeOptions.map(
                                    ({ value, label }) => ({
                                        value: String(value),
                                        label,
                                    }),
                                )}
                            />
                        </ResponsiveGrid>

                        {/* Time Range Filters */}
                        <ResponsiveGrid
                            columns={{ base: 1, md: 2 }}
                            gap="md"
                            align="start"
                        >
                            <div>
                                <label className="block text-sm font-medium mb-2">
                                    Start Time (From)
                                </label>
                                <input
                                    type="datetime-local"
                                    value={
                                        localFilters.start_time
                                            ? (() => {
                                                  const date = new Date(
                                                      localFilters.start_time,
                                                  );
                                                  return isNaN(date.getTime())
                                                      ? ""
                                                      : date
                                                            .toISOString()
                                                            .slice(0, 16);
                                              })()
                                            : ""
                                    }
                                    onChange={(e) =>
                                        handleFilterChange(
                                            "start_time",
                                            e.target.value
                                                ? new Date(
                                                      e.target.value,
                                                  ).toISOString()
                                                : undefined,
                                        )
                                    }
                                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                />
                            </div>
                            <div>
                                <label className="block text-sm font-medium mb-2">
                                    End Time (To)
                                </label>
                                <input
                                    type="datetime-local"
                                    value={
                                        localFilters.end_time
                                            ? (() => {
                                                  const date = new Date(
                                                      localFilters.end_time,
                                                  );
                                                  return isNaN(date.getTime())
                                                      ? ""
                                                      : date
                                                            .toISOString()
                                                            .slice(0, 16);
                                              })()
                                            : ""
                                    }
                                    onChange={(e) =>
                                        handleFilterChange(
                                            "end_time",
                                            e.target.value
                                                ? new Date(
                                                      e.target.value,
                                                  ).toISOString()
                                                : undefined,
                                        )
                                    }
                                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                                />
                            </div>
                        </ResponsiveGrid>

                        {/* Quick Time Filters */}
                        <div>
                            <label className="block text-sm font-medium mb-2">
                                Quick Time Filters
                            </label>
                            <div className="flex flex-wrap gap-2">
                                <Button
                                    variant={
                                        activeTimeFilter === "last-hour"
                                            ? "default"
                                            : "outline"
                                    }
                                    size="sm"
                                    aria-pressed={
                                        activeTimeFilter === "last-hour"
                                    }
                                    onClick={() => {
                                        setActiveTimeFilter("last-hour");
                                        const now = new Date();
                                        const oneHourAgo = new Date(
                                            now.getTime() - 60 * 60 * 1000,
                                        );
                                        handleFilterChange(
                                            "start_time",
                                            oneHourAgo.toISOString(),
                                        );
                                        handleFilterChange(
                                            "end_time",
                                            now.toISOString(),
                                        );
                                    }}
                                >
                                    Last Hour
                                </Button>
                                <Button
                                    variant={
                                        activeTimeFilter === "last-24h"
                                            ? "default"
                                            : "outline"
                                    }
                                    size="sm"
                                    aria-pressed={
                                        activeTimeFilter === "last-24h"
                                    }
                                    onClick={() => {
                                        setActiveTimeFilter("last-24h");
                                        const now = new Date();
                                        const oneDayAgo = new Date(
                                            now.getTime() - 24 * 60 * 60 * 1000,
                                        );
                                        handleFilterChange(
                                            "start_time",
                                            oneDayAgo.toISOString(),
                                        );
                                        handleFilterChange(
                                            "end_time",
                                            now.toISOString(),
                                        );
                                    }}
                                >
                                    Last 24 Hours
                                </Button>
                                <Button
                                    variant={
                                        activeTimeFilter === "last-week"
                                            ? "default"
                                            : "outline"
                                    }
                                    size="sm"
                                    aria-pressed={
                                        activeTimeFilter === "last-week"
                                    }
                                    onClick={() => {
                                        setActiveTimeFilter("last-week");
                                        const now = new Date();
                                        const oneWeekAgo = new Date(
                                            now.getTime() -
                                                7 * 24 * 60 * 60 * 1000,
                                        );
                                        handleFilterChange(
                                            "start_time",
                                            oneWeekAgo.toISOString(),
                                        );
                                        handleFilterChange(
                                            "end_time",
                                            now.toISOString(),
                                        );
                                    }}
                                >
                                    Last Week
                                </Button>
                                <Button
                                    variant={
                                        activeTimeFilter === "all-time"
                                            ? "default"
                                            : "outline"
                                    }
                                    size="sm"
                                    aria-pressed={
                                        activeTimeFilter === "all-time"
                                    }
                                    onClick={() => {
                                        setActiveTimeFilter("all-time");
                                        handleFilterChange(
                                            "start_time",
                                            undefined,
                                        );
                                        handleFilterChange(
                                            "end_time",
                                            undefined,
                                        );
                                    }}
                                >
                                    All Time
                                </Button>
                            </div>
                        </div>

                        {/* Actions */}
                        <div className="flex justify-between items-center pt-4 border-t">
                            <div className="text-body-small">
                                {getActiveFiltersCount() > 0 &&
                                    `${getActiveFiltersCount()} filter(s) active`}
                            </div>
                            <div className="flex space-x-2">
                                <Button
                                    variant="outline"
                                    size="sm"
                                    onClick={clearFilters}
                                >
                                    Clear All Filters
                                </Button>
                                <Button
                                    variant="outline"
                                    size="sm"
                                    onClick={() => setIsOpen(false)}
                                >
                                    Close
                                </Button>
                            </div>
                        </div>
                    </CardContent>
                </CollapsibleContent>
            </Collapsible>
        </Card>
    );
}
