"use client";

import {
  CaretDown,
  FunnelSimple,
  Pulse,
  SortAscending,
  SortDescending,
  Table,
  TreeStructure,
  User,
  FlowArrow,
} from "@/components/ui/icon-bridge";
import { useState, useEffect, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { cn } from "../lib/utils";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { Card, CardContent } from "./ui/card";
import { SearchBar } from "./ui/SearchBar";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "./ui/dropdown-menu";
import { EnhancedExecutionsTable } from "./EnhancedExecutionsTable";
import { WorkflowsTable } from "./WorkflowsTable";
import type {
  ViewMode,
  ExecutionViewFilters,
  ExecutionViewState,
  WorkflowsResponse,
  EnhancedExecutionsResponse,
  EnhancedExecution,
  WorkflowSummary,
} from "../types/workflows";
import {
  getExecutionsByViewMode,
  searchExecutionData,
  getExecutionViewStats,
} from "../services/workflowsApi";

const VIEW_MODES: ViewMode[] = [
  {
    id: 'executions',
    label: 'Executions',
    description: 'Individual execution calls',
    icon: 'pulse',
  },
  {
    id: 'workflows',
    label: 'Workflows',
    description: 'Grouped by workflow chains',
    icon: 'flow',
  },
  {
    id: 'sessions',
    label: 'Sessions',
    description: 'Grouped by user sessions',
    icon: 'user',
  },
  {
    id: 'agents',
    label: 'Agents',
    description: 'Grouped by agent nodes',
    icon: 'tree',
  },
];

const SORT_OPTIONS = [
  { value: 'when', label: 'When', field: 'started_at' },
  { value: 'status', label: 'Status', field: 'status' },
  { value: 'duration', label: 'Duration', field: 'duration_ms' },
  { value: 'agent', label: 'Agent', field: 'agent_node_id' },
];

const WORKFLOW_SORT_OPTIONS = [
  { value: 'latest_activity', label: 'Latest Activity', field: 'updated_at' },
  { value: 'status', label: 'Status', field: 'status' },
  { value: 'total_executions', label: 'Nodes', field: 'total_steps' },
  { value: 'failed', label: 'Issues', field: 'failed_steps' },
];

const STATUS_OPTIONS = [
  { value: '', label: 'All Statuses' },
  { value: 'running', label: 'Running' },
  { value: 'completed', label: 'Completed' },
  { value: 'failed', label: 'Failed' },
  { value: 'pending', label: 'Pending' },
];

function isWorkflowsResponse(value: unknown): value is WorkflowsResponse {
  return Boolean(value) && typeof value === 'object' && 'workflows' in (value as Record<string, unknown>);
}

function isEnhancedExecutionsResponse(value: unknown): value is EnhancedExecutionsResponse {
  return Boolean(value) && typeof value === 'object' && 'executions' in (value as Record<string, unknown>);
}

interface ExecutionViewTabsProps {
  className?: string;
}

export function ExecutionViewTabs({ className }: ExecutionViewTabsProps) {
  const navigate = useNavigate();
  const [viewState, setViewState] = useState<ExecutionViewState>({
    viewMode: 'executions',
    filters: {},
    sortBy: 'when',
    sortOrder: 'desc',
    page: 1,
    pageSize: 20,
  });

  const [data, setData] = useState<WorkflowsResponse | EnhancedExecutionsResponse | null>(null);
  const [stats, setStats] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState('');

  const workflowsResponse = isWorkflowsResponse(data) ? data : null;
  const executionsResponse = isEnhancedExecutionsResponse(data) ? data : null;
  const isFetchingMore = loading && viewState.page > 1;

  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const sortOptions = viewState.viewMode === 'workflows' ? WORKFLOW_SORT_OPTIONS : SORT_OPTIONS;
      const sortField = sortOptions.find(opt => opt.value === viewState.sortBy)?.field || 'started_at';

      const [result, statsResult] = await Promise.all([
        getExecutionsByViewMode(
          viewState.viewMode,
          viewState.filters,
          viewState.page,
          viewState.pageSize,
          sortField,
          viewState.sortOrder
        ),
        getExecutionViewStats(viewState.viewMode, viewState.filters),
      ]);

      setData(result);
      setStats(statsResult);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch data');
    } finally {
      setLoading(false);
    }
  }, [viewState]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const handleViewModeChange = (newViewMode: ViewMode['id']) => {
    setViewState(prev => ({
      ...prev,
      viewMode: newViewMode,
      page: 1,
      sortBy: newViewMode === 'workflows' ? 'latest_activity' : 'when',
    }));
  };

  const handleFiltersChange = (newFilters: Partial<ExecutionViewFilters>) => {
    setViewState(prev => ({
      ...prev,
      filters: { ...prev.filters, ...newFilters },
      page: 1,
    }));
  };

  const handleSortChange = (sortBy: string, sortOrder?: 'asc' | 'desc') => {
    setViewState(prev => ({
      ...prev,
      sortBy,
      sortOrder: sortOrder || (prev.sortBy === sortBy && prev.sortOrder === 'desc' ? 'asc' : 'desc'),
      page: 1,
    }));
  };

  const handlePageChange = (page: number) => {
    setViewState(prev => ({ ...prev, page }));
  };

  const handleSearch = async (term: string) => {
    setSearchTerm(term);
    if (term.trim()) {
      try {
        setLoading(true);
        const result = await searchExecutionData(
          term,
          viewState.viewMode,
          viewState.filters,
          1,
          viewState.pageSize
        );
        setData(result);
        setViewState(prev => ({ ...prev, page: 1 }));
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Search failed');
      } finally {
        setLoading(false);
      }
    } else {
      fetchData();
    }
  };

  const handleWorkflowClick = useCallback((workflow: WorkflowSummary) => {
    navigate(`/workflows/${workflow.run_id}`);
  }, [navigate]);

  const handleWorkflowsDeleted = useCallback(() => {
    // Refresh the data after workflows are deleted
    fetchData();
  }, [fetchData]);

  const handleExecutionClick = useCallback((execution: EnhancedExecution) => {
    navigate(`/executions/${execution.execution_id}`);
  }, [navigate]);

  const getViewModeIcon = (iconName: string) => {
    switch (iconName) {
      case 'pulse':
        return <Pulse className="h-4 w-4" />;
      case 'flow':
        return <FlowArrow className="h-4 w-4" />;
      case 'user':
        return <User className="h-4 w-4" />;
      case 'tree':
        return <TreeStructure className="h-4 w-4" />;
      default:
        return <Table className="h-4 w-4" />;
    }
  };

  const currentSortOptions = viewState.viewMode === 'workflows' ? WORKFLOW_SORT_OPTIONS : SORT_OPTIONS;

  return (
    <div className={cn("space-y-6", className)}>
      {/* Header */}
      <div className="space-y-2">
        <h2 className="text-display">Execution Monitor</h2>
        <p className="text-muted-foreground">
          Track and analyze workflow executions across your AI agent network
        </p>
      </div>

      {/* Stats Overview */}
      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Total</p>
          <p className="text-display">{stats.total_count}</p>
                </div>
                <Pulse className="h-8 w-8 text-muted-foreground" />
              </div>
            </CardContent>
          </Card>

          {Object.entries(stats.status_breakdown || {}).map(([status, count]) => (
            <Card key={status}>
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-muted-foreground capitalize">{status}</p>
                    <p className="text-display">{count as number}</p>
                  </div>
                  <Badge
                    variant={
                      status === 'completed' ? 'default' :
                      status === 'failed' ? 'destructive' :
                      status === 'running' ? 'secondary' : 'outline'
                    }
                    className="text-xs"
                  >
                    {status}
                  </Badge>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Controls */}
      <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-center justify-between">
        {/* Search */}
        <div className="w-full flex-1 max-w-md">
          <SearchBar
            value={searchTerm}
            onChange={handleSearch}
            placeholder="Search executions..."
            size="md"
            wrapperClassName="w-full"
            inputClassName="border border-border bg-background focus-visible:ring-1 focus-visible:ring-ring focus-visible:border-ring"
            clearButtonAriaLabel="Clear execution search"
          />
        </div>

        {/* Controls */}
        <div className="flex items-center gap-2">
          {/* Status Filter */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" size="sm">
                <FunnelSimple className="mr-2 h-4 w-4" />
                Status
                <CaretDown className="ml-2 h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              {STATUS_OPTIONS.map((option) => (
                <DropdownMenuItem
                  key={option.value}
                  onClick={() => handleFiltersChange({ status: option.value || undefined })}
                  className={cn(
                    viewState.filters.status === option.value && "bg-accent"
                  )}
                >
                  {option.label}
                </DropdownMenuItem>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>

          {/* Sort */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" size="sm">
                {viewState.sortOrder === 'asc' ? (
                  <SortAscending className="h-4 w-4 mr-2" />
                ) : (
                  <SortDescending className="h-4 w-4 mr-2" />
                )}
                Sort
                <CaretDown className="ml-2 h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              {currentSortOptions.map((option) => (
                <DropdownMenuItem
                  key={option.value}
                  onClick={() => handleSortChange(option.value)}
                  className={cn(
                    viewState.sortBy === option.value && "bg-accent"
                  )}
                >
                  {option.label}
                </DropdownMenuItem>
              ))}
              <DropdownMenuSeparator />
              <DropdownMenuItem
                onClick={() => handleSortChange(viewState.sortBy, 'asc')}
                className={cn(viewState.sortOrder === 'asc' && "bg-accent")}
              >
                <SortAscending className="h-4 w-4 mr-2" />
                Ascending
              </DropdownMenuItem>
              <DropdownMenuItem
                onClick={() => handleSortChange(viewState.sortBy, 'desc')}
                className={cn(viewState.sortOrder === 'desc' && "bg-accent")}
              >
                <SortDescending className="h-4 w-4 mr-2" />
                Descending
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>

      {/* View Mode Tabs */}
      <Tabs value={viewState.viewMode} onValueChange={(value) => handleViewModeChange(value as ViewMode['id'])}>
        <TabsList variant="underline" className="grid w-full grid-cols-4">
          {VIEW_MODES.map((mode) => (
            <TabsTrigger
              key={mode.id}
              value={mode.id}
              variant="underline"
              className="gap-2"
            >
              {getViewModeIcon(mode.icon)}
              <span className="hidden sm:inline">{mode.label}</span>
            </TabsTrigger>
          ))}
        </TabsList>

        {/* Content */}
        {VIEW_MODES.map((mode) => (
          <TabsContent key={mode.id} value={mode.id} className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-heading-3">{mode.label}</h3>
                <p className="text-body-small">{mode.description}</p>
              </div>
              {data && (
                <Badge variant="outline">
                  {data.total_count} {mode.label.toLowerCase()}
                </Badge>
              )}
            </div>

            {error && (
              <Card>
                <CardContent className="py-8">
                  <div className="text-center">
                    <div className="mb-2 text-status-error">Error loading data</div>
                    <div className="text-body-small">{error}</div>
                  </div>
                </CardContent>
              </Card>
            )}

            {mode.id === 'workflows' ? (
              <WorkflowsTable
                workflows={workflowsResponse?.workflows ?? []}
                loading={loading}
                hasMore={
                  Boolean(workflowsResponse?.has_more) ||
                  (workflowsResponse
                    ? workflowsResponse.page < workflowsResponse.total_pages
                    : false)
                }
                isFetchingMore={isFetchingMore}
                sortBy={viewState.sortBy}
                sortOrder={viewState.sortOrder}
                onSortChange={handleSortChange}
                onLoadMore={() => handlePageChange(viewState.page + 1)}
                onWorkflowClick={handleWorkflowClick}
                onWorkflowsDeleted={handleWorkflowsDeleted}
              />
            ) : (
              <EnhancedExecutionsTable
                executions={executionsResponse?.executions ?? []}
                loading={loading}
                hasMore={
                  Boolean(executionsResponse?.has_more) ||
                  (executionsResponse
                    ? executionsResponse.page < executionsResponse.total_pages
                    : false)
                }
                isFetchingMore={isFetchingMore}
                sortBy={viewState.sortBy}
                sortOrder={viewState.sortOrder}
                onSortChange={handleSortChange}
                onLoadMore={() => handlePageChange(viewState.page + 1)}
                onExecutionClick={handleExecutionClick}
              />
            )}
          </TabsContent>
        ))}
      </Tabs>
    </div>
  );
}
