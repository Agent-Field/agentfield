import {
  Filter,
  InProgress,
  Package,
  Reset,
  Search,
} from "@/components/ui/icon-bridge";
import React, { useEffect, useMemo, useState } from "react";
import {
  ConfigurationApiError,
  getAgentPackages,
  getRunningAgents,
} from "../../services/configurationApi";
import type { AgentLifecycleInfo, AgentPackage } from "../../types/haxen";
import { Badge } from "../ui/badge";
import { Button } from "../ui/button";
import { Card, CardContent, CardHeader } from "../ui/card";
import { Input } from "../ui/input";
import { Skeleton } from "../ui/skeleton";
import { ResponsiveGrid } from "../layout/ResponsiveGrid";
import { AgentPackageCard } from "./AgentPackageCard";

interface AgentPackageListProps {
  onConfigure: (pkg: AgentPackage) => void;
  onStart: (pkg: AgentPackage) => void;
  onStop: (pkg: AgentPackage) => void;
}

type FilterType = "all" | "configured" | "not_configured" | "running";

export const AgentPackageList: React.FC<AgentPackageListProps> = ({
  onConfigure,
  onStart,
  onStop,
}) => {
  const [packages, setPackages] = useState<AgentPackage[]>([]);
  const [runningAgents, setRunningAgents] = useState<AgentLifecycleInfo[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [activeFilter, setActiveFilter] = useState<FilterType>("all");
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const filters: { type: FilterType; label: string; count?: number }[] = [
    { type: "all", label: "All Packages" },
    { type: "configured", label: "Configured" },
    { type: "not_configured", label: "Not Configured" },
    { type: "running", label: "Running" },
  ];

  const loadData = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const [packagesData, runningData] = await Promise.all([
        getAgentPackages(searchQuery || undefined),
        getRunningAgents(),
      ]);

      setPackages(packagesData);
      setRunningAgents(runningData);
    } catch (err) {
      const errorMessage =
        err instanceof ConfigurationApiError
          ? err.message
          : "Failed to load agent packages";
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, [searchQuery]);

  // Create a map of running agents by ID for quick lookup
  const runningAgentsMap = useMemo(() => {
    const map = new Map<string, AgentLifecycleInfo>();
    runningAgents.forEach((agent) => {
      map.set(agent.id, agent);
    });
    return map;
  }, [runningAgents]);

  // Filter packages based on active filter
  const filteredPackages = useMemo(() => {
    let filtered = packages;

    switch (activeFilter) {
      case "configured":
        filtered = packages.filter(
          (pkg) => pkg.configuration_status === "configured"
        );
        break;
      case "not_configured":
        filtered = packages.filter(
          (pkg) => pkg.configuration_status === "not_configured"
        );
        break;
      case "running":
        filtered = packages.filter((pkg) => {
          const agentStatus = runningAgentsMap.get(pkg.id);
          return agentStatus?.status === "running";
        });
        break;
      default:
        // 'all' - no additional filtering
        break;
    }

    return filtered;
  }, [packages, activeFilter, runningAgentsMap]);

  // Update filter counts
  const filtersWithCounts = useMemo(() => {
    return filters.map((filter) => {
      let count: number;
      switch (filter.type) {
        case "all":
          count = packages.length;
          break;
        case "configured":
          count = packages.filter(
            (pkg) => pkg.configuration_status === "configured"
          ).length;
          break;
        case "not_configured":
          count = packages.filter(
            (pkg) => pkg.configuration_status === "not_configured"
          ).length;
          break;
        case "running":
          count = packages.filter((pkg) => {
            const agentStatus = runningAgentsMap.get(pkg.id);
            return agentStatus?.status === "running";
          }).length;
          break;
        default:
          count = 0;
      }
      return { ...filter, count };
    });
  }, [packages, runningAgentsMap]);

  const handleRefresh = () => {
    loadData();
  };

  const handleSearch = (value: string) => {
    setSearchQuery(value);
  };

  if (error) {
    return (
      <Card className="w-full">
        <CardContent className="flex flex-col items-center justify-center py-8">
          <Package className="h-12 w-12 text-gray-400 mb-4" />
          <h3 className="text-heading-3 text-text-heading mb-2">
            Error Loading Packages
          </h3>
          <p className="text-text-secondary text-center mb-4">{error}</p>
          <Button onClick={handleRefresh} variant="outline">
            <Reset className="h-4 w-4 mr-2" />
            Try Again
          </Button>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-display">Agent Packages</h1>
          <p className="text-secondary">
            Manage your installed agent packages and configurations
          </p>
        </div>
        <Button onClick={handleRefresh} variant="outline" disabled={isLoading}>
          {isLoading ? (
            <InProgress className="h-4 w-4 animate-spin mr-2" />
          ) : (
            <Reset className="h-4 w-4 mr-2" />
          )}
          Refresh
        </Button>
      </div>

      {/* Search and Filters */}
      <div className="flex flex-col sm:flex-row gap-4">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
          <Input
            placeholder="Search packages..."
            value={searchQuery}
            onChange={(e) => handleSearch(e.target.value)}
            className="pl-10"
          />
        </div>
        <div className="flex items-center gap-2">
          <Filter className="h-4 w-4 text-gray-500" />
          <div className="flex gap-1">
            {filtersWithCounts.map((filter) => (
              <Button
                key={filter.type}
                variant={activeFilter === filter.type ? "default" : "outline"}
                size="sm"
                onClick={() => setActiveFilter(filter.type)}
                className="text-xs"
              >
                {filter.label}
                {filter.count !== undefined && (
                  <Badge variant="secondary" className="ml-2 text-xs">
                    {filter.count}
                  </Badge>
                )}
              </Button>
            ))}
          </div>
        </div>
      </div>

      {/* Package Grid */}
      {isLoading ? (
        <ResponsiveGrid columns={{ base: 1, md: 2, lg: 3 }} gap="md" align="start">
          {[...Array(6)].map((_, i) => (
            <Card key={i}>
              <CardHeader>
                <div className="flex items-start gap-3">
                  <Skeleton className="h-10 w-10 rounded-lg" />
                  <div className="flex-1 space-y-2">
                    <Skeleton className="h-5 w-3/4" />
                    <Skeleton className="h-4 w-2/3" />
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-3">
                <Skeleton className="h-4" />
                <Skeleton className="h-4 w-2/3" />
                <div className="flex gap-2">
                  <Skeleton className="h-8 flex-1 rounded" />
                  <Skeleton className="h-8 flex-1 rounded" />
                </div>
              </CardContent>
            </Card>
          ))}
        </ResponsiveGrid>
      ) : filteredPackages.length === 0 ? (
        <Card className="w-full">
          <CardContent className="flex flex-col items-center justify-center py-12">
            <Package className="h-16 w-16 text-gray-400 mb-4" />
            <h3 className="text-heading-3 text-text-heading mb-2">
              {searchQuery ? "No packages found" : "No packages installed"}
            </h3>
            <p className="text-text-secondary text-center">
              {searchQuery
                ? `No packages match "${searchQuery}". Try adjusting your search or filters.`
                : "Install agent packages using the CLI to get started."}
            </p>
          </CardContent>
        </Card>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredPackages.map((pkg) => (
            <AgentPackageCard
              key={pkg.id}
              package={pkg}
              agentStatus={runningAgentsMap.get(pkg.id)}
              onConfigure={onConfigure}
              onStart={onStart}
              onStop={onStop}
              isLoading={isLoading}
            />
          ))}
        </div>
      )}
    </div>
  );
};
