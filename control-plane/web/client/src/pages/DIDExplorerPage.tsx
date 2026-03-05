import { useCallback, useEffect, useState } from "react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
    Identification,
    Function,
    Renew,
    Terminal,
    Copy,
} from "@/components/ui/icon-bridge";
import { CompactTable } from "@/components/ui/CompactTable";
import { SearchBar } from "@/components/ui/SearchBar";
import { PageHeader } from "../components/PageHeader";
import * as identityApi from "../services/identityApi";
import type {
    DIDSearchResult,
    AgentDIDResponse,
    ComponentDIDInfo,
} from "../services/identityApi";

const ITEMS_PER_PAGE = 50;
const GRID_TEMPLATE =
    "80px minmax(200px,1fr) minmax(300px,2fr) minmax(150px,1fr) 120px 80px";
const GRID_TEMPLATE_REASONERS =
    "minmax(200px,1fr) minmax(350px,2fr) 100px 140px";

export function DIDExplorerPage() {
    // State
    const [searchQuery, setSearchQuery] = useState("");
    const [searchResults, setSearchResults] = useState<DIDSearchResult[]>([]);
    const [recentAgents, setRecentAgents] = useState<AgentDIDResponse[]>([]);
    const [selectedAgent, setSelectedAgent] = useState<AgentDIDResponse | null>(
        null,
    );
    const [selectedAgentReasoners, setSelectedAgentReasoners] = useState<
        ComponentDIDInfo[]
    >([]);

    // Loading states
    const [loadingSearch, setLoadingSearch] = useState(false);
    const [loadingAgents, setLoadingAgents] = useState(false);
    const [loadingReasoners, setLoadingReasoners] = useState(false);

    // Pagination
    const [searchOffset, setSearchOffset] = useState(0);
    const [reasonersOffset, setReasonersOffset] = useState(0);
    const [hasMoreSearch, setHasMoreSearch] = useState(false);
    const [hasMoreReasoners, setHasMoreReasoners] = useState(false);

    const [error, setError] = useState<string | null>(null);
    const [_, setCopiedDID] = useState<string | null>(null);

    // Fetch recent agents
    const fetchRecentAgents = useCallback(async () => {
        try {
            setLoadingAgents(true);
            const data = await identityApi.listAgents(20, 0);
            setRecentAgents(data.agents || []);
        } catch (err) {
            console.error("Failed to fetch recent agents:", err);
            setError(
                err instanceof Error ? err.message : "Failed to fetch agents",
            );
        } finally {
            setLoadingAgents(false);
        }
    }, []);

    // Search DIDs
    const performSearch = useCallback(
        async (query: string, offset: number = 0) => {
            if (!query.trim()) {
                setSearchResults([]);
                return;
            }

            try {
                setLoadingSearch(true);
                setError(null);
                const data = await identityApi.searchDIDs(
                    query,
                    "all",
                    ITEMS_PER_PAGE,
                    offset,
                );

                if (offset === 0) {
                    setSearchResults(data.results || []);
                } else {
                    setSearchResults((prev) => [
                        ...prev,
                        ...(data.results || []),
                    ]);
                }

                setHasMoreSearch((data.total || 0) > offset + ITEMS_PER_PAGE);
                setSearchOffset(offset);
            } catch (err) {
                console.error("Failed to search DIDs:", err);
                setError(err instanceof Error ? err.message : "Search failed");
                setSearchResults([]);
            } finally {
                setLoadingSearch(false);
            }
        },
        [],
    );

    // Fetch agent reasoners
    const fetchAgentReasoners = useCallback(
        async (agentId: string, offset: number = 0) => {
            try {
                setLoadingReasoners(true);
                const data = await identityApi.getAgentDetails(
                    agentId,
                    ITEMS_PER_PAGE,
                    offset,
                );

                if (offset === 0) {
                    setSelectedAgentReasoners(data.agent.reasoners || []);
                } else {
                    setSelectedAgentReasoners((prev) => [
                        ...prev,
                        ...(data.agent.reasoners || []),
                    ]);
                }

                setHasMoreReasoners(data.reasoners_has_more);
                setReasonersOffset(offset);
            } catch (err) {
                console.error("Failed to fetch agent reasoners:", err);
            } finally {
                setLoadingReasoners(false);
            }
        },
        [],
    );

    // Initial load
    useEffect(() => {
        fetchRecentAgents();
    }, [fetchRecentAgents]);

    // Search when query changes
    useEffect(() => {
        const timer = setTimeout(() => {
            if (searchQuery.trim()) {
                performSearch(searchQuery, 0);
            } else {
                setSearchResults([]);
            }
        }, 300);

        return () => clearTimeout(timer);
    }, [searchQuery, performSearch]);

    // Handlers
    const handleAgentClick = (agent: AgentDIDResponse) => {
        setSelectedAgent(agent);
        fetchAgentReasoners(agent.agent_node_id, 0);
    };

    const handleCopyDID = async (did: string) => {
        try {
            await navigator.clipboard.writeText(did);
            setCopiedDID(did);
            setTimeout(() => setCopiedDID(null), 2000);
        } catch (err) {
            console.error("Failed to copy DID:", err);
        }
    };

    const handleRefresh = () => {
        if (searchQuery.trim()) {
            performSearch(searchQuery, 0);
        } else {
            fetchRecentAgents();
        }
    };

    const handleBackToList = () => {
        setSelectedAgent(null);
        setSelectedAgentReasoners([]);
    };

    // Table columns for search results
    const searchColumns = [
        {
            key: "type",
            header: "Type",
            sortable: true,
            align: "left" as const,
            render: (result: DIDSearchResult) => (
                <Badge variant="outline" size="sm" className="capitalize">
                    {result.type}
                </Badge>
            ),
        },
        {
            key: "name",
            header: "Name",
            sortable: true,
            align: "left" as const,
            render: (result: DIDSearchResult) => (
                <span className="font-medium">{result.name}</span>
            ),
        },
        {
            key: "did",
            header: "DID",
            sortable: false,
            align: "left" as const,
            render: (result: DIDSearchResult) => (
                <code className="text-xs font-mono text-muted-foreground truncate block">
                    {result.did}
                </code>
            ),
        },
        {
            key: "parent_name",
            header: "Parent",
            sortable: false,
            align: "left" as const,
            render: (result: DIDSearchResult) => (
                <span className="text-sm text-muted-foreground">
                    {result.parent_name || "—"}
                </span>
            ),
        },
        {
            key: "derivation_path",
            header: "Path",
            sortable: false,
            align: "left" as const,
            render: (result: DIDSearchResult) => (
                <code className="text-xs">{result.derivation_path}</code>
            ),
        },
        {
            key: "actions",
            header: "",
            sortable: false,
            align: "center" as const,
            render: (result: DIDSearchResult) => (
                <Button
                    variant="ghost"
                    size="icon"
                    className="h-7 w-7"
                    onClick={(e) => {
                        e.stopPropagation();
                        handleCopyDID(result.did);
                    }}
                    title="Copy DID"
                >
                    <Copy className="w-3.5 h-3.5" />
                </Button>
            ),
        },
    ];

    // Table columns for reasoners
    const reasonerColumns = [
        {
            key: "name",
            header: "Reasoner Name",
            sortable: true,
            align: "left" as const,
            render: (reasoner: ComponentDIDInfo) => (
                <span className="font-medium">{reasoner.name}</span>
            ),
        },
        {
            key: "did",
            header: "DID",
            sortable: false,
            align: "left" as const,
            render: (reasoner: ComponentDIDInfo) => (
                <div className="flex items-center gap-2 min-w-0">
                    <code className="text-xs font-mono text-muted-foreground truncate block">
                        {reasoner.did}
                    </code>
                    <Button
                        variant="ghost"
                        size="icon"
                        className="h-6 w-6 flex-shrink-0"
                        onClick={(e) => {
                            e.stopPropagation();
                            handleCopyDID(reasoner.did);
                        }}
                        title="Copy DID"
                    >
                        <Copy className="w-3 h-3" />
                    </Button>
                </div>
            ),
        },
        {
            key: "derivation_path",
            header: "Index",
            sortable: false,
            align: "left" as const,
            render: (reasoner: ComponentDIDInfo, index?: number) => {
                // Use the array index for proper sequential numbering
                return (
                    <span className="text-sm text-muted-foreground">
                        #{index !== undefined ? index : reasoner.derivation_path}
                    </span>
                );
            },
        },
        {
            key: "created_at",
            header: "Created",
            sortable: true,
            align: "left" as const,
            render: (reasoner: ComponentDIDInfo) => (
                <span className="text-sm text-muted-foreground">
                    {new Date(reasoner.created_at).toLocaleDateString()}
                </span>
            ),
        },
    ];

    return (
        <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
            <div className="px-6 pt-6 pb-4">
                <PageHeader
                    title={selectedAgent ? selectedAgent.agent_node_id : "DID Explorer"}
                    description={
                        selectedAgent
                            ? `Viewing reasoners for agent ${selectedAgent.agent_node_id}`
                            : "Explore decentralized identifiers for agents and reasoners"
                    }
                    aside={
                        <div className="flex items-center gap-2">
                            {selectedAgent && (
                                <Button
                                    variant="ghost"
                                    size="sm"
                                    onClick={handleBackToList}
                                >
                                    ← Back to Agents
                                </Button>
                            )}
                            <Button
                                variant="outline"
                                size="sm"
                                onClick={handleRefresh}
                                disabled={loadingSearch || loadingAgents}
                                className="flex items-center gap-2"
                            >
                                <Renew
                                    size={14}
                                    className={
                                        loadingSearch || loadingAgents
                                            ? "animate-spin"
                                            : ""
                                    }
                                />
                                Refresh
                            </Button>
                        </div>
                    }
                />
            </div>

            {/* Content Area */}
            <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
                {/* Error Alert */}
                {error && (
                    <Alert variant="destructive" className="mb-4 mx-6">
                        <Terminal className="h-4 w-4" />
                        <AlertTitle>Error</AlertTitle>
                        <AlertDescription>{error}</AlertDescription>
                    </Alert>
                )}
                {selectedAgent ? (
                    // Agent Detail View
                    <div className="flex min-h-0 flex-1 flex-col gap-6 overflow-hidden px-6 pb-6">
                        <div className="bg-card border border-border rounded-lg p-6 flex-shrink-0">
                            <div className="flex items-start justify-between mb-4">
                                <div className="flex items-center gap-3">
                                    <Identification
                                        size={20}
                                        className="text-primary"
                                    />
                                    <div className="min-w-0">
                                        <h2 className="text-lg font-semibold">
                                            {selectedAgent.agent_node_id}
                                        </h2>
                                        <div className="space-y-1.5 mt-2">
                                            <div className="flex items-center gap-2">
                                                <span className="text-xs text-muted-foreground font-medium w-20 flex-shrink-0">did:key</span>
                                                <code className="text-xs text-muted-foreground font-mono truncate">
                                                    {selectedAgent.did}
                                                </code>
                                                <Button
                                                    variant="ghost"
                                                    size="icon"
                                                    className="h-5 w-5 flex-shrink-0"
                                                    onClick={() =>
                                                        handleCopyDID(
                                                            selectedAgent.did,
                                                        )
                                                    }
                                                    title="Copy Crypto DID"
                                                >
                                                    <Copy className="w-3 h-3" />
                                                </Button>
                                            </div>
                                            {selectedAgent.did_web && (
                                                <div className="flex items-center gap-2">
                                                    <span className="text-xs text-muted-foreground font-medium w-20 flex-shrink-0">did:web</span>
                                                    <code className="text-xs text-muted-foreground font-mono truncate">
                                                        {selectedAgent.did_web}
                                                    </code>
                                                    <Button
                                                        variant="ghost"
                                                        size="icon"
                                                        className="h-5 w-5 flex-shrink-0"
                                                        onClick={() =>
                                                            handleCopyDID(
                                                                selectedAgent.did_web!,
                                                            )
                                                        }
                                                        title="Copy Web DID"
                                                    >
                                                        <Copy className="w-3 h-3" />
                                                    </Button>
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                </div>
                                <Badge variant="outline" size="sm">
                                    {selectedAgent.status}
                                </Badge>
                            </div>

                            <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
                                <div>
                                    <span className="text-muted-foreground">
                                        Reasoners:
                                    </span>
                                    <div className="mt-1 font-medium">
                                        {selectedAgentReasoners.length}
                                    </div>
                                </div>
                                <div>
                                    <span className="text-muted-foreground">
                                        Skills:
                                    </span>
                                    <div className="mt-1 font-medium">
                                        {selectedAgent.skill_count}
                                    </div>
                                </div>
                                <div>
                                    <span className="text-muted-foreground">
                                        Created:
                                    </span>
                                    <div className="mt-1">
                                        {new Date(
                                            selectedAgent.created_at,
                                        ).toLocaleDateString()}
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Reasoners Table */}
                        <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
                            <h3 className="text-lg font-semibold mb-4 flex-shrink-0">
                                Reasoners ({selectedAgentReasoners.length})
                            </h3>
                            <div className="flex-1 overflow-hidden">
                                <CompactTable
                                data={selectedAgentReasoners}
                                loading={
                                    loadingReasoners &&
                                    selectedAgentReasoners.length === 0
                                }
                                hasMore={hasMoreReasoners}
                                isFetchingMore={
                                    loadingReasoners &&
                                    selectedAgentReasoners.length > 0
                                }
                                sortBy="name"
                                sortOrder="asc"
                                onSortChange={() => {}}
                                onLoadMore={() =>
                                    fetchAgentReasoners(
                                        selectedAgent.agent_node_id,
                                        reasonersOffset + ITEMS_PER_PAGE,
                                    )
                                }
                                columns={reasonerColumns}
                                gridTemplate={GRID_TEMPLATE_REASONERS}
                                emptyState={{
                                    title: "No reasoners found",
                                    description:
                                        "This agent doesn't have any reasoners yet.",
                                    icon: (
                                        <Function className="h-6 w-6 text-muted-foreground" />
                                    ),
                                }}
                                    getRowKey={(reasoner) => reasoner.did}
                                />
                            </div>
                        </div>
                    </div>
                ) : (
                    // Search & Browse View
                    <div className="flex min-h-0 flex-1 flex-col gap-6 overflow-hidden px-6 pb-6">
                        <SearchBar
                            value={searchQuery}
                            onChange={setSearchQuery}
                            placeholder="Search by name, DID, or type..."
                            wrapperClassName="w-full lg:max-w-md flex-shrink-0"
                        />

                        <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
                            {searchQuery.trim() ? (
                                // Search Results
                                <CompactTable
                                    data={searchResults}
                                    loading={
                                        loadingSearch &&
                                        searchResults.length === 0
                                    }
                                    hasMore={hasMoreSearch}
                                    isFetchingMore={
                                        loadingSearch &&
                                        searchResults.length > 0
                                    }
                                    sortBy="name"
                                    sortOrder="asc"
                                    onSortChange={() => {}}
                                    onLoadMore={() =>
                                        performSearch(
                                            searchQuery,
                                            searchOffset + ITEMS_PER_PAGE,
                                        )
                                    }
                                    onRowClick={(result) => {
                                        if (result.type === "agent") {
                                            const agent = recentAgents.find(
                                                (a) =>
                                                    a.agent_node_id ===
                                                    result.name,
                                            );
                                            if (agent) handleAgentClick(agent);
                                        }
                                    }}
                                    columns={searchColumns}
                                    gridTemplate={GRID_TEMPLATE}
                                    emptyState={{
                                        title: "No results found",
                                        description: `No DIDs match "${searchQuery}"`,
                                        icon: (
                                            <Identification className="h-6 w-6 text-muted-foreground" />
                                        ),
                                    }}
                                    getRowKey={(result) => result.did}
                                />
                            ) : (
                                // Recent Agents
                                <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
                                    <h3 className="text-lg font-semibold mb-4 flex-shrink-0">
                                        Recent Agents
                                    </h3>
                                    {loadingAgents ? (
                                        <div className="text-center py-12 text-muted-foreground">
                                            Loading agents...
                                        </div>
                                    ) : recentAgents.length === 0 ? (
                                        <div className="bg-card border border-border rounded-lg p-12 text-center">
                                            <Identification
                                                size={48}
                                                className="mx-auto mb-4 text-muted-foreground"
                                            />
                                            <p className="text-muted-foreground">
                                                No agents found
                                            </p>
                                        </div>
                                    ) : (
                                        <div className="flex-1 overflow-y-auto space-y-2">
                                            {recentAgents.map((agent) => (
                                                <div
                                                    key={agent.agent_node_id}
                                                    className="bg-card border border-border rounded-lg p-4 hover:bg-muted/50 transition-colors cursor-pointer"
                                                    onClick={() =>
                                                        handleAgentClick(agent)
                                                    }
                                                >
                                                    <div className="flex items-center justify-between">
                                                        <div className="flex items-center gap-3 flex-1 min-w-0">
                                                            <Identification
                                                                size={16}
                                                                className="text-primary flex-shrink-0"
                                                            />
                                                            <div className="flex-1 min-w-0">
                                                                <p className="font-medium">
                                                                    {
                                                                        agent.agent_node_id
                                                                    }
                                                                </p>
                                                                <code className="text-xs text-muted-foreground font-mono truncate block">
                                                                    {agent.did}
                                                                </code>
                                                            </div>
                                                        </div>
                                                        <Badge
                                                            variant="outline"
                                                            size="sm"
                                                        >
                                                            {
                                                                agent.reasoner_count
                                                            }{" "}
                                                            reasoners
                                                        </Badge>
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    )}
                                </div>
                            )}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
