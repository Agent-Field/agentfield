export interface ReasonerWithNode {
  // Reasoner identification
  reasoner_id: string;   // Format: "node_id.reasoner_id"
  name: string;          // Human-readable name
  description: string;   // Reasoner description

  // Node context
  node_id: string;
  node_status: 'active' | 'inactive' | 'unknown';
  node_version: string;

  // Reasoner details
  input_schema: any;
  output_schema: any;
  memory_config: {
    auto_inject: string[];
    memory_retention: string;
    cache_results: boolean;
  };
  tags?: string[];

  // Performance metrics (optional)
  avg_response_time_ms?: number;
  success_rate?: number;
  total_runs?: number;
  last_executed?: string;

  // Timestamps
  last_updated: string;
}

export interface ReasonersResponse {
  reasoners: ReasonerWithNode[];
  total: number;
  online_count: number;
  offline_count: number;
  nodes_count: number;
}

export interface ReasonerFilters {
  status?: 'all' | 'online' | 'offline';
  search?: string;
  limit?: number;
  offset?: number;
}

export type ReasonerStatus = 'online' | 'degraded' | 'offline' | 'unknown';

export interface ReasonerCardProps {
  reasoner: ReasonerWithNode;
  onClick?: (reasoner: ReasonerWithNode) => void;
}

export interface ReasonerGridProps {
  reasoners: ReasonerWithNode[];
  loading?: boolean;
  onReasonerClick?: (reasoner: ReasonerWithNode) => void;
  viewMode?: 'grid' | 'table';
}

export interface SearchFiltersProps {
  filters: ReasonerFilters;
  onFiltersChange: (filters: ReasonerFilters) => void;
  totalCount: number;
  onlineCount: number;
  offlineCount: number;
}
