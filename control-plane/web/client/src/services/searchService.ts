import { getNodesSummary } from './api';
import type { AgentNodeSummary } from '../types/agentfield';

export interface SearchResult {
  id: string;
  type: 'agent' | 'workflow' | 'execution' | 'reasoner' | 'package';
  title: string;
  subtitle?: string;
  description?: string;
  href: string;
  status?: string;
}

export type SearchCategory = 'agents' | 'workflows' | 'executions' | 'packages' | 'reasoners';

const MAX_RESULTS_PER_CATEGORY = 10;

class SearchService {
  private activeController: AbortController | null = null;

  public async search(
    query: string,
    categories: SearchCategory[],
    limit: number = MAX_RESULTS_PER_CATEGORY
  ): Promise<SearchResult[]> {
    if (this.activeController) {
      this.activeController.abort();
    }
    this.activeController = new AbortController();
    if (!query) {
      return [];
    }

    try {
      const results = await Promise.all([
        categories.includes('agents') ? this.searchAgents(query, limit) : Promise.resolve([]),
        categories.includes('executions') ? this.searchExecutions(query, limit) : Promise.resolve([]),
        categories.includes('workflows') ? this.searchWorkflows(query, limit) : Promise.resolve([]),
        categories.includes('packages') ? this.searchPackages(query, limit) : Promise.resolve([]),
        categories.includes('reasoners') ? this.searchReasoners(query, limit) : Promise.resolve([])
      ]);

      const flattenedResults = results.flat();
      return flattenedResults.sort((a, b) => a.title.localeCompare(b.title));
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        console.log('Search aborted');
        return [];
      }
      console.error('Search failed:', error);
      return [];
    } finally {
      this.activeController = null;
    }
  }

  private async searchAgents(query: string, limit: number): Promise<SearchResult[]> {
    try {
      const { nodes } = await getNodesSummary();
      const filteredNodes = nodes
        .filter((node: AgentNodeSummary) =>
          node.id.toLowerCase().includes(query.toLowerCase()) ||
          node.base_url.toLowerCase().includes(query.toLowerCase())
        )
        .slice(0, limit);

      return filteredNodes.map((node: AgentNodeSummary) => ({
        id: node.id,
        type: 'agent' as const,
        title: node.id,
        subtitle: `v${node.version}`,
        description: `Agent at ${node.base_url}`,
        href: `/nodes/${node.id}`,
        status: node.health_status,
      }));
    } catch (error) {
      console.error('Failed to search agents:', error);
      return [];
    }
  }

  private async searchExecutions(_query: string, _limit: number): Promise<SearchResult[]> {
    return [];
  }

  private async searchWorkflows(_query: string, _limit: number): Promise<SearchResult[]> {
    return [];
  }

  private async searchPackages(_query: string, _limit: number): Promise<SearchResult[]> {
    return [];
  }

  private async searchReasoners(_query: string, _limit: number): Promise<SearchResult[]> {
    return [];
  }
}

export const searchService = new SearchService();
