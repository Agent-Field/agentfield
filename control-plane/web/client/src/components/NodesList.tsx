import React, { useState, useEffect, useCallback } from 'react';
import type { AgentNodeSummary } from '../types/haxen';
import { getNodesSummary, streamNodeEvents } from '../services/api';
import AgentNodesTable from './AgentNodesTable';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Terminal } from '@/components/ui/icon-bridge';

interface NodeEvent {
  type: string;
  node: AgentNodeSummary | { id: string };
  timestamp: string;
}

const NodesList: React.FC = () => {
  const [nodes, setNodes] = useState<AgentNodeSummary[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [totalCount, setTotalCount] = useState(0);
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'disconnected' | 'connecting'>('connecting');

  const fetchNodes = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await getNodesSummary();
      setNodes(data.nodes);
      setTotalCount(data.count);
    } catch (err) {
      console.error('Failed to load nodes summary:', err);
      setError('Failed to load agent nodes. Please ensure the Haxen server is running and accessible.');
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchNodes();

    const eventSource = streamNodeEvents();

    eventSource.onopen = () => {
      setConnectionStatus('connected');
      console.log('SSE connection opened.');
    };

    eventSource.onerror = (err) => {
      setConnectionStatus('disconnected');
      console.error('SSE error:', err);
      eventSource.close(); // Attempt to close and let browser retry
    };

    eventSource.addEventListener('node_registered', (event) => {
      const data: NodeEvent = JSON.parse(event.data);
      console.log('Node registered event:', data);
      setNodes(prevNodes => {
        const newNode = data.node as AgentNodeSummary;
        // Check if node already exists (for updates)
        const existingIndex = prevNodes.findIndex(node => node.id === newNode.id);
        if (existingIndex > -1) {
          const updatedNodes = [...prevNodes];
          updatedNodes[existingIndex] = newNode;
          return updatedNodes;
        }
        setTotalCount(prevCount => prevCount + 1);
        return [...prevNodes, newNode];
      });
    });

    eventSource.addEventListener('node_health_changed', (event) => {
      const data: NodeEvent = JSON.parse(event.data);
      console.log('Node health changed event:', data);
      setNodes(prevNodes => {
        const updatedNode = data.node as AgentNodeSummary;
        return prevNodes.map(node =>
          node.id === updatedNode.id ? updatedNode : node
        );
      });
    });

    eventSource.addEventListener('node_removed', (event) => {
      const data: NodeEvent = JSON.parse(event.data);
      console.log('Node removed event:', data);
      setNodes(prevNodes => {
        setTotalCount(prevCount => Math.max(0, prevCount - 1));
        return prevNodes.filter(node => node.id !== (data.node as { id: string }).id);
      });
    });

    eventSource.addEventListener('heartbeat', (event) => {
      console.log('SSE Heartbeat:', event.data);
    });

    return () => {
      eventSource.close();
      console.log('SSE connection closed.');
    };
  }, [fetchNodes]);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <h3 className="text-heading-3">
            Registered Nodes ({totalCount})
          </h3>
          <Badge variant={connectionStatus === 'connected' ? 'default' : 'destructive'}>
            {connectionStatus === 'connected' ? 'Live Updates' : 'Disconnected'}
          </Badge>
        </div>
      </div>
      
      {error && (
        <Alert variant="destructive">
          <Terminal className="h-4 w-4" />
          <AlertTitle>Connection Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}
      
      <AgentNodesTable nodes={nodes} isLoading={isLoading} error={error} />
    </div>
  );
};

export default NodesList;
