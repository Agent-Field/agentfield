import React, { useState } from 'react';
import type { AgentNodeSummary, AgentNode } from '../types/haxen';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Button } from '@/components/ui/button';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { ChevronDown, ChevronUp, ServerProxy, Time, Earth, Network_3 } from '@/components/ui/icon-bridge';
import { getNodeDetails } from '../services/api';
import StatusIndicator from './ui/status-indicator';
import ReasonersList from './ReasonersList';
import SkillsList from './SkillsList';
import { Skeleton } from '@/components/ui/skeleton';
import { Separator } from '@/components/ui/separator';

interface AgentNodesTableProps {
  nodes: AgentNodeSummary[];
  isLoading: boolean;
  error: string | null;
}

const AgentNodesTable: React.FC<AgentNodesTableProps> = ({ nodes, isLoading, error }) => {
  if (isLoading && nodes.length === 0) {
    return (
      <div className="rounded-md border">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-[50px]"></TableHead>
              <TableHead>Node ID</TableHead>
              <TableHead>Team</TableHead>
              <TableHead>Version</TableHead>
              <TableHead>Health</TableHead>
              <TableHead>Reasoners</TableHead>
              <TableHead>Skills</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {Array.from({ length: 5 }).map((_, i) => (
              <TableRow key={i}>
                <TableCell><Skeleton className="h-4 w-4" /></TableCell>
                <TableCell><Skeleton className="h-4 w-[150px]" /></TableCell>
                <TableCell><Skeleton className="h-4 w-[100px]" /></TableCell>
                <TableCell><Skeleton className="h-4 w-[80px]" /></TableCell>
                <TableCell><Skeleton className="h-4 w-[60px]" /></TableCell>
                <TableCell><Skeleton className="h-4 w-[70px]" /></TableCell>
                <TableCell><Skeleton className="h-4 w-[70px]" /></TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    );
  }

  if (error) {
    return <p className="text-center text-red-500">{error}</p>;
  }

  if (nodes.length === 0) {
    return (
      <div className="text-center py-12">
        <div className="mx-auto w-24 h-24 bg-muted rounded-full flex items-center justify-center mb-4">
          <Network_3 className="h-12 w-12 text-muted-foreground" />
        </div>
        <h3 className="text-heading-3 mb-2">No Agent Nodes</h3>
        <p className="text-muted-foreground">No agent nodes are currently registered with the Haxen server.</p>
      </div>
    );
  }

  return (
    <div className="rounded-lg border bg-card">
      <Table>
        <TableHeader>
          <TableRow className="border-b">
            <TableHead className="w-[50px]"></TableHead>
            <TableHead className="font-medium">Node ID</TableHead>
            <TableHead className="font-medium">Team</TableHead>
            <TableHead className="font-medium">Version</TableHead>
            <TableHead className="font-medium">Status</TableHead>
            <TableHead className="font-medium">Reasoners</TableHead>
            <TableHead className="font-medium">Skills</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {nodes.map((nodeSummary) => (
            <NodeRow key={nodeSummary.id} nodeSummary={nodeSummary} />
          ))}
        </TableBody>
      </Table>
    </div>
  );
};

interface NodeRowProps {
  nodeSummary: AgentNodeSummary;
}

const NodeRow: React.FC<NodeRowProps> = ({ nodeSummary }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [nodeDetails, setNodeDetails] = useState<AgentNode | null>(null);
  const [isLoadingDetails, setIsLoadingDetails] = useState(false);
  const [errorDetails, setErrorDetails] = useState<string | null>(null);

  const toggleExpand = () => {
    setIsExpanded(!isExpanded);
    if (!isExpanded && !nodeDetails && !isLoadingDetails) {
      setIsLoadingDetails(true);
      setErrorDetails(null);
      getNodeDetails(nodeSummary.id)
        .then(setNodeDetails)
        .catch((err) => {
          console.error('Failed to load node details:', err);
          setErrorDetails('Failed to load details.');
        })
        .finally(() => setIsLoadingDetails(false));
    }
  };

  return (
    <Collapsible
      asChild
      open={isExpanded}
      onOpenChange={setIsExpanded}
    >
      <>
        <TableRow>
          <TableCell>
            <CollapsibleTrigger asChild>
              <Button
                variant="ghost"
                  onClick={toggleExpand}
                className="h-8 w-8 p-0"
              >
                {isExpanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                <span className="sr-only">Toggle</span>
              </Button>
            </CollapsibleTrigger>
          </TableCell>
          <TableCell className="font-medium">{nodeSummary.id}</TableCell>
          <TableCell>{nodeSummary.team_id}</TableCell>
          <TableCell>{nodeSummary.version}</TableCell>
          <TableCell>
            <StatusIndicator 
              status={nodeSummary.lifecycle_status} 
              healthStatus={nodeSummary.health_status}
            />
          </TableCell>
          <TableCell>{nodeSummary.reasoner_count}</TableCell>
          <TableCell>{nodeSummary.skill_count}</TableCell>
        </TableRow>
        <CollapsibleContent asChild>
          <TableRow>
            <TableCell colSpan={7} className="p-0">
              <div className="p-6 bg-muted/30 border-t">
                {isLoadingDetails && (
                  <div className="flex items-center gap-2">
                    <div className="h-4 w-4 animate-spin rounded-full border-2 border-primary border-t-transparent" />
                    <p className="text-body-small">Loading node details...</p>
                  </div>
                )}
                {errorDetails && (
                  <p className="text-sm text-destructive">{errorDetails}</p>
                )}
                {nodeDetails && (
                  <div className="space-y-6">
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div className="flex items-center gap-2">
                        <Earth className="h-4 w-4 text-muted-foreground" />
                        <div>
                          <p className="text-body-small">Base URL</p>
                          <p className="text-sm font-mono">{nodeDetails.base_url}</p>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <ServerProxy className="h-4 w-4 text-muted-foreground" />
                        <div>
                          <p className="text-body-small">Registered</p>
                          <p className="text-sm">{nodeDetails.registered_at ? (() => {
                            const date = new Date(nodeDetails.registered_at);
                            return !isNaN(date.getTime()) ? date.toLocaleString() : 'Invalid Date';
                          })() : 'N/A'}</p>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <Time className="h-4 w-4 text-muted-foreground" />
                        <div>
                          <p className="text-body-small">Last Heartbeat</p>
                          <p className="text-sm">{nodeDetails.last_heartbeat ? (() => {
                            const date = new Date(nodeDetails.last_heartbeat);
                            return !isNaN(date.getTime()) ? date.toLocaleString() : 'Invalid Date';
                          })() : 'N/A'}</p>
                        </div>
                      </div>
                    </div>
                    
                    <Separator />

                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                      <ReasonersList reasoners={nodeDetails.reasoners ?? []} />
                      <SkillsList skills={nodeDetails.skills ?? []} />
                    </div>
                  </div>
                )}
              </div>
            </TableCell>
          </TableRow>
        </CollapsibleContent>
      </>
    </Collapsible>
  );
};

export default AgentNodesTable;
