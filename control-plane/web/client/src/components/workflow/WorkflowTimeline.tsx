import {
  Events,
  ExpandAll,
  CollapseAll,
  Filter,
  ArrowDown,
  ArrowUp,
} from "@/components/ui/icon-bridge";
import { useCallback, useEffect, useState } from "react";
import type { ExecutionNote } from "../../types/notes";
import { Button } from "../ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import { TagBadge } from "../notes/TagBadge";
import { TimelineNodeCard, TimelineNodeCardSkeleton } from "./TimelineNodeCard";
import { getExecutionNotes } from "../../services/executionsApi";

interface WorkflowDAGNode {
  workflow_id: string;
  execution_id: string;
  agent_node_id: string;
  reasoner_id: string;
  status: string;
  started_at: string;
  completed_at?: string;
  duration_ms?: number;
  parent_workflow_id?: string;
  parent_execution_id?: string;
  workflow_depth: number;
  children: WorkflowDAGNode[];
  agent_name?: string;
  task_name?: string;
}

interface WorkflowTimelineProps {
  nodes: WorkflowDAGNode[];
  onNodeSelect?: (node: WorkflowDAGNode) => void;
  onTagFilter?: (tags: string[]) => void;
  className?: string;
  // Lifted state props
  sortOrder?: 'asc' | 'desc';
  onSortOrderChange?: (sortOrder: 'asc' | 'desc') => void;
  selectedTags?: string[];
  expandAll?: boolean;
  onExpandAllChange?: (expandAll: boolean) => void;
  cardExpansions?: Record<string, boolean>;
  onCardExpansionChange?: (executionId: string, expanded: boolean) => void;
  noteExpansions?: Record<string, Set<number>>;
  onNoteExpansionChange?: (executionId: string, noteIndex: number, expanded: boolean) => void;
}

interface NodeWithNotes extends WorkflowDAGNode {
  notes: ExecutionNote[];
}

export function WorkflowTimeline({
  nodes,
  onNodeSelect,
  onTagFilter,
  className = "",
  // Lifted state props with defaults
  sortOrder = 'asc',
  onSortOrderChange,
  selectedTags = [],
  expandAll = false,
  onExpandAllChange,
  cardExpansions = {},
  onCardExpansionChange,
  noteExpansions = {},
  onNoteExpansionChange,
}: WorkflowTimelineProps) {
  const [nodesWithNotes, setNodesWithNotes] = useState<NodeWithNotes[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [availableTags, setAvailableTags] = useState<string[]>([]);

  // Fetch notes for all nodes
  const fetchNotesForNodes = useCallback(async () => {
    if (nodes.length === 0) {
      setNodesWithNotes([]);
      setLoading(false);
      return;
    }

    try {
      setLoading(true);
      setError(null);

      // Fetch notes for each execution
      const notesPromises = nodes.map(async (node) => {
        try {
          const notesResponse = await getExecutionNotes(node.execution_id, {});
          return {
            ...node,
            notes: notesResponse.notes || [],
          };
        } catch (err) {
          console.warn(`Failed to fetch notes for execution ${node.execution_id}:`, err);
          return {
            ...node,
            notes: [],
          };
        }
      });

      const nodesWithNotesData = await Promise.all(notesPromises);

      // Collect all unique tags
      const allTags = new Set<string>();
      nodesWithNotesData.forEach(node => {
        node.notes.forEach(note => {
          note.tags.forEach(tag => allTags.add(tag));
        });
      });

      setNodesWithNotes(nodesWithNotesData);
      setAvailableTags(Array.from(allTags).sort());
    } catch (err) {
      console.error("Failed to fetch notes for timeline:", err);
      setError(err instanceof Error ? err.message : "Failed to load timeline notes");
    } finally {
      setLoading(false);
    }
  }, [nodes]);

  useEffect(() => {
    fetchNotesForNodes();
  }, [fetchNotesForNodes]);

  // Filter and sort nodes
  const filteredAndSortedNodes = nodesWithNotes
    .filter(node => {
      if (selectedTags.length === 0) return true;

      return node.notes.some(note =>
        note.tags.some(tag =>
          selectedTags.some(selectedTag =>
            tag.toLowerCase().includes(selectedTag.toLowerCase())
          )
        )
      );
    })
    .sort((a, b) => {
      const dateA = new Date(a.started_at).getTime();
      const dateB = new Date(b.started_at).getTime();
      return sortOrder === 'asc' ? dateA - dateB : dateB - dateA;
    });

  const handleTagsChange = (tags: string[]) => {
    onTagFilter?.(tags);
  };

  const handleTagClick = (tag: string) => {
    if (!selectedTags.includes(tag)) {
      const newTags = [...selectedTags, tag];
      handleTagsChange(newTags);
    }
  };

  const handleSortToggle = () => {
    const newSortOrder = sortOrder === 'asc' ? 'desc' : 'asc';
    onSortOrderChange?.(newSortOrder);
  };

  const handleExpandAllToggle = () => {
    onExpandAllChange?.(!expandAll);
  };

  const handleNodeClick = (node: WorkflowDAGNode) => {
    onNodeSelect?.(node);
  };

  // Get total notes count
  const totalNotesCount = nodesWithNotes.reduce((sum, node) => sum + node.notes.length, 0);

  // Empty state
  if (!loading && nodes.length === 0) {
    return (
      <Card className={className}>
        <CardContent className="py-12">
          <div className="text-center">
            <div className="rounded-full bg-muted p-4 mb-4 mx-auto w-fit">
              <Events size={32} className="text-muted-foreground" />
            </div>
            <h3 className="text-base font-semibold mb-2">No workflow nodes</h3>
            <p className="text-sm text-muted-foreground">
              Timeline will appear here when workflow nodes are available.
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Error state
  if (error) {
    return (
      <Card className={className}>
        <CardContent className="py-12">
          <div className="text-center">
            <div className="rounded-full bg-status-error/10 p-4 mb-4 mx-auto w-fit">
              <Events size={32} className="text-status-error" />
            </div>
            <h3 className="text-base font-semibold mb-2">Failed to load timeline</h3>
            <p className="text-sm text-muted-foreground mb-4">{error}</p>
            <Button variant="outline" onClick={fetchNotesForNodes}>
              Try Again
            </Button>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={className}>
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2 text-base font-semibold">
            <Events size={20} />
            Workflow Timeline
            {totalNotesCount > 0 && (
              <span className="text-tertiary-foundation font-normal">
                ({totalNotesCount} notes)
              </span>
            )}
          </CardTitle>

          <div className="flex items-center gap-2">
            {/* Sort toggle */}
            <Button
              variant="ghost"
              size="sm"
              onClick={handleSortToggle}
              className="h-8 px-2"
              title={`Sort ${sortOrder === 'asc' ? 'newest first' : 'oldest first'}`}
            >
              {sortOrder === 'asc' ? (
                <ArrowDown size={16} />
              ) : (
                <ArrowUp size={16} />
              )}
            </Button>

            {/* Expand all toggle */}
            {totalNotesCount > 0 && (
              <Button
                variant="ghost"
                size="sm"
                onClick={handleExpandAllToggle}
                className="h-8 px-2"
                title={expandAll ? "Collapse all notes" : "Expand all notes"}
              >
                {expandAll ? (
                  <CollapseAll size={16} />
                ) : (
                  <ExpandAll size={16} />
                )}
              </Button>
            )}
          </div>
        </div>

        {/* Tag filter - horizontal list */}
        {availableTags.length > 0 && (
          <div className="mt-4">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-sm font-medium">Filter by tags:</span>
              {selectedTags.length > 0 && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => handleTagsChange([])}
                  className="h-6 px-2 text-xs text-muted-foreground hover:text-muted-foreground"
                >
                  Clear all
                </Button>
              )}
            </div>
            <div className="flex flex-wrap gap-1.5 max-h-20 overflow-y-auto">
              {availableTags.map(tag => (
                <TagBadge
                  key={tag}
                  tag={tag}
                  size="sm"
                  onClick={() => {
                    if (selectedTags.includes(tag)) {
                      handleTagsChange(selectedTags.filter(t => t !== tag));
                    } else {
                      handleTagsChange([...selectedTags, tag]);
                    }
                  }}
                  className={`cursor-pointer transition-all ${
                    selectedTags.includes(tag)
                      ? 'bg-muted border-text-secondary'
                      : 'hover:bg-accent'
                  }`}
                />
              ))}
            </div>
          </div>
        )}
      </CardHeader>

      <CardContent className="pt-0">
        {loading ? (
          <div className="space-y-2">
            {Array.from({ length: 3 }).map((_, index) => (
              <TimelineNodeCardSkeleton key={index} />
            ))}
          </div>
        ) : filteredAndSortedNodes.length === 0 ? (
          <div className="text-center py-8">
            <div className="rounded-full bg-muted p-4 mb-4 mx-auto w-fit">
              <Filter size={24} className="text-muted-foreground" />
            </div>
            <p className="text-sm text-muted-foreground mb-2">
              No nodes match the selected filters.
            </p>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => handleTagsChange([])}
            >
              Clear filters
            </Button>
          </div>
        ) : (
          <div className="space-y-2 max-h-[800px] overflow-y-auto scrollbar-thin">
            {filteredAndSortedNodes.map((node) => (
              <TimelineNodeCard
                key={node.execution_id}
                node={node}
                notes={node.notes}
                onClick={() => handleNodeClick(node)}
                onTagClick={handleTagClick}
                forceExpanded={expandAll}
                // Lifted state props
                isExpanded={cardExpansions[node.execution_id]}
                onExpansionChange={(expanded) => onCardExpansionChange?.(node.execution_id, expanded)}
                expandedNotes={noteExpansions[node.execution_id]}
                onNoteExpansionChange={(noteIndex, expanded) => onNoteExpansionChange?.(node.execution_id, noteIndex, expanded)}
              />
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
