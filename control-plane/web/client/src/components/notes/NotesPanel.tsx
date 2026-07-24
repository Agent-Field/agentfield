import {
  ArrowDown,
  ArrowUp,
  Document,
  Renew,
  Chat
} from "@/components/ui/icon-bridge";
import { useCallback, useEffect, useState } from "react";
import {
  getExecutionNotes,
  getExecutionNoteTags
} from "../../services/executionsApi";
import type { ExecutionNote, NotesFilters } from "../../types/notes";
import { Button } from "../ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import { NoteCard, NoteCardSkeleton } from "./NoteCard";
import { TagFilter } from "./TagFilter";

interface NotesPanelProps {
  executionId: string;
  className?: string;
}

interface NotesState {
  notes: ExecutionNote[];
  availableTags: string[];
  loading: boolean;
  error: string | null;
  lastUpdated: Date | null;
}

export function NotesPanel({ executionId, className = "" }: NotesPanelProps) {
  const [state, setState] = useState<NotesState>({
    notes: [],
    availableTags: [],
    loading: true,
    error: null,
    lastUpdated: null,
  });

  const [filters, setFilters] = useState<NotesFilters>({
    tags: [],
  });

  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
  const [refreshing, setRefreshing] = useState(false);

  // Fetch notes and tags
  const fetchNotes = useCallback(async (showRefreshing = false) => {
    try {
      if (showRefreshing) {
        setRefreshing(true);
      } else {
        setState(prev => ({ ...prev, loading: true, error: null }));
      }

      const [notesResponse, availableTags] = await Promise.all([
        getExecutionNotes(executionId, filters),
        getExecutionNoteTags(executionId),
      ]);

      setState(prev => ({
        ...prev,
        notes: notesResponse.notes || [],
        availableTags,
        loading: false,
        error: null,
        lastUpdated: new Date(),
      }));
    } catch (err) {
      console.error("Failed to fetch notes:", err);
      setState(prev => ({
        ...prev,
        loading: false,
        error: err instanceof Error ? err.message : "Failed to load notes",
      }));
    } finally {
      setRefreshing(false);
    }
  }, [executionId, filters]);

  // Initial load
  useEffect(() => {
    fetchNotes();
  }, [fetchNotes]);

  // Real-time updates with polling
  useEffect(() => {
    const interval = setInterval(() => {
      fetchNotes(true);
    }, 5000); // Poll every 5 seconds

    return () => clearInterval(interval);
  }, [fetchNotes]);

  // Filter and sort notes
  const filteredAndSortedNotes = state.notes
    .filter(note => {
      if (filters.tags && filters.tags.length > 0) {
        return filters.tags.some(filterTag =>
          note.tags.some(noteTag =>
            noteTag.toLowerCase().includes(filterTag.toLowerCase())
          )
        );
      }
      return true;
    })
    .sort((a, b) => {
      const dateA = new Date(a.timestamp).getTime();
      const dateB = new Date(b.timestamp).getTime();
      return sortOrder === 'desc' ? dateB - dateA : dateA - dateB;
    });

  const handleTagsChange = (tags: string[]) => {
    setFilters(prev => ({ ...prev, tags }));
  };

  const handleTagClick = (tag: string) => {
    if (!filters.tags?.includes(tag)) {
      handleTagsChange([...(filters.tags || []), tag]);
    }
  };

  const handleSortToggle = () => {
    setSortOrder(prev => prev === 'desc' ? 'asc' : 'desc');
  };

  const handleRefresh = () => {
    fetchNotes(true);
  };

  // Empty state component
  const EmptyState = () => (
    <div className="flex flex-col items-center justify-center py-12 text-center">
      <div className="rounded-full bg-muted p-4 mb-4">
        <Document size={32} className="text-muted-foreground" />
      </div>
      <h3 className="text-base font-semibold mb-2">No notes yet</h3>
      <p className="text-muted-foreground mb-4 max-w-sm">
        Notes will appear here as they are added during execution.
        Use app.note() in your code to add notes.
      </p>
      <Button variant="outline" onClick={handleRefresh} disabled={refreshing}>
        {refreshing ? (
          <Renew size={16} className="animate-spin mr-2" />
        ) : (
          <Renew size={16} className="mr-2" />
        )}
        Refresh
      </Button>
    </div>
  );

  // Error state component
  const ErrorState = () => (
    <div className="flex flex-col items-center justify-center py-12 text-center">
      <div className="rounded-full bg-status-error/10 p-4 mb-4">
        <Document size={32} className="text-status-error" />
      </div>
      <h3 className="text-base font-semibold mb-2">Failed to load notes</h3>
      <p className="text-muted-foreground mb-4 max-w-sm">
        {state.error}
      </p>
      <Button variant="outline" onClick={() => fetchNotes()}>
        <Renew size={16} className="mr-2" />
        Try Again
      </Button>
    </div>
  );

  // Loading state component
  const LoadingState = () => (
    <div className="space-y-4">
      {Array.from({ length: 3 }).map((_, index) => (
        <NoteCardSkeleton key={index} />
      ))}
    </div>
  );

  return (
    <Card className={`${className}`}>
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2 text-base font-semibold">
            <Chat size={20} />
            Execution Notes
            {state.notes.length > 0 && (
              <span className="text-muted-foreground font-normal">
                ({filteredAndSortedNotes.length} of {state.notes.length})
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
              title={`Sort ${sortOrder === 'desc' ? 'oldest first' : 'newest first'}`}
            >
              {sortOrder === 'desc' ? (
                <ArrowDown size={16} />
              ) : (
                <ArrowUp size={16} />
              )}
            </Button>

            {/* Refresh button */}
            <Button
              variant="ghost"
              size="sm"
              onClick={handleRefresh}
              disabled={refreshing}
              className="h-8 px-2"
              title="Refresh notes"
            >
              <Renew size={16} className={refreshing ? 'animate-spin' : ''} />
            </Button>
          </div>
        </div>

        {/* Tag filter */}
        {state.availableTags.length > 0 && (
          <TagFilter
            availableTags={state.availableTags}
            selectedTags={filters.tags || []}
            onTagsChange={handleTagsChange}
            className="mt-4"
          />
        )}

        {/* Last updated indicator */}
        {state.lastUpdated && (
          <div className="text-muted-foreground mt-2">
            Last updated: {state.lastUpdated.toLocaleTimeString()}
          </div>
        )}
      </CardHeader>

      <CardContent className="pt-0">
        {state.loading ? (
          <LoadingState />
        ) : state.error ? (
          <ErrorState />
        ) : filteredAndSortedNotes.length === 0 ? (
          state.notes.length === 0 ? (
            <EmptyState />
          ) : (
            <div className="text-center py-8">
              <p className="text-muted-foreground">
                No notes match the selected filters.
              </p>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => handleTagsChange([])}
                className="mt-2"
              >
                Clear filters
              </Button>
            </div>
          )
        ) : (
          <div className="space-y-4 max-h-96 overflow-y-auto scrollbar-thin">
            {filteredAndSortedNotes.map((note, index) => (
              <NoteCard
                key={`${note.timestamp}-${index}`}
                note={note}
                onTagClick={handleTagClick}
              />
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
