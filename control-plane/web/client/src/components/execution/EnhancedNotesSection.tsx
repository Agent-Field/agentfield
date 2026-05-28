import { useState, useMemo } from "react";
import {
  FileText,
  RefreshCw,
  ChevronDown,
  ChevronUp,
  Clock,
  Tag,
  ArrowUpDown,
  ArrowUp,
  ArrowDown,
} from "@/components/ui/icon-bridge";
import type { WorkflowExecution, ExecutionNote } from "../../types/executions";

import { Button } from "../ui/button";

interface EnhancedNotesSectionProps {
  execution: WorkflowExecution;
  onRefresh?: () => void;
}

type SortOrder = "newest" | "oldest" | "chronological";

interface ExpandableNoteProps {
  note: ExecutionNote;
  index: number;
}

function formatTimeForEvent(timestamp: string): {
  date: string;
  time: string;
  relative: string;
} {
  try {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / (1000 * 60));
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

    let relative: string;
    if (diffMins < 1) {
      relative = "just now";
    } else if (diffMins < 60) {
      relative = `${diffMins}m ago`;
    } else if (diffHours < 24) {
      relative = `${diffHours}h ago`;
    } else if (diffDays < 7) {
      relative = `${diffDays}d ago`;
    } else {
      relative = date.toLocaleDateString();
    }

    return {
      date: date.toLocaleDateString(),
      time: date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      relative,
    };
  } catch {
    return { date: timestamp, time: "", relative: timestamp };
  }
}

function ExpandableNote({ note }: ExpandableNoteProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const { time, relative } = formatTimeForEvent(note.timestamp);

  const isLongNote = note.message.length > 150;
  const displayMessage =
    isLongNote && !isExpanded
      ? `${note.message.slice(0, 150)}...`
      : note.message;

  return (
    <div className="group relative pl-6 pb-4 last:pb-0">
      {/* Timeline dot */}
      <div className="absolute left-0 top-1 w-2 h-2 bg-blue-500 rounded-full ring-2 ring-background border border-border"></div>

      {/* Timeline line */}
      <div className="absolute left-0.5 top-3 w-0.5 h-full bg-border group-last:hidden"></div>

      {/* Event content */}
      <div className="space-y-2">
        {/* Header with time and tags */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 text-sm">
              <Clock className="w-3 h-3 text-muted-foreground" />
              <span className="font-medium text-foreground">{time}</span>
              <span className="text-muted-foreground">·</span>
              <span className="text-sm text-muted-foreground">{relative}</span>
            </div>
          </div>

          {note.tags && note.tags.length > 0 && (
            <div className="flex items-center gap-1">
              {note.tags.map((tag, tagIndex) => (
                <span
                  key={tagIndex}
                  className="inline-flex items-center gap-1 text-xs bg-muted text-muted-foreground px-2 py-0.5 rounded-full"
                >
                  <Tag className="w-2 h-2" />
                  {tag}
                </span>
              ))}
            </div>
          )}
        </div>

        {/* Message content */}
        <div className="text-sm text-foreground leading-relaxed">
          <div className="whitespace-pre-wrap break-words">
            {displayMessage}
          </div>

          {isLongNote && (
            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className="inline-flex items-center gap-1 mt-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
            >
              {isExpanded ? (
                <>
                  <ChevronUp className="w-3 h-3" />
                  Show less
                </>
              ) : (
                <>
                  <ChevronDown className="w-3 h-3" />
                  Show more
                </>
              )}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

export function EnhancedNotesSection({
  execution,
  onRefresh,
}: EnhancedNotesSectionProps) {
  const [sortOrder, setSortOrder] = useState<SortOrder>("newest");
  const [isRefreshing, setIsRefreshing] = useState(false);

  // Check if we have any notes content
  const hasNotes = execution.notes && execution.notes.length > 0;

  // Don't render the section at all if no content
  if (!hasNotes) {
    return null;
  }

  const sortedNotes = useMemo(() => {
    if (!execution.notes) return [];

    const notesCopy = [...execution.notes];

    switch (sortOrder) {
      case "newest":
        return notesCopy.sort(
          (a, b) =>
            new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
        );
      case "oldest":
        return notesCopy.sort(
          (a, b) =>
            new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
        );
      case "chronological":
        return notesCopy.sort(
          (a, b) =>
            new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
        );
      default:
        return notesCopy;
    }
  }, [execution.notes, sortOrder]);

  const handleRefresh = async (e: React.MouseEvent) => {
    e.stopPropagation(); // Prevent section collapse
    if (onRefresh) {
      setIsRefreshing(true);
      try {
        await onRefresh();
      } finally {
        setTimeout(() => setIsRefreshing(false), 500); // Minimum refresh animation time
      }
    }
  };

  const getSortIcon = () => {
    switch (sortOrder) {
      case "newest":
        return <ArrowDown className="w-3 h-3" />;
      case "oldest":
        return <ArrowUp className="w-3 h-3" />;
      case "chronological":
        return <ArrowUpDown className="w-3 h-3" />;
      default:
        return <ArrowUpDown className="w-3 h-3" />;
    }
  };

  const getSortLabel = () => {
    switch (sortOrder) {
      case "newest":
        return "Newest first";
      case "oldest":
        return "Oldest first";
      case "chronological":
        return "Chronological";
      default:
        return "Sort";
    }
  };

  const cycleSortOrder = (e: React.MouseEvent) => {
    e.stopPropagation(); // Prevent section collapse
    const orders: SortOrder[] = ["newest", "oldest", "chronological"];
    const currentIndex = orders.indexOf(sortOrder);
    const nextIndex = (currentIndex + 1) % orders.length;
    setSortOrder(orders[nextIndex]);
  };

  const badge = (
    <div className="flex items-center gap-2">
      <span className="text-sm text-muted-foreground bg-blue-500/10 text-blue-600 px-2 py-0.5 rounded">
        {execution.notes?.length || 0}{" "}
        {execution.notes?.length === 1 ? "Event" : "Events"}
      </span>

      <div className="flex items-center gap-1">
        <Button
          variant="ghost"
          size="sm"
          onClick={cycleSortOrder}
          className="h-6 text-xs px-2 text-muted-foreground hover:text-foreground"
          title={getSortLabel()}
        >
          {getSortIcon()}
        </Button>

        {onRefresh && (
          <Button
            variant="ghost"
            size="sm"
            onClick={handleRefresh}
            disabled={isRefreshing}
            className="h-6 w-6 p-0 text-muted-foreground hover:text-foreground"
            title="Refresh notes"
          >
            <RefreshCw
              className={`w-3 h-3 ${isRefreshing ? "animate-spin" : ""}`}
            />
          </Button>
        )}
      </div>
    </div>
  );

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <FileText className="w-5 h-5" />
          <h3 className="text-base font-semibold text-foreground">
            Execution Events
          </h3>
          {badge}
        </div>
      </div>

      {/* Timeline container */}
      <div className="relative">
        {sortedNotes.map((note, index) => (
          <ExpandableNote
            key={`${note.timestamp}-${index}`}
            note={note}
            index={index}
          />
        ))}
      </div>

      {/* Summary footer */}
      {sortedNotes.length > 3 && (
        <div className="mt-4 pt-3 border-t border-border">
          <div className="text-sm text-muted-foreground text-center">
            {sortedNotes.length} events • Sorted by{" "}
            {getSortLabel().toLowerCase()}
          </div>
        </div>
      )}
    </div>
  );
}
