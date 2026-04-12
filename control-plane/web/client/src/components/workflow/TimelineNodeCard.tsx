import {
  ChevronDown,
  ChevronRight,
} from "@/components/ui/icon-bridge";
import { useState, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import type { ExecutionNote } from "../../types/notes";
import { Button } from "../ui/button";
import { Skeleton } from "../ui/skeleton";
import { getTagColor } from "../notes/TagBadge";
import { normalizeExecutionStatus } from "../../utils/status";

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

interface TimelineNodeCardProps {
  node: WorkflowDAGNode;
  notes: ExecutionNote[];
  onClick?: () => void;
  onTagClick?: (tag: string) => void;
  forceExpanded?: boolean;
  className?: string;
  // Lifted state props
  isExpanded?: boolean;
  onExpansionChange?: (expanded: boolean) => void;
  expandedNotes?: Set<number>;
  onNoteExpansionChange?: (noteIndex: number, expanded: boolean) => void;
}

function getStatusDot(status: string) {
  const normalized = normalizeExecutionStatus(status);
  switch (normalized) {
    case "succeeded":
      return <div className="w-2 h-2 rounded-full bg-status-success flex-shrink-0" />;
    case "running":
      return <div className="w-2 h-2 rounded-full bg-status-info flex-shrink-0 animate-pulse" />;
    case "failed":
      return <div className="w-2 h-2 rounded-full bg-status-error flex-shrink-0" />;
    case "cancelled":
      return <div className="w-2 h-2 rounded-full bg-text-quaternary flex-shrink-0" />;
    case "timeout":
      return <div className="w-2 h-2 rounded-full bg-purple-500 flex-shrink-0" />;
    case "queued":
    case "pending":
      return <div className="w-2 h-2 rounded-full bg-status-info flex-shrink-0" />;
    default:
      return <div className="w-2 h-2 rounded-full bg-text-tertiary flex-shrink-0" />;
  }
}

function humanizeText(text: string): string {
  return text
    .replace(/_/g, " ")
    .replace(/-/g, " ")
    .replace(/\b\w/g, (l) => l.toUpperCase())
    .replace(/\s+/g, " ")
    .trim();
}

function formatTimestamp(timestamp: string): string {
  const date = new Date(timestamp);
  return date.toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
  });
}

function getCompactNotesPreview(notes: ExecutionNote[]): string {
  if (notes.length === 0) return "";

  const firstNote = notes[0];
  const preview = firstNote.message.replace(/\n/g, " ").trim();
  // Much shorter preview for compact display
  const truncatedPreview = preview.length > 40 ? preview.substring(0, 40) + "..." : preview;

  // If there are multiple notes, indicate this
  if (notes.length > 1) {
    return `${truncatedPreview} (+${notes.length - 1})`;
  }

  return truncatedPreview;
}

function getCompactTags(notes: ExecutionNote[]): string[] {
  const allTags = notes.flatMap((note) => note.tags);
  const uniqueTags = Array.from(new Set(allTags));
  // Show only first 2 tags for ultra-compact display
  return uniqueTags.slice(0, 2);
}

export function TimelineNodeCard({
  node,
  notes,
  onClick,
  onTagClick,
  forceExpanded = false,
  className = "",
  // Lifted state props with defaults
  isExpanded,
  onExpansionChange,
  expandedNotes = new Set(),
  onNoteExpansionChange,
}: TimelineNodeCardProps) {
  // Use lifted state if provided, otherwise fall back to local state for backward compatibility
  const [localNotesExpanded, setLocalNotesExpanded] = useState(false);
  const [localExpandedNotes, setLocalExpandedNotes] = useState<Set<number>>(new Set());

  // Determine which state to use
  const notesExpanded = isExpanded !== undefined ? isExpanded : localNotesExpanded;
  const currentExpandedNotes = onNoteExpansionChange ? expandedNotes : localExpandedNotes;

  // Update expansion state when forceExpanded changes
  useEffect(() => {
    if (isExpanded === undefined) {
      setLocalNotesExpanded(forceExpanded);
    }
  }, [forceExpanded, isExpanded]);

  const agentName = humanizeText(node.agent_name || node.agent_node_id);
  const reasonerName = humanizeText(node.task_name || node.reasoner_id);
  const hasNotes = notes.length > 0;
  const compactPreview = getCompactNotesPreview(notes);
  const compactTags = getCompactTags(notes);

  const handleCardClick = () => {
    // If has notes, toggle expansion instead of selection
    if (hasNotes) {
      if (onExpansionChange) {
        onExpansionChange(!notesExpanded);
      } else {
        setLocalNotesExpanded(!notesExpanded);
      }
    }
    // Still call onClick for any parent component logic
    onClick?.();
  };

  const handleNotesToggle = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (onExpansionChange) {
      onExpansionChange(!notesExpanded);
    } else {
      setLocalNotesExpanded(!notesExpanded);
    }
  };

  return (
    <div
      className={`
        group transition-all duration-200 ease-out
        border border-border rounded-lg
        ${hasNotes ? 'bg-card hover:hover:bg-accent cursor-pointer' : 'bg-muted hover:bg-muted'}
        ${className}
      `}
      onClick={handleCardClick}
    >
      {/* Ultra-Compact Single Bar - Same for all cards */}
      <div className="px-3 py-2 flex items-center justify-between gap-3 text-sm">
        {/* Left: Status + Names */}
        <div className="flex items-center gap-3 min-w-0 flex-1">
          {/* Status dot */}
          <div className="flex-shrink-0">
            {getStatusDot(node.status)}
          </div>

          {/* Reasoner Name */}
          <span className={`font-medium truncate ${hasNotes ? 'text-foreground' : 'text-muted-foreground'}`}>
            {reasonerName}
          </span>

          <span className="text-muted-foreground">•</span>

          {/* Agent Name */}
          <span className="text-muted-foreground truncate text-xs">
            {agentName}
          </span>
        </div>

        {/* Right: Time + Expand Button */}
        <div className="flex items-center gap-2 flex-shrink-0">
          <span className="text-xs text-muted-foreground">
            {formatTimestamp(node.started_at)}
          </span>

          {hasNotes && (
            <Button
              variant="ghost"
              size="sm"
              onClick={handleNotesToggle}
              className="h-5 w-5 p-0 opacity-0 group-hover:opacity-100 transition-opacity"
            >
              {notesExpanded ? (
                <ChevronDown size={12} />
              ) : (
                <ChevronRight size={12} />
              )}
            </Button>
          )}
        </div>
      </div>

      {/* Subtle Preview Row for Cards with Notes (Only when collapsed) */}
      {hasNotes && !notesExpanded && (
        <div className="px-3 pb-2">
          <div className="flex items-center gap-2 text-xs">
            {/* Offset to align with content above */}
            <div className="w-2 flex-shrink-0"></div>
            <div className="w-3 flex-shrink-0"></div>

            {/* Compact preview text */}
            <span className="text-muted-foreground truncate flex-1 leading-relaxed">
              {compactPreview}
            </span>

            {/* Compact tags */}
            {compactTags.length > 0 && (
              <div className="flex items-center gap-1 flex-shrink-0">
                {compactTags.map((tag, index) => {
                  const tagColor = getTagColor(tag);
                  const textColorMatch = tagColor.match(/text-([^\s]+)/);
                  const textColorClass = textColorMatch ? textColorMatch[0] : 'text-muted-foreground';

                  return (
                    <button
                      key={`${tag}-${index}`}
                      onClick={(e) => {
                        e.stopPropagation();
                        onTagClick?.(tag);
                      }}
                      className={`text-xs ${textColorClass} opacity-60 hover:opacity-100 transition-opacity cursor-pointer`}
                      title={`Filter by ${tag}`}
                    >
                      #{tag}
                    </button>
                  );
                })}
                {notes.flatMap(n => n.tags).length > 2 && (
                  <span className="text-xs text-muted-foreground opacity-50">
                    +{notes.flatMap(n => n.tags).length - 2}
                  </span>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Expandable Notes Section */}
      {hasNotes && notesExpanded && (
        <div className="px-3 pb-3 border-t border-border bg-muted/50">
          <div className="pt-3 space-y-2.5">
            {notes.map((note, index) => (
              <div key={`${note.timestamp}-${index}`} className="space-y-1.5">
                <div className="text-xs text-foreground leading-relaxed prose prose-xs max-w-none prose-headings:text-xs prose-headings:font-semibold prose-headings:text-foreground prose-headings:mt-2 prose-headings:mb-1 prose-h1:text-sm prose-h1:font-semibold prose-h2:text-xs prose-h2:font-semibold prose-h3:text-xs prose-h3:font-medium prose-p:text-xs prose-p:text-foreground prose-p:leading-relaxed prose-p:my-1 prose-ul:text-xs prose-ul:text-foreground prose-ul:my-1 prose-ol:text-xs prose-ol:text-foreground prose-ol:my-1 prose-li:text-xs prose-li:text-foreground prose-code:text-xs prose-code:bg-muted prose-code:px-1 prose-code:py-0.5 prose-code:rounded prose-pre:text-xs prose-pre:bg-muted prose-pre:p-2 prose-pre:rounded prose-pre:my-2 prose-blockquote:text-xs prose-blockquote:text-muted-foreground prose-blockquote:border-l-border-secondary prose-blockquote:my-2 prose-strong:text-foreground prose-strong:font-medium prose-em:text-foreground">
                  {note.message.length > 200 && !currentExpandedNotes.has(index) ? (
                    <>
                      <ReactMarkdown>
                        {note.message.substring(0, 200) + "..."}
                      </ReactMarkdown>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-auto p-0 ml-1 text-accent-primary text-xs hover:text-accent-primary-hover"
                        onClick={(e) => {
                          e.stopPropagation();
                          if (onNoteExpansionChange) {
                            onNoteExpansionChange(index, true);
                          } else {
                            setLocalExpandedNotes(prev => new Set([...prev, index]));
                          }
                        }}
                      >
                        Show more
                      </Button>
                    </>
                  ) : (
                    <>
                      <ReactMarkdown>
                        {note.message}
                      </ReactMarkdown>
                      {note.message.length > 200 && currentExpandedNotes.has(index) && (
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-auto p-0 ml-1 text-accent-primary text-xs hover:text-accent-primary-hover"
                          onClick={(e) => {
                            e.stopPropagation();
                            if (onNoteExpansionChange) {
                              onNoteExpansionChange(index, false);
                            } else {
                              setLocalExpandedNotes(prev => {
                                const newSet = new Set(prev);
                                newSet.delete(index);
                                return newSet;
                              });
                            }
                          }}
                        >
                          Show less
                        </Button>
                      )}
                    </>
                  )}
                </div>

                {note.tags.length > 0 && (
                  <div className="flex flex-wrap gap-1.5">
                    {note.tags.map((tag, tagIndex) => {
                      const tagColor = getTagColor(tag);
                      // Extract text color from the tag color classes
                      const textColorMatch = tagColor.match(/text-([^\s]+)/);
                      const textColorClass = textColorMatch ? textColorMatch[0] : 'text-muted-foreground';

                      return (
                        <button
                          key={`${tag}-${tagIndex}`}
                          onClick={(e) => {
                            e.stopPropagation();
                            onTagClick?.(tag);
                          }}
                          className={`text-xs ${textColorClass} hover:opacity-75 transition-opacity cursor-pointer`}
                          title={`Filter by ${tag}`}
                        >
                          #{tag}
                        </button>
                      );
                    })}
                  </div>
                )}

                <div className="text-xs text-muted-foreground">
                  {formatTimestamp(note.timestamp)}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// Skeleton component for loading state
export function TimelineNodeCardSkeleton({
  className = "",
}: {
  className?: string;
}) {
  return (
    <div className={`border border-border rounded-lg bg-card ${className}`}>
      <div className="px-3 py-2 flex items-center justify-between gap-3">
        {/* Left: Status + Names skeleton */}
        <div className="flex items-center gap-3 min-w-0 flex-1">
          {/* Status dot skeleton */}
          <Skeleton className="h-2 w-2 rounded-full flex-shrink-0" />

          {/* Reasoner name skeleton */}
          <Skeleton className="h-4 w-20" />

          {/* Separator */}
          <Skeleton className="h-1 w-1 rounded-full" />

          {/* Agent name skeleton */}
          <Skeleton className="h-3 w-16" />
        </div>

        {/* Right: Time skeleton */}
        <Skeleton className="h-3 w-12 flex-shrink-0" />
      </div>
    </div>
  );
}
