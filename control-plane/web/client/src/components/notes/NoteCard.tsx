import { Time, Copy, Chat } from "@/components/ui/icon-bridge";
import { useState } from "react";
import ReactMarkdown from "react-markdown";
import type { ExecutionNote } from "../../types/notes";
import { Button } from "../ui/button";
import { Card, CardContent } from "../ui/card";
import { Skeleton } from "../ui/skeleton";
import { TagBadge } from "./TagBadge";

interface NoteCardProps {
  note: ExecutionNote;
  onTagClick?: (tag: string) => void;
  className?: string;
}

function formatTimestamp(timestamp: string): string {
  const date = new Date(timestamp);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMinutes = Math.floor(diffMs / (1000 * 60));
  const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

  if (diffMinutes < 1) {
    return "Just now";
  } else if (diffMinutes < 60) {
    return `${diffMinutes}m ago`;
  } else if (diffHours < 24) {
    return `${diffHours}h ago`;
  } else if (diffDays < 7) {
    return `${diffDays}d ago`;
  } else {
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: date.getFullYear() !== now.getFullYear() ? 'numeric' : undefined,
    });
  }
}

function formatFullTimestamp(timestamp: string): string {
  const date = new Date(timestamp);
  return date.toLocaleString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
}

export function NoteCard({ note, onTagClick, className = "" }: NoteCardProps) {
  const [copied, setCopied] = useState(false);

  const handleCopyNote = async () => {
    try {
      const noteText = `${note.message}\n\nTags: ${note.tags.join(', ')}\nTime: ${formatFullTimestamp(note.timestamp)}`;
      await navigator.clipboard.writeText(noteText);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("Failed to copy note:", err);
    }
  };

  return (
    <Card className={`group bg-card border border-border rounded-lg shadow-sm ${className}`}>
      <CardContent className="p-4">
        {/* Header with timestamp and actions */}
        <div className="flex items-start justify-between gap-3 mb-3">
          <div className="flex items-center gap-2 text-muted-foreground">
            <Time size={16} />
            <span title={formatFullTimestamp(note.timestamp)}>
              {formatTimestamp(note.timestamp)}
            </span>
          </div>

          <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity duration-fast">
            <Button
              variant="ghost"
              size="sm"
              onClick={handleCopyNote}
              className="h-8 w-8 p-0"
              title="Copy note"
            >
              {copied ? (
                <Chat size={16} className="text-status-success" />
              ) : (
                <Copy size={16} />
              )}
            </Button>
          </div>
        </div>

        {/* Note message */}
        <div className="mb-3">
          <div className="text-sm text-foreground leading-relaxed prose prose-sm max-w-none prose-headings:text-sm prose-headings:font-semibold prose-headings:text-foreground prose-headings:mt-3 prose-headings:mb-2 prose-h1:text-base prose-h1:font-semibold prose-h2:text-sm prose-h2:font-semibold prose-h3:text-sm prose-h3:font-medium prose-p:text-sm prose-p:text-foreground prose-p:leading-relaxed prose-p:my-2 prose-ul:text-sm prose-ul:text-foreground prose-ul:my-2 prose-ol:text-sm prose-ol:text-foreground prose-ol:my-2 prose-li:text-sm prose-li:text-foreground prose-code:text-sm prose-code:bg-muted prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded prose-pre:text-sm prose-pre:bg-muted prose-pre:p-3 prose-pre:rounded prose-pre:my-3 prose-blockquote:text-sm prose-blockquote:text-muted-foreground prose-blockquote:border-l-border-secondary prose-blockquote:my-3 prose-strong:text-foreground prose-strong:font-medium prose-em:text-foreground">
            <ReactMarkdown>
              {note.message}
            </ReactMarkdown>
          </div>
        </div>

        {/* Tags */}
        {note.tags && note.tags.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {note.tags.map((tag, index) => (
              <TagBadge
                key={`${tag}-${index}`}
                tag={tag}
                size="sm"
                onClick={onTagClick}
              />
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// Skeleton component for loading state
export function NoteCardSkeleton({ className = "" }: { className?: string }) {
  return (
    <Card className={`${className}`}>
      <CardContent className="p-4">
        {/* Header skeleton */}
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Skeleton className="h-4 w-4 rounded" />
            <Skeleton className="h-4 w-16" />
          </div>
          <Skeleton className="h-8 w-8 rounded" />
        </div>

        {/* Message skeleton */}
        <div className="space-y-2 mb-3">
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-3/4" />
        </div>

        {/* Tags skeleton */}
        <div className="flex gap-1.5">
          <Skeleton className="h-6 w-12 rounded-md" />
          <Skeleton className="h-6 w-16 rounded-md" />
        </div>
      </CardContent>
    </Card>
  );
}
