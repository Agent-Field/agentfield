import { FileText } from "@/components/ui/icon-bridge";
import type { WorkflowExecution } from "../../types/executions";
import { CollapsibleSection } from "./CollapsibleSection";

interface SmartNotesSectionProps {
  execution: WorkflowExecution;
}

export function SmartNotesSection({ execution }: SmartNotesSectionProps) {
  // Check if we have any notes content
  const hasNotes = execution.notes && execution.notes.length > 0;

  // Don't render the section at all if no content
  if (!hasNotes) {
    return null;
  }

  const formatTimestamp = (timestamp: string) => {
    try {
      return new Date(timestamp).toLocaleString();
    } catch {
      return timestamp;
    }
  };

  return (
    <CollapsibleSection
      title="Execution Notes"
      icon={FileText}
      defaultOpen={false}
      badge={
        <span className="text-sm text-muted-foreground bg-blue-500/10 text-blue-600 px-2 py-0.5 rounded">
          {execution.notes?.length || 0} {execution.notes?.length === 1 ? 'Note' : 'Notes'}
        </span>
      }
    >
      <div className="p-4 space-y-3">
        {execution.notes?.map((note, index) => (
          <div
            key={index}
            className="p-3 bg-blue-50/50 dark:bg-blue-950/20 border border-blue-200/50 dark:border-blue-800/50 rounded-lg"
          >
            <div className="flex items-start justify-between mb-2">
              <div className="flex items-center gap-2">
                {note.tags && note.tags.length > 0 && (
                  <div className="flex gap-1">
                    {note.tags.map((tag, tagIndex) => (
                      <span
                        key={tagIndex}
                        className="text-xs bg-blue-100 dark:bg-blue-900/50 text-blue-700 dark:text-blue-300 px-1.5 py-0.5 rounded"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                )}
              </div>
              <span className="text-sm text-muted-foreground">
                {formatTimestamp(note.timestamp)}
              </span>
            </div>
            <div className="text-sm text-foreground leading-relaxed whitespace-pre-wrap">
              {note.message}
            </div>
          </div>
        ))}
      </div>
    </CollapsibleSection>
  );
}
