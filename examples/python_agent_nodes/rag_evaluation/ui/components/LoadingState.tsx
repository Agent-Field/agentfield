'use client'

import { useEffect, useRef } from 'react'
import { Loader2, CheckCircle, Circle, ArrowRight } from 'lucide-react'
import { Card, CardContent } from '@/components/ui/card'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Badge } from '@/components/ui/badge'
import { WorkflowNote } from '@/lib/types'

interface LoadingStateProps {
  notes: WorkflowNote[]
  currentStep: string | null
}

// Get badge variant based on tag
function getTagVariant(tag: string): 'default' | 'secondary' | 'outline' {
  const variants: Record<string, 'default' | 'secondary' | 'outline'> = {
    'orchestration': 'default',
    'faithfulness': 'secondary',
    'relevance': 'secondary',
    'hallucination': 'secondary',
    'constitutional': 'secondary',
    'complete': 'default',
  }
  return variants[tag] || 'outline'
}

// Format timestamp to relative time
function formatTime(timestamp: string): string {
  const now = new Date()
  const then = new Date(timestamp)
  const diffMs = now.getTime() - then.getTime()
  const diffSec = Math.floor(diffMs / 1000)

  if (diffSec < 1) return 'now'
  if (diffSec < 60) return `${diffSec}s ago`
  return `${Math.floor(diffSec / 60)}m ago`
}

export function LoadingState({ notes, currentStep }: LoadingStateProps) {
  const scrollRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to bottom when new notes arrive
  useEffect(() => {
    if (scrollRef.current) {
      const scrollElement = scrollRef.current.querySelector('[data-radix-scroll-area-viewport]')
      if (scrollElement) {
        scrollElement.scrollTop = scrollElement.scrollHeight
      }
    }
  }, [notes])

  return (
    <div className="max-w-lg mx-auto">
      <Card>
        <CardContent className="pt-8 pb-6">
          {/* Spinner and Title */}
          <div className="text-center mb-6">
            <Loader2 className="h-10 w-10 animate-spin text-primary mx-auto mb-4" />
            <h3 className="text-lg font-medium">Evaluating your response...</h3>
            <p className="text-sm text-muted-foreground mt-1">
              Multi-perspective analysis in progress
            </p>
          </div>

          {/* Current Step */}
          {currentStep && (
            <div className="flex items-center justify-center gap-2 mb-6 py-2 px-4 rounded-lg bg-muted/50">
              <ArrowRight className="h-4 w-4 text-primary animate-pulse" />
              <span className="text-sm font-medium truncate">{currentStep}</span>
            </div>
          )}

          {/* Workflow Log */}
          <div className="border rounded-lg">
            <div className="px-3 py-2 border-b bg-muted/30 flex items-center justify-between">
              <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                Workflow Log
              </span>
              <span className="text-xs text-muted-foreground">
                {notes.length} events
              </span>
            </div>
            <ScrollArea className="h-56" ref={scrollRef}>
              <div className="p-2 space-y-1">
                {notes.length === 0 ? (
                  <div className="text-center py-8 text-sm text-muted-foreground">
                    <Loader2 className="h-5 w-5 animate-spin mx-auto mb-2 opacity-50" />
                    Waiting for workflow events...
                  </div>
                ) : (
                  notes.map((note, index) => (
                    <div
                      key={`${note.timestamp}-${index}`}
                      className="flex items-start gap-2 py-1.5 px-2 rounded hover:bg-muted/30 transition-colors animate-in fade-in slide-in-from-bottom-1 duration-300"
                    >
                      {/* Status indicator */}
                      <div className="mt-0.5 flex-shrink-0">
                        {index === notes.length - 1 ? (
                          <Circle className="h-3.5 w-3.5 text-primary fill-primary animate-pulse" />
                        ) : (
                          <CheckCircle className="h-3.5 w-3.5 text-emerald-500" />
                        )}
                      </div>

                      {/* Message */}
                      <div className="flex-1 min-w-0">
                        <p className={`text-sm leading-snug ${index === notes.length - 1 ? 'text-foreground' : 'text-muted-foreground'}`}>
                          {note.message}
                        </p>
                        <div className="flex items-center gap-1.5 mt-1 flex-wrap">
                          {note.tags.slice(0, 3).map((tag) => (
                            <Badge
                              key={tag}
                              variant={getTagVariant(tag)}
                              className="text-xs px-1.5 py-0"
                            >
                              {tag}
                            </Badge>
                          ))}
                          <span className="text-xs text-muted-foreground">
                            {formatTime(note.timestamp)}
                          </span>
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </ScrollArea>
          </div>

          {/* Footer */}
          <p className="text-center text-xs text-muted-foreground mt-6">
            Typically takes 5-15 seconds
          </p>
        </CardContent>
      </Card>
    </div>
  )
}
