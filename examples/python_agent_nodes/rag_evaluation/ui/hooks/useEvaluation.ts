'use client'

import { useState, useCallback, useRef } from 'react'
import { EvaluationInput, EvaluationResult, WorkflowNote } from '@/lib/types'
import { startAsyncEvaluation, pollExecutionResult } from '@/lib/api'

export type EvaluationStatus = 'idle' | 'evaluating' | 'success' | 'error'

export interface UseEvaluationReturn {
  status: EvaluationStatus
  result: EvaluationResult | null
  error: string | null
  notes: WorkflowNote[]
  currentStep: string | null
  evaluate: (input: EvaluationInput) => Promise<void>
  reset: () => void
}

// Map tags to human-readable step names
function getStepFromTags(tags: string[]): string | null {
  const tagToStep: Record<string, string> = {
    'orchestration': 'Orchestrating evaluation',
    'extraction': 'Extracting claims',
    'faithfulness': 'Checking faithfulness',
    'prosecution': 'Prosecutor reviewing claims',
    'defense': 'Defender responding',
    'judge': 'Judge deliberating',
    'relevance': 'Analyzing relevance',
    'jury': 'Jury voting',
    'hallucination': 'Detecting hallucinations',
    'ml': 'ML verification',
    'llm': 'LLM verification',
    'constitutional': 'Constitutional compliance',
    'complete': 'Completing evaluation',
  }

  for (const tag of tags) {
    if (tagToStep[tag]) {
      return tagToStep[tag]
    }
  }
  return null
}

export function useEvaluation(): UseEvaluationReturn {
  const [status, setStatus] = useState<EvaluationStatus>('idle')
  const [result, setResult] = useState<EvaluationResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [notes, setNotes] = useState<WorkflowNote[]>([])
  const [currentStep, setCurrentStep] = useState<string | null>(null)

  const executionIdRef = useRef<string | null>(null)

  const evaluate = useCallback(async (input: EvaluationInput) => {
    // Reset state
    setStatus('evaluating')
    setError(null)
    setResult(null)
    setNotes([])
    setCurrentStep('Starting evaluation')

    try {
      // Start async execution to get execution_id
      const startResult = await startAsyncEvaluation(input)

      if ('error' in startResult) {
        setError(startResult.error)
        setStatus('error')
        return
      }

      const { executionId } = startResult
      executionIdRef.current = executionId

      // Poll for result with notes (SSE endpoint doesn't exist, use polling)
      const pollResult = await pollExecutionResult(executionId, (note) => {
        setNotes(prev => {
          // Avoid duplicates
          const exists = prev.some(n =>
            n.message === note.message && n.timestamp === note.timestamp
          )
          if (exists) return prev
          return [...prev, note]
        })

        const step = getStepFromTags(note.tags)
        if (step) {
          setCurrentStep(step)
        }
      })

      if (pollResult.success && pollResult.result) {
        setResult(pollResult.result)
        setStatus('success')
      } else {
        setError(pollResult.error || 'Evaluation failed')
        setStatus('error')
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
      setStatus('error')
    } finally {
      setCurrentStep(null)
    }
  }, [])

  const reset = useCallback(() => {
    setStatus('idle')
    setResult(null)
    setError(null)
    setNotes([])
    setCurrentStep(null)
    executionIdRef.current = null
  }, [])

  return {
    status,
    result,
    error,
    notes,
    currentStep,
    evaluate,
    reset,
  }
}
