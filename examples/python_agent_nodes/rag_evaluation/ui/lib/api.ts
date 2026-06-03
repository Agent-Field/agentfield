import { EvaluationInput, EvaluationResult, EvaluationResponse, WorkflowNote, AsyncExecutionResponse, ExecutionEventPayload } from './types'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080'
const WEBHOOK_URL = process.env.NEXT_PUBLIC_WEBHOOK_URL
const WEBHOOK_SECRET = process.env.NEXT_PUBLIC_WEBHOOK_SECRET

// Start async execution and return execution_id for SSE subscription
export async function startAsyncEvaluation(input: EvaluationInput): Promise<{ executionId: string, runId?: string, workflowId?: string, webhookRegistered?: boolean } | { error: string }> {
  try {
    const body: any = {
      input: {
        question: input.question,
        context: input.context,
        response: input.response,
        mode: input.mode || 'standard',
        domain: input.domain || 'general',
        model: input.model || 'openrouter/google/gemini-2.0-flash-001',
      },
    }

    // Register webhook when available so backend can push completion (preferred over polling)
    if (WEBHOOK_URL) {
      body.webhook = {
        url: WEBHOOK_URL,
        ...(WEBHOOK_SECRET ? { secret: WEBHOOK_SECRET } : {}),
      }
    }

    const response = await fetch(`${API_URL}/api/v1/execute/async/rag-evaluation.evaluate_rag_response`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    })

    if (!response.ok) {
      const errorText = await response.text()
      return { error: `Failed to start evaluation: ${response.status} - ${errorText}` }
    }

    const data = await response.json()
    return { executionId: data.execution_id, runId: data.run_id, workflowId: data.workflow_id, webhookRegistered: data.webhook_registered }
  } catch (error) {
    return { error: error instanceof Error ? error.message : 'Failed to start evaluation' }
  }
}

// Subscribe to UI SSE stream for execution events and notes
export function subscribeToExecutionEvents(
  executionId: string,
  runId: string | null,
  {
    onNote,
    onStatus,
    onError,
  }: {
    onNote: (note: WorkflowNote) => void
    onStatus: (status: string) => void
    onError?: (err: string) => void
  }
): () => void {
  const eventSource = new EventSource(`${API_URL}/api/ui/v1/executions/events`)

  eventSource.onmessage = (event) => {
    try {
      const payload: ExecutionEventPayload = JSON.parse(event.data)

      // Heartbeat or malformed events
      if (!payload || payload.type === 'heartbeat') return

      const matchesExecution =
        payload.execution_id === executionId ||
        (runId && payload.run_id === runId) ||
        (runId && payload.workflow_id === runId)

      if (!matchesExecution) return

      if (payload.type === 'workflow_note_added') {
        const noteData = payload.data?.note
        if (!noteData) return
        onNote({
          type: 'note',
          message: noteData.message || '',
          tags: noteData.tags || [],
          timestamp: noteData.timestamp || payload.timestamp || new Date().toISOString(),
        })
        return
      }

      if (payload.status) {
        onStatus(payload.status)
      }
      if (payload.type && (payload.type === 'execution_completed' || payload.type === 'execution_failed')) {
        onStatus(payload.type === 'execution_completed' ? 'succeeded' : 'failed')
      }
    } catch (err) {
      console.error('Error parsing SSE message:', err)
      onError?.('Unable to parse SSE payload')
    }
  }

  eventSource.onerror = (error) => {
    console.error('SSE connection error:', error)
    onError?.('SSE connection error')
  }

  return () => {
    eventSource.close()
  }
}

// Fetch execution status/result with retry for running status
export async function fetchExecutionStatus(executionId: string, maxRetries: number = 10): Promise<EvaluationResponse> {
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      const response = await fetch(`${API_URL}/api/v1/executions/${executionId}`)
      if (!response.ok) {
        const errorText = await response.text()
        return { success: false, error: `Status fetch failed: ${errorText}` }
      }

      const data = await response.json()
      if (data.status === 'succeeded' && data.result) {
        return {
          success: true,
          result: transformResult(data.result),
          executionId,
          runId: data.run_id,
          workflowId: data.workflow_id,
        }
      }

      if (data.status === 'failed') {
        return {
          success: false,
          error: data.error || 'Execution failed',
          executionId,
        }
      }

      // If still running, wait and retry
      if (data.status === 'running' || data.status === 'pending') {
        await new Promise(resolve => setTimeout(resolve, 500))
        continue
      }

      return {
        success: false,
        error: data.error || `Execution ${data.status || 'unknown'}`,
        executionId,
      }
    } catch (err) {
      return { success: false, error: err instanceof Error ? err.message : 'Unknown status error', executionId }
    }
  }

  // Exhausted retries, execution still running
  return { success: false, error: 'Execution still running after retries', executionId }
}

// Fetch notes for an execution
export async function fetchExecutionNotes(executionId: string): Promise<WorkflowNote[]> {
  try {
    const response = await fetch(`${API_URL}/api/ui/v1/executions/${executionId}/notes`)
    if (!response.ok) return []

    const data = await response.json()
    return (data.notes || []).map((note: any) => ({
      type: 'note' as const,
      message: note.message || '',
      tags: note.tags || [],
      timestamp: note.timestamp || new Date().toISOString(),
    }))
  } catch {
    return []
  }
}

// Poll for execution result with notes
export async function pollExecutionResult(
  executionId: string,
  onNote?: (note: WorkflowNote) => void
): Promise<EvaluationResponse> {
  const maxAttempts = 120 // 2 minutes
  const pollInterval = 500 // Poll faster for notes
  const seenNotes = new Set<string>()

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    try {
      // Fetch notes in parallel with status
      const [statusResponse, notes] = await Promise.all([
        fetch(`${API_URL}/api/v1/executions/${executionId}`),
        fetchExecutionNotes(executionId),
      ])

      // Emit new notes
      if (onNote && notes.length > 0) {
        notes.forEach((note) => {
          const noteKey = `${note.message}-${note.timestamp}`
          if (!seenNotes.has(noteKey)) {
            seenNotes.add(noteKey)
            onNote(note)
          }
        })
      }

      if (!statusResponse.ok) {
        await new Promise(resolve => setTimeout(resolve, pollInterval))
        continue
      }

      const data = await statusResponse.json()

      if (data.status === 'succeeded') {
        // Fetch final notes
        const finalNotes = await fetchExecutionNotes(executionId)
        if (onNote) {
          finalNotes.forEach((note) => {
            const noteKey = `${note.message}-${note.timestamp}`
            if (!seenNotes.has(noteKey)) {
              seenNotes.add(noteKey)
              onNote(note)
            }
          })
        }

        return {
          success: true,
          result: transformResult(data.result),
          executionId,
        }
      }

      if (data.status === 'failed') {
        return {
          success: false,
          error: data.error || 'Execution failed',
          executionId,
        }
      }

      await new Promise(resolve => setTimeout(resolve, pollInterval))
    } catch (error) {
      await new Promise(resolve => setTimeout(resolve, pollInterval))
    }
  }

  return {
    success: false,
    error: 'Evaluation timed out',
    executionId,
  }
}

export async function evaluateRAG(input: EvaluationInput): Promise<EvaluationResponse> {
  try {
    const response = await fetch(`${API_URL}/api/v1/execute/rag-evaluation.evaluate_rag_response`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        input: {
          question: input.question,
          context: input.context,
          response: input.response,
          mode: input.mode || 'standard',
          domain: input.domain || 'general',
          model: input.model || 'openrouter/google/gemini-2.0-flash-001',
        },
      }),
    })

    if (!response.ok) {
      const errorText = await response.text()
      throw new Error(`API error: ${response.status} - ${errorText}`)
    }

    const data = await response.json()

    // Transform API response to our format
    return {
      success: true,
      result: transformResult(data.result),
      executionId: data.execution_id,
    }
  } catch (error) {
    console.error('Evaluation error:', error)
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error occurred',
    }
  }
}

// Transform the raw API response into our typed format
function transformResult(raw: any): EvaluationResult {
  // Handle both direct result and nested result structures
  const result = raw?.result || raw

  // Calculate hallucination ML stats from available data
  const hallucinationData = result.hallucination || {}
  const totalStatements = hallucinationData.total_statements ?? 0
  const fabricationsCount = hallucinationData.fabrications?.length ?? 0
  const contradictionsCount = hallucinationData.contradictions?.length ?? 0
  const mlHandledPercent = hallucinationData.ml_handled_percent ?? 0
  // ML verified = statements that were handled by ML and NOT found to be fabrications
  const mlVerifiedCount = Math.round((mlHandledPercent / 100) * totalStatements) - fabricationsCount - contradictionsCount
  // LLM escalated = statements that ML couldn't verify (100% - ml_handled_percent)
  const llmEscalatedCount = Math.round(((100 - mlHandledPercent) / 100) * totalStatements)

  // Calculate faithfulness stats from available data
  const faithfulnessData = result.faithfulness || {}
  const unfaithfulClaims = faithfulnessData.unfaithful_claims || []
  // Prosecutor issues = number of unfaithful claims found
  const prosecutorIssues = unfaithfulClaims.length

  return {
    overallScore: result.overall_score ?? result.overallScore ?? 0,
    qualityTier: result.quality_tier ?? result.qualityTier ?? 'poor',
    faithfulness: {
      score: faithfulnessData.score ?? 0,
      claims: transformClaims(faithfulnessData, unfaithfulClaims),
      prosecutorIssues: prosecutorIssues,
      defenderUpheld: Math.max(0, (faithfulnessData.claims?.length ?? prosecutorIssues) - prosecutorIssues),
      unfaithfulClaims: prosecutorIssues,
      debateSummary: faithfulnessData.debate_summary ?? faithfulnessData.debateSummary ?? faithfulnessData.reasoning,
    },
    relevance: {
      score: result.relevance?.overall_score ?? result.relevance?.score ?? 0,
      literalScore: result.relevance?.literal_score ?? result.relevance?.literalScore ?? 0,
      intentScore: result.relevance?.intent_score ?? result.relevance?.intentScore ?? 0,
      scopeScore: result.relevance?.scope_score ?? result.relevance?.scopeScore ?? 0,
      disagreementLevel: result.relevance?.disagreement_level ?? result.relevance?.disagreementLevel ?? 0,
      verdict: result.relevance?.verdict ?? '',
    },
    hallucination: {
      score: hallucinationData.score ?? 0,
      statementsAnalyzed: totalStatements,
      mlVerified: Math.max(0, mlVerifiedCount),
      llmEscalated: llmEscalatedCount,
      fabricationsFound: fabricationsCount + contradictionsCount,
    },
    constitutional: {
      score: result.constitutional?.overall_score ?? result.constitutional?.score ?? 0,
      complianceStatus: result.constitutional?.compliance_status ?? result.constitutional?.complianceStatus ?? 'non_compliant',
      principles: transformPrinciples(result.constitutional?.principle_scores, result.constitutional?.improvement_needed),
      criticalViolations: result.constitutional?.critical_violations ?? result.constitutional?.criticalViolations,
    },
    recommendations: result.recommendations ?? [],
    needsHumanReview: result.requires_human_review ?? result.needs_human_review ?? result.needsHumanReview ?? false,
  }
}

function transformClaims(faithfulness: any, unfaithfulClaims: string[]): any[] {
  // If we have explicit claims array, use it
  const claims = faithfulness?.claims || faithfulness?.extracted_claims || []
  if (claims.length > 0) {
    return claims.map((claim: any, index: number) => ({
      id: claim.id || `claim-${index}`,
      text: claim.text || claim.claim,
      type: claim.type || 'factual',
      importance: claim.importance || 'supporting',
      status: determineClaimStatus(claim, faithfulness),
      evidence: claim.evidence || claim.context_support,
      prosecution: claim.prosecution,
      defense: claim.defense,
      judgeRuling: claim.judge_ruling || claim.judgeRuling,
    }))
  }

  // Otherwise, create claims from unfaithful_claims array
  return unfaithfulClaims.map((claimText: string, index: number) => ({
    id: `claim-${index}`,
    text: claimText,
    type: 'factual',
    importance: 'critical',
    status: 'fabricated',
    evidence: null,
    prosecution: { argument: 'Claim not supported by context', severity: 'critical', type: 'unsupported' },
    defense: null,
    judgeRuling: { verdict: 'unfaithful', reasoning: faithfulness?.reasoning || 'Claim ruled unfaithful by judge' },
  }))
}

function determineClaimStatus(claim: any, faithfulness: any): 'grounded' | 'uncertain' | 'fabricated' {
  const unfaithfulClaims = faithfulness?.unfaithful_claims || []
  if (unfaithfulClaims.some((uc: any) => uc.claim === claim.text || uc.id === claim.id)) {
    return 'fabricated'
  }
  if (claim.status) return claim.status
  if (claim.is_faithful === false) return 'fabricated'
  if (claim.is_faithful === true) return 'grounded'
  return 'uncertain'
}

function transformPrinciples(principleScores: any, improvementNeeded?: string[]): any[] {
  // Handle if principleScores is an object (API returns {principle_id: score})
  if (principleScores && typeof principleScores === 'object' && !Array.isArray(principleScores)) {
    const principleNames: Record<string, string> = {
      no_fabrication: 'No Fabrication',
      accurate_attribution: 'Accurate Attribution',
      completeness: 'Completeness',
      safety: 'No Harmful Advice',
      uncertainty_expression: 'Uncertainty Expression',
    }

    return Object.entries(principleScores).map(([id, score]: [string, any]) => ({
      id,
      name: principleNames[id] || id,
      score: typeof score === 'number' ? score : 1,
      passed: typeof score === 'number' ? score >= 0.7 : true,
      violations: improvementNeeded?.filter(v => v.toLowerCase().includes(id.replace('_', ' '))) || [],
    }))
  }

  // Handle if principleScores is already an array
  if (!principleScores || !Array.isArray(principleScores)) return []
  return principleScores.map((p: any) => ({
    id: p.principle_id || p.id,
    name: p.name || p.principle_id || p.id,
    score: p.score ?? 1,
    passed: p.passed ?? (p.score >= 0.7),
    violations: p.violations,
  }))
}

// Async evaluation with polling
export async function evaluateRAGAsync(input: EvaluationInput): Promise<EvaluationResponse> {
  try {
    // Start async execution
    const startResponse = await fetch(`${API_URL}/api/v1/execute/async/rag-evaluation.evaluate_rag_response`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        input: {
          question: input.question,
          context: input.context,
          response: input.response,
          mode: input.mode || 'standard',
          domain: input.domain || 'general',
          model: input.model || 'openrouter/google/gemini-2.0-flash-001',
        },
      }),
    })

    if (!startResponse.ok) {
      throw new Error(`Failed to start evaluation: ${startResponse.status}`)
    }

    const { execution_id } = await startResponse.json()

    // Poll for result
    const maxAttempts = 60
    const pollInterval = 1000

    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      await new Promise(resolve => setTimeout(resolve, pollInterval))

      const statusResponse = await fetch(`${API_URL}/api/v1/executions/${execution_id}`)
      if (!statusResponse.ok) continue

      const status = await statusResponse.json()

      if (status.status === 'succeeded') {
        return {
          success: true,
          result: transformResult(status.result),
          executionId: execution_id,
        }
      }

      if (status.status === 'failed') {
        throw new Error(status.error || 'Evaluation failed')
      }
    }

    throw new Error('Evaluation timed out')
  } catch (error) {
    console.error('Async evaluation error:', error)
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error occurred',
    }
  }
}
