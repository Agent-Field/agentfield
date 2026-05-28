// Evaluation input types
export interface EvaluationInput {
  question: string
  context: string
  response: string
  mode?: 'quick' | 'standard' | 'thorough'
  domain?: 'general' | 'medical' | 'legal' | 'financial'
  model?: string
}

// Available models for selection (using full OpenRouter model IDs)
export const AVAILABLE_MODELS = [
  'openrouter/openai/gpt-4o',
  'openrouter/google/gemini-2.5-flash',
  'openrouter/anthropic/claude-3.5-sonnet',
  'openrouter/meta-llama/llama-3.3-70b-instruct',
  'openrouter/deepseek/deepseek-chat-v3-0324',
] as const

export type ModelId = typeof AVAILABLE_MODELS[number] | string

// Claim types
export interface Claim {
  id: string
  text: string
  type: 'factual' | 'inferential' | 'opinion'
  importance: 'critical' | 'supporting' | 'minor'
  status: 'grounded' | 'uncertain' | 'fabricated'
  evidence?: string
  prosecution?: {
    argument: string
    severity: 'critical' | 'major' | 'minor'
    type: 'unsupported' | 'contradicted' | 'exaggerated' | 'out_of_context'
  }
  defense?: {
    argument: string
    supportType: 'direct_support' | 'implicit_support' | 'reasonable_inference' | 'acknowledged_issue'
    strength: number
  }
  judgeRuling?: {
    verdict: string
    reasoning: string
  }
}

// Metric result types
export interface FaithfulnessResult {
  score: number
  claims: Claim[]
  prosecutorIssues: number
  defenderUpheld: number
  unfaithfulClaims: number
  debateSummary?: string
}

export interface RelevanceResult {
  score: number
  literalScore: number
  intentScore: number
  scopeScore: number
  disagreementLevel: number
  verdict: string
  juryVotes?: {
    literal: { score: number; reasoning: string }
    intent: { score: number; reasoning: string }
    scope: { score: number; reasoning: string }
  }
}

export interface HallucinationResult {
  score: number
  statementsAnalyzed: number
  mlVerified: number
  llmEscalated: number
  fabricationsFound: number
  statements?: {
    text: string
    status: 'verified' | 'uncertain' | 'fabricated'
    verifiedBy: 'ml' | 'llm'
    explanation?: string
  }[]
}

export interface ConstitutionalResult {
  score: number
  complianceStatus: 'compliant' | 'minor_issues' | 'major_issues' | 'non_compliant'
  principles: {
    id: string
    name: string
    score: number
    passed: boolean
    violations?: string[]
  }[]
  criticalViolations?: string[]
}

// Full evaluation result
export interface EvaluationResult {
  overallScore: number
  qualityTier: 'excellent' | 'good' | 'acceptable' | 'poor' | 'critical'
  faithfulness: FaithfulnessResult
  relevance: RelevanceResult
  hallucination: HallucinationResult
  constitutional: ConstitutionalResult
  recommendations?: string[]
  needsHumanReview?: boolean
}

// API response types
export interface EvaluationResponse {
  success: boolean
  result?: EvaluationResult
  error?: string
  executionId?: string
  runId?: string
  workflowId?: string
}

// Workflow note for SSE streaming
export interface WorkflowNote {
  type: 'note' | 'error' | 'complete'
  message: string
  tags: string[]
  timestamp: string
}

// Generic execution event shape from UI SSE stream
export interface ExecutionEventPayload {
  type?: string
  execution_id?: string
  workflow_id?: string
  run_id?: string
  status?: string
  data?: any
  timestamp?: string
}

// Async execution response
export interface AsyncExecutionResponse {
  execution_id: string
  status: 'pending' | 'running' | 'succeeded' | 'failed'
}
