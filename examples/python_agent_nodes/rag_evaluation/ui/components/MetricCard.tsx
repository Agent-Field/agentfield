'use client'

import { Scale, Target, Search, ScrollText, Check, AlertTriangle, X } from 'lucide-react'
import { EvaluationResult } from '@/lib/types'
import { Card, CardContent, CardHeader } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Separator } from '@/components/ui/separator'
import { MarkdownContent } from './MarkdownContent'

interface MetricCardProps {
  metric: 'faithfulness' | 'relevance' | 'hallucination' | 'constitutional'
  result: EvaluationResult
}

const metricConfig = {
  faithfulness: {
    icon: Scale,
    title: 'Faithfulness',
    subtitle: 'Adversarial Debate Pattern',
  },
  relevance: {
    icon: Target,
    title: 'Relevance',
    subtitle: 'Multi-Jury Consensus',
  },
  hallucination: {
    icon: Search,
    title: 'Hallucination',
    subtitle: 'Hybrid ML + LLM Verification',
  },
  constitutional: {
    icon: ScrollText,
    title: 'Constitutional',
    subtitle: 'Principles-Based Compliance',
  },
}

function formatScore(score: number): string {
  return `${Math.round(score * 100)}%`
}

export function MetricCard({ metric, result }: MetricCardProps) {
  const config = metricConfig[metric]
  const Icon = config.icon
  const data = result[metric]

  return (
    <Card className="flex flex-col">
      {/* Header */}
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-muted">
              <Icon className="h-5 w-5 text-muted-foreground" />
            </div>
            <div>
              <h3 className="font-medium text-foreground">{config.title}</h3>
              <p className="text-xs text-muted-foreground">{config.subtitle}</p>
            </div>
          </div>
          <span className="text-2xl font-bold tabular-nums">
            {formatScore(data.score)}
          </span>
        </div>
      </CardHeader>

      <Separator />

      {/* Content with Tabs */}
      <CardContent className="flex-1 pt-4">
        <Tabs defaultValue="summary" className="h-full">
          <TabsList className="grid w-full grid-cols-2 mb-4">
            <TabsTrigger value="summary">Summary</TabsTrigger>
            <TabsTrigger value="details">Details</TabsTrigger>
          </TabsList>

          <TabsContent value="summary" className="mt-0">
            {metric === 'faithfulness' && (
              <div className="space-y-3">
                <div className="grid grid-cols-3 gap-3 text-center">
                  <div className="space-y-1">
                    <div className="text-2xl font-bold tabular-nums">
                      {result.faithfulness.prosecutorIssues}
                    </div>
                    <div className="text-xs text-muted-foreground">Issues Found</div>
                  </div>
                  <div className="space-y-1">
                    <div className="text-2xl font-bold tabular-nums">
                      {result.faithfulness.defenderUpheld}
                    </div>
                    <div className="text-xs text-muted-foreground">Claims Upheld</div>
                  </div>
                  <div className="space-y-1">
                    <div className={`text-2xl font-bold tabular-nums ${
                      result.faithfulness.unfaithfulClaims > 0 ? 'text-destructive' : 'text-emerald-500'
                    }`}>
                      {result.faithfulness.unfaithfulClaims}
                    </div>
                    <div className="text-xs text-muted-foreground">Unfaithful</div>
                  </div>
                </div>
              </div>
            )}

            {metric === 'relevance' && (
              <div className="space-y-3">
                {['literal', 'intent', 'scope'].map((type) => {
                  const score = result.relevance[`${type}Score` as keyof typeof result.relevance] as number
                  return (
                    <div key={type} className="space-y-1.5">
                      <div className="flex justify-between text-sm">
                        <span className="text-muted-foreground capitalize">{type}</span>
                        <span className="font-mono tabular-nums">{formatScore(score)}</span>
                      </div>
                      <Progress value={score * 100} className="h-2" />
                    </div>
                  )
                })}
              </div>
            )}

            {metric === 'hallucination' && (
              <div className="space-y-3">
                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-1 text-center p-3 rounded-lg bg-muted/50">
                    <div className="text-2xl font-bold tabular-nums">
                      {result.hallucination.statementsAnalyzed}
                    </div>
                    <div className="text-xs text-muted-foreground">Statements</div>
                  </div>
                  <div className="space-y-1 text-center p-3 rounded-lg bg-muted/50">
                    <div className="text-2xl font-bold tabular-nums">
                      {result.hallucination.mlVerified}
                    </div>
                    <div className="text-xs text-muted-foreground">ML Verified</div>
                  </div>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">LLM Escalated</span>
                  <span className="font-mono tabular-nums">{result.hallucination.llmEscalated}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Fabrications Found</span>
                  <span className={`font-mono tabular-nums ${
                    result.hallucination.fabricationsFound > 0 ? 'text-destructive' : 'text-emerald-500'
                  }`}>
                    {result.hallucination.fabricationsFound}
                  </span>
                </div>
              </div>
            )}

            {metric === 'constitutional' && (
              <div className="space-y-2">
                {result.constitutional.principles.slice(0, 5).map((principle) => (
                  <div key={principle.id} className="flex items-center justify-between py-1">
                    <span className="text-sm text-muted-foreground">
                      {principle.name.replace(/_/g, ' ')}
                    </span>
                    <Badge
                      variant={principle.passed ? 'default' : principle.score >= 0.5 ? 'secondary' : 'destructive'}
                      className="gap-1"
                    >
                      {principle.passed ? (
                        <Check className="h-3 w-3" />
                      ) : principle.score >= 0.5 ? (
                        <AlertTriangle className="h-3 w-3" />
                      ) : (
                        <X className="h-3 w-3" />
                      )}
                      {principle.passed ? 'passed' : principle.score >= 0.5 ? 'issue' : 'failed'}
                    </Badge>
                  </div>
                ))}
              </div>
            )}
          </TabsContent>

          <TabsContent value="details" className="mt-0">
            <ScrollArea className="h-48">
              {metric === 'faithfulness' && (
                <div>
                  <h4 className="text-sm font-medium mb-2">Debate Summary</h4>
                  {result.faithfulness.debateSummary ? (
                    <MarkdownContent content={result.faithfulness.debateSummary} />
                  ) : (
                    <p className="text-sm text-muted-foreground">No debate summary available</p>
                  )}
                </div>
              )}

              {metric === 'relevance' && (
                <div>
                  <h4 className="text-sm font-medium mb-2">Jury Verdict</h4>
                  {result.relevance.verdict ? (
                    <MarkdownContent content={result.relevance.verdict} />
                  ) : (
                    <p className="text-sm text-muted-foreground">No verdict summary available</p>
                  )}
                  {result.relevance.disagreementLevel > 0.3 && (
                    <div className="mt-3 p-2 rounded border border-amber-500/20 bg-amber-500/5">
                      <p className="text-xs text-amber-500">
                        Note: Jury had {formatScore(result.relevance.disagreementLevel)} disagreement
                      </p>
                    </div>
                  )}
                </div>
              )}

              {metric === 'hallucination' && (
                <div>
                  <h4 className="text-sm font-medium mb-2">Verification Process</h4>
                  <p className="text-sm text-muted-foreground">
                    The hybrid verification process first used ML models (embeddings + NLI) to verify
                    {' '}{result.hallucination.mlVerified} statements. {result.hallucination.llmEscalated > 0 && (
                      <>{result.hallucination.llmEscalated} uncertain cases were escalated to LLM for deeper analysis.</>
                    )}
                  </p>
                  {result.hallucination.fabricationsFound > 0 && (
                    <div className="mt-3 p-2 rounded border border-destructive/20 bg-destructive/5">
                      <p className="text-xs text-destructive">
                        {result.hallucination.fabricationsFound} fabrication(s) detected
                      </p>
                    </div>
                  )}
                </div>
              )}

              {metric === 'constitutional' && (
                <div>
                  {result.constitutional.criticalViolations && result.constitutional.criticalViolations.length > 0 ? (
                    <>
                      <h4 className="text-sm font-medium text-destructive mb-2">Critical Violations</h4>
                      <ul className="space-y-2">
                        {result.constitutional.criticalViolations.map((v, i) => (
                          <li key={i} className="flex items-start gap-2 text-sm text-muted-foreground">
                            <X className="h-4 w-4 text-destructive mt-0.5 flex-shrink-0" />
                            <span>{typeof v === 'string' ? v : JSON.stringify(v)}</span>
                          </li>
                        ))}
                      </ul>
                    </>
                  ) : (
                    <>
                      <h4 className="text-sm font-medium mb-2">Compliance Status</h4>
                      <Badge variant={
                        result.constitutional.complianceStatus === 'compliant' ? 'default' :
                        result.constitutional.complianceStatus === 'minor_issues' ? 'secondary' : 'destructive'
                      } className="capitalize">
                        {result.constitutional.complianceStatus.replace(/_/g, ' ')}
                      </Badge>
                      <p className="text-sm text-muted-foreground mt-2">
                        All constitutional principles have been evaluated.
                        {result.constitutional.principles.filter(p => p.passed).length} of{' '}
                        {result.constitutional.principles.length} principles passed.
                      </p>
                    </>
                  )}
                </div>
              )}
            </ScrollArea>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}
