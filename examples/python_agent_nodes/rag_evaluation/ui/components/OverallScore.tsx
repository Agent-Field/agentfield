'use client'

import { EvaluationResult } from '@/lib/types'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { AlertTriangle } from 'lucide-react'

interface OverallScoreProps {
  result: EvaluationResult
}

function formatScore(score: number): string {
  return `${Math.round(score * 100)}%`
}

function getQualityVariant(tier: string): 'default' | 'secondary' | 'destructive' | 'outline' {
  switch (tier) {
    case 'excellent':
    case 'good':
      return 'default'
    case 'acceptable':
      return 'secondary'
    case 'poor':
    case 'critical':
      return 'destructive'
    default:
      return 'outline'
  }
}

export function OverallScore({ result }: OverallScoreProps) {
  const { overallScore, qualityTier, faithfulness, relevance, hallucination, constitutional } = result

  const metrics = [
    { name: 'Faithfulness', score: faithfulness.score },
    { name: 'Relevance', score: relevance.score },
    { name: 'Hallucination', score: hallucination.score },
    { name: 'Constitutional', score: constitutional.score },
  ]

  return (
    <Card>
      <CardHeader className="pb-4">
        <CardTitle className="text-lg font-medium">Overall Score</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Main Score Display */}
        <div className="flex items-start justify-between">
          <div className="space-y-1">
            <div className="flex items-baseline gap-3">
              <span className="text-4xl font-bold tabular-nums">
                {formatScore(overallScore)}
              </span>
              <Badge variant={getQualityVariant(qualityTier)} className="capitalize">
                {qualityTier}
              </Badge>
            </div>
            <p className="text-sm text-muted-foreground">
              Quality assessment of the RAG response
            </p>
          </div>
        </div>

        {/* Metric Progress Bars */}
        <div className="space-y-3">
          {metrics.map((metric) => (
            <div key={metric.name} className="space-y-1.5">
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground">{metric.name}</span>
                <span className="font-mono tabular-nums">{formatScore(metric.score)}</span>
              </div>
              <Progress value={metric.score * 100} className="h-2" />
            </div>
          ))}
        </div>

        {/* Recommendations */}
        {result.recommendations && result.recommendations.length > 0 && (
          <div className="rounded-lg border border-amber-500/20 bg-amber-500/5 p-4">
            <div className="flex items-center gap-2 mb-2">
              <AlertTriangle className="h-4 w-4 text-amber-500" />
              <span className="text-sm font-medium text-amber-500">Recommendations</span>
            </div>
            <ul className="space-y-1">
              {result.recommendations.slice(0, 3).map((rec, i) => (
                <li key={i} className="text-sm text-muted-foreground flex items-start gap-2">
                  <span className="text-amber-500 mt-1">•</span>
                  <span>{rec}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
