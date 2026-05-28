'use client'

import React from 'react'
import { RotateCcw, AlertCircle, FileJson, BarChart2, ListChecks, Activity } from 'lucide-react'
import { EvaluationResult, WorkflowNote } from '@/lib/types'
import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { ScrollArea } from '@/components/ui/scroll-area'
import { LoadingState } from '@/components/LoadingState'
import { MetricCard } from '@/components/MetricCard'
import { OverallScore } from '@/components/OverallScore'
import { ClaimBreakdown } from '@/components/ClaimBreakdown'
import { cn } from '@/lib/utils'

interface ResultsPanelProps {
  status: 'idle' | 'evaluating' | 'success' | 'error'
  result: EvaluationResult | null
  error: string | null
  notes: WorkflowNote[]
  currentStep: string | null
  onReset: () => void
  className?: string
}

export function ResultsPanel({
  status,
  result,
  error,
  notes,
  currentStep,
  onReset,
  className
}: ResultsPanelProps) {

  if (status === 'idle') {
    return (
      <div className={cn("flex flex-col items-center justify-center h-full text-center p-8 text-muted-foreground bg-muted/5", className)}>
        <div className="w-20 h-20 rounded-full bg-muted/50 flex items-center justify-center mb-6 ring-1 ring-border">
          <Activity className="h-10 w-10 opacity-40" />
        </div>
        <h3 className="text-xl font-medium text-foreground tracking-tight">Ready to Evaluate</h3>
        <p className="max-w-sm mt-3 text-sm text-muted-foreground leading-relaxed">
          Configure your evaluation parameters and inputs on the left, then click "Run Evaluation".
        </p>
      </div>
    )
  }

  if (status === 'evaluating') {
    return (
      <div className={cn("h-full p-8 flex items-center justify-center bg-muted/5", className)}>
        <LoadingState notes={notes} currentStep={currentStep} />
      </div>
    )
  }

  if (status === 'error') {
    return (
      <div className={cn("h-full p-8 flex items-center justify-center bg-muted/5", className)}>
        <div className="max-w-md w-full">
            <div className="border border-destructive/50 bg-destructive/5 rounded-xl p-8 text-center shadow-sm">
                <div className="w-12 h-12 rounded-full bg-destructive/10 flex items-center justify-center mx-auto mb-4">
                    <AlertCircle className="h-6 w-6 text-destructive" />
                </div>
                <h3 className="text-lg font-semibold text-destructive mb-2">
                Evaluation Failed
                </h3>
                <p className="text-sm text-muted-foreground mb-6 leading-relaxed">
                {error || 'An unknown error occurred'}
                </p>
                <Button variant="outline" onClick={onReset} className="min-w-[120px]">
                Try Again
                </Button>
            </div>
        </div>
      </div>
    )
  }

  return (
    <div className={cn("flex flex-col h-full bg-background font-sans min-w-0", className)}>
      {/* Header */}
      <div className="h-16 border-b flex items-center justify-between px-6 bg-background shrink-0 z-10">
        <div className="flex items-center gap-2">
            <h2 className="font-semibold text-lg tracking-tight">Evaluation Report</h2>
        </div>
        <Button variant="ghost" size="sm" onClick={onReset} className="gap-2 text-muted-foreground hover:text-foreground">
          <RotateCcw className="h-4 w-4" />
          New Evaluation
        </Button>
      </div>

      <div className="flex-1 overflow-hidden flex flex-col min-w-0">
        <Tabs defaultValue="metrics" className="flex-1 flex flex-col min-h-0 min-w-0">
          <div className="px-6 border-b bg-muted/5 shrink-0">
            <TabsList className="w-full justify-start h-12 bg-transparent p-0 gap-6">
               <TabsTrigger
                 value="metrics"
                 className="data-[state=active]:bg-transparent data-[state=active]:shadow-none data-[state=active]:border-b-2 data-[state=active]:border-primary data-[state=active]:text-primary rounded-none px-0 h-12 text-muted-foreground hover:text-foreground transition-colors"
                >
                  <BarChart2 className="h-4 w-4 mr-2" />
                  Metrics Overview
               </TabsTrigger>
               <TabsTrigger
                 value="claims"
                 className="data-[state=active]:bg-transparent data-[state=active]:shadow-none data-[state=active]:border-b-2 data-[state=active]:border-primary data-[state=active]:text-primary rounded-none px-0 h-12 text-muted-foreground hover:text-foreground transition-colors"
                >
                   <ListChecks className="h-4 w-4 mr-2" />
                   Claim Analysis
               </TabsTrigger>
               <TabsTrigger
                 value="json"
                 className="data-[state=active]:bg-transparent data-[state=active]:shadow-none data-[state=active]:border-b-2 data-[state=active]:border-primary data-[state=active]:text-primary rounded-none px-0 h-12 text-muted-foreground hover:text-foreground transition-colors"
                >
                  <FileJson className="h-4 w-4 mr-2" />
                  Raw Output
               </TabsTrigger>
            </TabsList>
          </div>

          <div className="flex-1 overflow-y-auto bg-muted/5 min-w-0">
            <div className="max-w-6xl mx-auto w-full p-8 min-w-0">
                <TabsContent value="metrics" className="m-0 space-y-8 focus-visible:outline-none">
                    {result && (
                    <div className="space-y-8">
                        <OverallScore result={result} />
                        <div>
                            <h3 className="text-lg font-semibold mb-4 tracking-tight">Detailed Metrics</h3>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                <MetricCard metric="faithfulness" result={result} />
                                <MetricCard metric="relevance" result={result} />
                                <MetricCard metric="hallucination" result={result} />
                                <MetricCard metric="constitutional" result={result} />
                            </div>
                        </div>
                    </div>
                    )}
                </TabsContent>

                <TabsContent value="claims" className="m-0 focus-visible:outline-none">
                    {result && <ClaimBreakdown result={result} />}
                </TabsContent>

                <TabsContent value="json" className="m-0 focus-visible:outline-none">
                    {result && (
                        <div className="rounded-xl border bg-card overflow-hidden">
                            <ScrollArea className="h-[calc(100vh-320px)] w-full">
                                <div className="p-4">
                                    <pre className="font-mono text-xs text-muted-foreground leading-relaxed whitespace-pre-wrap break-all">
                                        {JSON.stringify(result, null, 2)}
                                    </pre>
                                </div>
                            </ScrollArea>
                        </div>
                    )}
                </TabsContent>
            </div>
          </div>
        </Tabs>
      </div>
    </div>
  )
}
