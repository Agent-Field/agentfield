'use client'

import Image from 'next/image'
import { useEvaluation } from '@/hooks/useEvaluation'
import { EvaluationInput } from '@/lib/types'
import { DashboardShell, DashboardHeader, DashboardMain } from '@/components/eval/DashboardShell'
import { InputPanel } from '@/components/eval/InputPanel'
import { ResultsPanel } from '@/components/eval/ResultsPanel'
import { PoweredBy } from '@/components/PoweredBy'

export default function Home() {
  const { status, result, error, notes, currentStep, evaluate, reset } = useEvaluation()

  const handleSubmit = async (input: EvaluationInput) => {
    await evaluate(input)
  }

  return (
    <DashboardShell>
      <DashboardHeader>
        <div className="flex items-center gap-3">
            <div className="h-8 w-8 rounded-lg bg-primary/10 border border-primary/20 flex items-center justify-center shadow-sm">
                <Image
                  src="/agentfield-icon-dark.svg"
                  alt="AgentField"
                  width={16}
                  height={16}
                />
            </div>
            <div>
              <h1 className="text-sm font-semibold tracking-tight">RAG Evaluation Studio</h1>
              <p className="text-[10px] text-muted-foreground font-mono">v0.1.0-beta</p>
            </div>
        </div>
        <div className="flex items-center gap-6">
             {/* You could add a 'Settings' or 'Help' button here */}
             <div className="hidden border-l pl-6 md:block">
                <PoweredBy />
             </div>
        </div>
      </DashboardHeader>

      <DashboardMain className="flex-col md:flex-row divide-y md:divide-y-0 md:divide-x">
        <InputPanel
            onSubmit={handleSubmit}
            isLoading={status === 'evaluating'}
            className="w-full md:w-[420px] lg:w-[480px] shrink-0 h-full overflow-hidden"
        />
        <ResultsPanel
            status={status}
            result={result}
            error={error}
            notes={notes}
            currentStep={currentStep}
            onReset={reset}
            className="flex-1 min-w-0 h-full overflow-hidden"
        />
      </DashboardMain>
    </DashboardShell>
  )
}
