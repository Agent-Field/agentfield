'use client'

import { useState } from 'react'
import { ChevronDown, Check, AlertTriangle, X, Gavel, ArrowRight } from 'lucide-react'
import { Claim, EvaluationResult } from '@/lib/types'
import { cn } from '@/lib/utils'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Separator } from '@/components/ui/separator'
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible'

interface ClaimBreakdownProps {
  result: EvaluationResult
}

type FilterType = 'all' | 'grounded' | 'uncertain' | 'fabricated'

export function ClaimBreakdown({ result }: ClaimBreakdownProps) {
  const [filter, setFilter] = useState<FilterType>('all')

  const claims = result.faithfulness.claims || []

  const filteredClaims = claims.filter((claim) => {
    if (filter === 'all') return true
    return claim.status === filter
  })

  // Count helper
  const getCount = (status: string) => claims.filter(c => c.status === status).length

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h3 className="text-lg font-medium leading-none tracking-tight">Claims Analysis</h3>
          <p className="text-sm text-muted-foreground mt-1.5">
            {claims.length} claims were identified and verified against the context.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <Badge
            variant={filter === 'all' ? 'default' : 'outline'}
            className="cursor-pointer hover:bg-primary/90"
            onClick={() => setFilter('all')}
          >
            All <span className="ml-1 opacity-70">{claims.length}</span>
          </Badge>
          <Badge
            variant={filter === 'grounded' ? 'default' : 'outline'}
            className="cursor-pointer hover:bg-primary/90"
            onClick={() => setFilter('grounded')}
          >
            Grounded <span className="ml-1 opacity-70">{getCount('grounded')}</span>
          </Badge>
          <Badge
            variant={filter === 'uncertain' ? 'secondary' : 'outline'}
            className="cursor-pointer hover:bg-secondary/80"
            onClick={() => setFilter('uncertain')}
          >
            Uncertain <span className="ml-1 opacity-70">{getCount('uncertain')}</span>
          </Badge>
          <Badge
            variant={filter === 'fabricated' ? 'destructive' : 'outline'}
            className="cursor-pointer hover:bg-destructive/90"
            onClick={() => setFilter('fabricated')}
          >
            Fabricated <span className="ml-1 opacity-70">{getCount('fabricated')}</span>
          </Badge>
        </div>
      </div>

      <div className="space-y-4">
        {filteredClaims.length === 0 ? (
          <div className="text-center py-12 border rounded-lg border-dashed text-muted-foreground">
             No claims found matching this filter.
          </div>
        ) : (
          filteredClaims.map((claim, index) => (
            <ClaimItem key={claim.id || index} claim={claim} />
          ))
        )}
      </div>
    </div>
  )
}

function ClaimItem({ claim }: { claim: Claim }) {
  const [isOpen, setIsOpen] = useState(false)

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'grounded': return <Badge variant="default" className="capitalize">{status}</Badge>
      case 'uncertain': return <Badge variant="secondary" className="capitalize">{status}</Badge>
      case 'fabricated': return <Badge variant="destructive" className="capitalize">{status}</Badge>
      default: return <Badge variant="outline" className="capitalize">{status}</Badge>
    }
  }

  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen}>
      <Card className="overflow-hidden">
        <CollapsibleTrigger asChild>
          <div className="flex items-start gap-4 p-6 cursor-pointer hover:bg-muted/40 transition-colors">
            <div className="flex-1 space-y-3">
              <div className="flex items-start justify-between gap-4">
                <p className="font-medium text-sm leading-relaxed">
                  "{claim.text}"
                </p>
                <ChevronDown className={cn(
                  "h-4 w-4 text-muted-foreground transition-transform duration-200 mt-0.5",
                  isOpen && "rotate-180"
                )} />
              </div>

              <div className="flex items-center gap-3">
                {getStatusBadge(claim.status)}

                {claim.type && (
                    <Badge variant="outline" className="font-normal text-muted-foreground border-border/60">
                        {claim.type.replace(/_/g, ' ')}
                    </Badge>
                )}

                {claim.evidence && (
                  <span className="text-xs text-muted-foreground hidden sm:inline-block border-l pl-3 truncate max-w-[300px]">
                    Evidence: {claim.evidence}
                  </span>
                )}
              </div>
            </div>
          </div>
        </CollapsibleTrigger>

        <CollapsibleContent>
            <Separator />
            <div className="p-6 bg-muted/5 space-y-6">

                {/* Debate Section Side-by-Side */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {/* Prosecution Pannel */}
                    <Card className="shadow-sm border-border/60">
                         <CardHeader className="pb-3">
                             <div className="flex items-center justify-between">
                                <CardTitle className="text-base font-medium flex items-center gap-2">
                                    Prosecution
                                </CardTitle>
                                {claim.prosecution && (
                                    <Badge variant="destructive" className="font-normal uppercase text-[10px] tracking-wider">
                                        {claim.prosecution.severity}
                                    </Badge>
                                )}
                             </div>
                         </CardHeader>
                         <CardContent>
                            {claim.prosecution ? (
                                <div className="space-y-4">
                                    <p className="text-sm text-foreground/90 leading-relaxed">
                                        {claim.prosecution.argument}
                                    </p>
                                    <div className="flex items-center gap-2 text-xs text-muted-foreground">
                                        <span className="font-medium">Type:</span>
                                        <span className="bg-muted px-2 py-0.5 rounded capitalize">
                                            {claim.prosecution.type.replace(/_/g, ' ')}
                                        </span>
                                    </div>
                                </div>
                            ) : (
                                <p className="text-sm text-muted-foreground italic">No prosecution arguments.</p>
                            )}
                         </CardContent>
                    </Card>

                    {/* Defense Panel */}
                    <Card className="shadow-sm border-border/60">
                         <CardHeader className="pb-3">
                             <div className="flex items-center justify-between">
                                <CardTitle className=" text-base font-medium flex items-center gap-2">
                                    Defense
                                </CardTitle>
                                {claim.defense && (
                                     <Badge variant="secondary" className="font-normal">
                                        Strength: {Math.round(claim.defense.strength * 100)}%
                                     </Badge>
                                )}
                             </div>
                         </CardHeader>
                         <CardContent>
                            {claim.defense ? (
                                <div className="space-y-4">
                                    <p className="text-sm text-foreground/90 leading-relaxed">
                                        {claim.defense.argument}
                                    </p>
                                    <div className="flex items-center gap-2 text-xs text-muted-foreground">
                                        <span className="font-medium">Support:</span>
                                        <span className="bg-muted px-2 py-0.5 rounded capitalize">
                                            {claim.defense.supportType.replace(/_/g, ' ')}
                                        </span>
                                    </div>
                                </div>
                            ) : (
                                <p className="text-sm text-muted-foreground italic">No defense arguments.</p>
                            )}
                         </CardContent>
                    </Card>
                </div>

                {/* Judge Ruling (Full Width) */}
                <Card className="border-primary/20 shadow-sm">
                    <CardHeader className="pb-3">
                        <div className="flex items-center justify-between">
                             <CardTitle className="text-base font-medium flex items-center gap-2 text-primary">
                                <Gavel className="h-4 w-4" />
                                Judge Ruling
                             </CardTitle>
                             {claim.judgeRuling && (
                                <Badge variant="outline" className="uppercase text-[10px] tracking-wider">
                                    {claim.judgeRuling.verdict}
                                </Badge>
                             )}
                        </div>
                    </CardHeader>
                    <CardContent>
                        {claim.judgeRuling ? (
                            <p className="text-sm text-foreground/90 leading-relaxed">
                                {claim.judgeRuling.reasoning}
                            </p>
                        ) : (
                            <p className="text-sm text-muted-foreground italic">Pending ruling...</p>
                        )}
                    </CardContent>
                </Card>

            </div>
        </CollapsibleContent>
      </Card>
    </Collapsible>
  )
}
