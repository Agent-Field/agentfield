'use client'

import { useState, useRef, useEffect } from 'react'
import { Play, ChevronDown, Check, Settings2, Sparkles } from 'lucide-react'
import { EvaluationInput, AVAILABLE_MODELS } from '@/lib/types'
import { PRESETS, Preset } from '@/lib/presets'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { Label } from '@/components/ui/label'
import { Input } from '@/components/ui/input'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  DropdownMenuLabel,
  DropdownMenuSeparator,
} from '@/components/ui/dropdown-menu'
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Separator } from '@/components/ui/separator'
import { cn } from '@/lib/utils'

interface InputPanelProps {
  onSubmit: (input: EvaluationInput) => void
  isLoading: boolean
  className?: string
}

export function InputPanel({ onSubmit, isLoading, className }: InputPanelProps) {
  const [question, setQuestion] = useState('')
  const [context, setContext] = useState('')
  const [response, setResponse] = useState('')
  const [mode, setMode] = useState<'quick' | 'standard' | 'thorough'>('standard')
  const [domain, setDomain] = useState<'general' | 'medical' | 'legal' | 'financial'>('general')
  const [model, setModel] = useState<string>(AVAILABLE_MODELS[0])
  const [modelInputOpen, setModelInputOpen] = useState(false)
  const [modelSearch, setModelSearch] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)

  const handleSubmit = (e?: React.FormEvent) => {
    e?.preventDefault()
    if (!question.trim() || !context.trim() || !response.trim()) return
    onSubmit({ question, context, response, mode, domain, model })
  }

  const loadPreset = (preset: Preset) => {
    setQuestion(preset.question)
    setContext(preset.context)
    setResponse(preset.response)
    setMode(preset.mode || 'standard')
    setDomain(preset.domain || 'general')
  }

  const isValid = question.trim() && context.trim() && response.trim()
  const filteredModels = AVAILABLE_MODELS.filter((m) =>
    m.toLowerCase().includes(modelSearch.toLowerCase())
  )

  useEffect(() => {
    if (modelInputOpen && inputRef.current) {
      inputRef.current.focus()
    }
  }, [modelInputOpen])

  return (
    <div className={cn("flex flex-col h-full bg-muted/10 border-r", className)}>
      <div className="p-4 border-b space-y-4 bg-background/50 backdrop-blur-sm">
        <div className="flex items-center justify-between">
          <h2 className="font-semibold text-sm text-muted-foreground uppercase tracking-wider">Configuration</h2>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="sm" className="h-8 gap-1.5 text-muted-foreground hover:text-foreground">
                <Sparkles className="h-3.5 w-3.5" />
                <span>Load Existing</span>
                <ChevronDown className="h-3 w-3 opacity-50" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-64">
              <DropdownMenuLabel>Example Scenarios</DropdownMenuLabel>
              <DropdownMenuSeparator />
              {PRESETS.map((preset) => (
                <DropdownMenuItem
                  key={preset.id}
                  onClick={() => loadPreset(preset)}
                  className="flex flex-col items-start gap-0.5 py-2 cursor-pointer"
                >
                  <span className="font-medium text-sm">{preset.name}</span>
                  <span className="text-xs text-muted-foreground">
                    {preset.description}
                  </span>
                </DropdownMenuItem>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div className="space-y-1.5">
            <Label className="text-xs font-normal text-muted-foreground">Evaluation Mode</Label>
            <Select value={mode} onValueChange={(v) => setMode(v as typeof mode)}>
              <SelectTrigger className="h-8 text-xs bg-background">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="quick">Quick</SelectItem>
                <SelectItem value="standard">Standard</SelectItem>
                <SelectItem value="thorough">Thorough</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-1.5">
            <Label className="text-xs font-normal text-muted-foreground">Domain Context</Label>
            <Select value={domain} onValueChange={(v) => setDomain(v as typeof domain)}>
              <SelectTrigger className="h-8 text-xs bg-background">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="general">General</SelectItem>
                <SelectItem value="medical">Medical</SelectItem>
                <SelectItem value="legal">Legal</SelectItem>
                <SelectItem value="financial">Financial</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>

        <div className="space-y-1.5">
           <Label className="text-xs font-normal text-muted-foreground">Judge Model</Label>
           <Popover open={modelInputOpen} onOpenChange={setModelInputOpen}>
              <PopoverTrigger asChild>
                <Button
                  variant="outline"
                  role="combobox"
                  aria-expanded={modelInputOpen}
                  className="w-full justify-between font-mono text-xs h-8 bg-background"
                >
                  <span className="truncate">{model}</span>
                  <ChevronDown className="ml-2 h-3 w-3 shrink-0 opacity-50" />
                </Button>
              </PopoverTrigger>
              <PopoverContent className="w-[var(--radix-popover-trigger-width)] p-0" align="start">
                <div className="p-2 border-b">
                  <Input
                    ref={inputRef}
                    placeholder="Search model..."
                    value={modelSearch}
                    onChange={(e) => setModelSearch(e.target.value)}
                    className="font-mono text-xs h-8"
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && modelSearch.trim()) {
                        e.preventDefault()
                        setModel(modelSearch.trim())
                        setModelSearch('')
                        setModelInputOpen(false)
                      }
                    }}
                  />
                </div>
                <div className="max-h-48 overflow-y-auto p-1">
                  {modelSearch.trim() && !AVAILABLE_MODELS.includes(modelSearch.trim() as any) && (
                    <button
                      type="button"
                      className="w-full text-left px-2 py-1.5 rounded text-sm hover:bg-muted flex items-center gap-2"
                      onClick={() => {
                        setModel(modelSearch.trim())
                        setModelSearch('')
                        setModelInputOpen(false)
                      }}
                    >
                      <span className="text-muted-foreground text-xs">Use:</span>
                      <span className="font-mono text-xs truncate">{modelSearch.trim()}</span>
                    </button>
                  )}
                  {filteredModels.map((m) => (
                    <button
                      key={m}
                      type="button"
                      className={cn(
                        "w-full text-left px-2 py-1.5 rounded text-xs font-mono hover:bg-muted flex items-center gap-2",
                        model === m && "bg-muted"
                      )}
                      onClick={() => {
                        setModel(m)
                        setModelSearch('')
                        setModelInputOpen(false)
                      }}
                    >
                      {model === m && <Check className="h-3 w-3 shrink-0" />}
                      <span className={cn("truncate", model !== m && "ml-5")}>{m}</span>
                    </button>
                  ))}
                </div>
              </PopoverContent>
            </Popover>
        </div>
      </div>

      <ScrollArea className="flex-1">
        <div className="p-4 space-y-6">
          <div className="space-y-3">
             <div className="flex items-center justify-between">
              <Label htmlFor="question" className="font-medium text-foreground">User Query</Label>
              <span className="text-[10px] text-muted-foreground bg-muted px-1.5 py-0.5 rounded">Required</span>
            </div>
            <Textarea
              id="question"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="What question was asked?"
              className="resize-none min-h-[80px] text-sm bg-background border-muted-foreground/20 focus-visible:border-primary/50"
            />
          </div>

          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <Label htmlFor="context" className="font-medium text-foreground">Retrieved Context</Label>
              <span className="text-[10px] text-muted-foreground bg-muted px-1.5 py-0.5 rounded">Required</span>
            </div>
            <Textarea
              id="context"
              value={context}
              onChange={(e) => setContext(e.target.value)}
              placeholder="Paste the source passages or context here..."
              className="resize-none min-h-[160px] text-sm bg-background border-muted-foreground/20 focus-visible:border-primary/50 font-mono"
            />
          </div>

          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <Label htmlFor="response" className="font-medium text-foreground">Generated Response</Label>
              <span className="text-[10px] text-muted-foreground bg-muted px-1.5 py-0.5 rounded">Required</span>
            </div>
            <Textarea
              id="response"
              value={response}
              onChange={(e) => setResponse(e.target.value)}
              placeholder="The answer generated by the system..."
              className="resize-none min-h-[120px] text-sm bg-background border-muted-foreground/20 focus-visible:border-primary/50"
            />
          </div>
        </div>
      </ScrollArea>

      <div className="p-4 border-t bg-background/50 backdrop-blur-sm">
        <Button
          onClick={() => handleSubmit()}
          disabled={isLoading || !isValid}
          className="w-full shadow-lg hover:shadow-xl transition-all"
          size="lg"
        >
          {isLoading ? (
             <>
               <span className="animate-spin mr-2">⟳</span> Evaluating...
             </>
          ) : (
             <>
               <Play className="h-4 w-4 mr-2 fill-current" /> Run Evaluation
             </>
          )}
        </Button>
      </div>
    </div>
  )
}
