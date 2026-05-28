'use client'

import { useState, useRef, useEffect } from 'react'
import { Play, ChevronDown, Check } from 'lucide-react'
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
} from '@/components/ui/dropdown-menu'
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover'
import { Card, CardContent } from '@/components/ui/card'
import { cn } from '@/lib/utils'

interface EvaluationFormProps {
  onSubmit: (input: EvaluationInput) => void
  isLoading: boolean
}

export function EvaluationForm({ onSubmit, isLoading }: EvaluationFormProps) {
  const [question, setQuestion] = useState('')
  const [context, setContext] = useState('')
  const [response, setResponse] = useState('')
  const [mode, setMode] = useState<'quick' | 'standard' | 'thorough'>('standard')
  const [domain, setDomain] = useState<'general' | 'medical' | 'legal' | 'financial'>('general')
  const [model, setModel] = useState<string>(AVAILABLE_MODELS[0])
  const [modelInputOpen, setModelInputOpen] = useState(false)
  const [modelSearch, setModelSearch] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
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

  // Filter models based on search
  const filteredModels = AVAILABLE_MODELS.filter((m) =>
    m.toLowerCase().includes(modelSearch.toLowerCase())
  )

  // Focus input when popover opens
  useEffect(() => {
    if (modelInputOpen && inputRef.current) {
      inputRef.current.focus()
    }
  }, [modelInputOpen])

  return (
    <Card>
      <CardContent className="pt-6">
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Preset Selector */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" size="sm" className="gap-2">
                Try an Example
                <ChevronDown className="h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="start" className="w-72">
              {PRESETS.map((preset) => (
                <DropdownMenuItem
                  key={preset.id}
                  onClick={() => loadPreset(preset)}
                  className="flex flex-col items-start gap-1 py-3"
                >
                  <span className="font-medium">{preset.name}</span>
                  <span className="text-xs text-muted-foreground">
                    {preset.description}
                  </span>
                </DropdownMenuItem>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>

          {/* Input Fields */}
          <div className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="question">Question</Label>
              <Textarea
                id="question"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="What question was asked?"
                rows={2}
                className="resize-none"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="context">Context</Label>
              <Textarea
                id="context"
                value={context}
                onChange={(e) => setContext(e.target.value)}
                placeholder="What source documents or context was available?"
                rows={6}
                className="resize-none"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="response">Response</Label>
              <Textarea
                id="response"
                value={response}
                onChange={(e) => setResponse(e.target.value)}
                placeholder="What response did the RAG system generate?"
                rows={6}
                className="resize-none"
              />
            </div>
          </div>

          {/* Options Row */}
          <div className="flex flex-wrap items-center gap-4">
            <div className="flex items-center gap-2">
              <Label htmlFor="mode" className="text-muted-foreground">
                Mode
              </Label>
              <Select value={mode} onValueChange={(v) => setMode(v as typeof mode)}>
                <SelectTrigger id="mode" className="w-28">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="quick">Quick</SelectItem>
                  <SelectItem value="standard">Standard</SelectItem>
                  <SelectItem value="thorough">Thorough</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="flex items-center gap-2">
              <Label htmlFor="domain" className="text-muted-foreground">
                Domain
              </Label>
              <Select value={domain} onValueChange={(v) => setDomain(v as typeof domain)}>
                <SelectTrigger id="domain" className="w-28">
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

            {/* Model Combobox */}
            <div className="flex items-center gap-2">
              <Label className="text-muted-foreground">Model</Label>
              <Popover open={modelInputOpen} onOpenChange={setModelInputOpen}>
                <PopoverTrigger asChild>
                  <Button
                    variant="outline"
                    role="combobox"
                    aria-expanded={modelInputOpen}
                    className="w-80 justify-between font-mono text-xs"
                  >
                    <span className="truncate">{model}</span>
                    <ChevronDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
                  </Button>
                </PopoverTrigger>
                <PopoverContent className="w-80 p-0" align="start">
                  <div className="p-2 border-b">
                    <Input
                      ref={inputRef}
                      placeholder="Type model ID or select..."
                      value={modelSearch}
                      onChange={(e) => setModelSearch(e.target.value)}
                      className="font-mono text-xs"
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' && modelSearch.trim()) {
                          e.preventDefault()
                          setModel(modelSearch.trim())
                          setModelSearch('')
                          setModelInputOpen(false)
                        }
                      }}
                    />
                    <p className="text-xs text-muted-foreground mt-1.5 px-1">
                      Enter custom model ID or select from list
                    </p>
                  </div>
                  <div className="max-h-48 overflow-y-auto p-1">
                    {/* Show custom input option if search doesn't match any model */}
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
                        <span className="text-muted-foreground">Use:</span>
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
                    {filteredModels.length === 0 && !modelSearch.trim() && (
                      <div className="text-sm text-muted-foreground text-center py-2">
                        No models found
                      </div>
                    )}
                  </div>
                </PopoverContent>
              </Popover>
            </div>
          </div>

          {/* Submit Button */}
          <Button
            type="submit"
            disabled={isLoading || !isValid}
            className="w-full"
            size="lg"
          >
            <Play className="h-4 w-4 mr-2" />
            {isLoading ? 'Evaluating...' : 'Evaluate Response'}
          </Button>
        </form>
      </CardContent>
    </Card>
  )
}
