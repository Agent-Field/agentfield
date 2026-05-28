'use client'

import Image from 'next/image'

export function PoweredBy({ className }: { className?: string }) {
  return (
    <div className={className}>
      <a
        href="https://github.com/Agent-Field/agentfield"
        target="_blank"
        rel="noopener noreferrer"
        className="inline-flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground transition-colors"
      >
        <span>Powered by</span>
        <Image
          src="/agentfield-logo-dark.svg"
          alt="AgentField"
          width={80}
          height={16}
          className="opacity-80 hover:opacity-100 transition-opacity"
        />
      </a>
    </div>
  )
}
