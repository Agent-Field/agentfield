import React from 'react'
import { cn } from '@/lib/utils'

export function DashboardShell({ children, className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={cn("flex flex-col h-screen overflow-hidden bg-background font-sans", className)} {...props}>
      {children}
    </div>
  )
}

export function DashboardHeader({ children, className, ...props }: React.HTMLAttributes<HTMLElement>) {
  return (
    <header
      className={cn(
        "h-14 border-b flex items-center justify-between px-6 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 shrink-0 z-10",
        className
      )}
      {...props}
    >
      {children}
    </header>
  )
}

export function DashboardMain({ children, className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <main className={cn("flex-1 flex overflow-hidden", className)} {...props}>
      {children}
    </main>
  )
}
