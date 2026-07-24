import { cn } from "../../lib/utils";
import type { LucideIcon } from "lucide-react";
import {
  Activity,
  LayoutGrid,
  Layers,
  Cpu,
  Play,
  Workflow,
  Settings,
  UserCircle,
  Package,
  Sun,
  Moon,
  Monitor,
  ShieldCheck,
  Shield,
  IdCard,
  FileText,
  Github,
  HelpCircle,
  Clock,
  Server,
  Lightbulb,
  FunctionSquare,
} from "lucide-react";

const icons = {
  activity: Activity,
  dashboard: LayoutGrid,
  "data-center": Layers,
  /** Agent nodes (microservice-style processes) */
  "agent-node": Server,
  /** Thinking / LLM-capable API endpoints */
  reasoner: Lightbulb,
  /** Deterministic callable APIs */
  skill: FunctionSquare,
  function: Cpu,
  run: Play,
  "flow-data": Workflow,
  settings: Settings,
  user: UserCircle,
  grid: LayoutGrid,
  package: Package,
  sun: Sun,
  moon: Moon,
  monitor: Monitor,
  "shield-check": ShieldCheck,
  shield: Shield,
  identification: IdCard,
  documentation: FileText,
  github: Github,
  support: HelpCircle,
  hourglass: Clock,
  history: Clock,
  lock: Shield,
} as const;

export interface IconProps {
  name: keyof typeof icons;
  className?: string;
  size?: number;
}

export function Icon({ name, className, size = 16 }: IconProps) {
  const IconComponent = icons[name] as LucideIcon;

  if (!IconComponent) {
    console.warn(`Icon "${name}" not found`);
    return null;
  }

  return (
    <IconComponent
      className={cn("shrink-0", className)}
      size={size}
    />
  );
}
