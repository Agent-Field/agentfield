import {
  LayoutDashboard,
  Play,
  Server,
  FlaskConical,
  Settings,
  KeyRound,
  FileCheck2,
  BookOpen,
  Github,
  type LucideIcon,
} from "lucide-react";

export type ResourceLinkItem = {
  title: string;
  icon: LucideIcon;
  href: string;
};

export type NavItem = {
  title: string;
  icon: LucideIcon;
  path: string;
};

export type NavGroup = {
  label: string;
  items: NavItem[];
};

export const navigation: NavGroup[] = [
  {
    label: "Build",
    items: [
      { title: "Dashboard", icon: LayoutDashboard, path: "/dashboard" },
      { title: "Playground", icon: FlaskConical, path: "/playground" },
      { title: "Runs", icon: Play, path: "/runs" },
      { title: "Agent nodes", icon: Server, path: "/agents" },
    ],
  },
  {
    label: "Govern",
    items: [
      { title: "Access management", icon: KeyRound, path: "/access" },
      { title: "Provenance", icon: FileCheck2, path: "/verify" },
      { title: "Settings", icon: Settings, path: "/settings" },
    ],
  },
];

/** External links shown below Platform nav (opens in new tab). */
export const resourceLinks: ResourceLinkItem[] = [
  {
    title: "Docs",
    icon: BookOpen,
    href: "https://agentfield.ai/docs",
  },
  {
    title: "GitHub",
    icon: Github,
    href: "https://github.com/Agent-Field/agentfield/",
  },
];
