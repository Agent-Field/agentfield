import {
  Activity,
  LayoutDashboard,
  Play,
  Server,
  FlaskConical,
  Settings,
  KeyRound,
  FileCheck2,
  BookOpen,
  Github,
  Webhook,
  Plug,
  type LucideIcon,
} from "lucide-react";

export type ResourceLinkItem = {
  title: string;
  icon: LucideIcon;
  href: string;
};

export type NavLeaf = {
  title: string;
  icon: LucideIcon;
  path: string;
};

export type NavBranch = {
  title: string;
  icon: LucideIcon;
  /** Nested children. The branch button itself toggles the section open. */
  children: NavLeaf[];
};

export type NavItem = NavLeaf | NavBranch;

export type NavGroup = {
  label: string;
  items: NavItem[];
};

export function isNavBranch(item: NavItem): item is NavBranch {
  return (item as NavBranch).children !== undefined;
}

export const navigation: NavGroup[] = [
  {
    label: "Build",
    items: [
      { title: "Dashboard", icon: LayoutDashboard, path: "/dashboard" },
      { title: "Playground", icon: FlaskConical, path: "/playground" },
      { title: "Runs", icon: Play, path: "/runs" },
      { title: "Agent nodes", icon: Server, path: "/agents" },
      {
        title: "Triggers",
        icon: Webhook,
        children: [
          { title: "Integrations", icon: Plug, path: "/integrations" },
          { title: "Active", icon: Activity, path: "/triggers" },
        ],
      },
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
