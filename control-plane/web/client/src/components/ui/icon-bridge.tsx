import type { LucideIcon, LucideProps } from "lucide-react";
import {
  Activity,
  AlertCircle,
  AlertTriangle,
  ArrowDown,
  ArrowDownNarrowWide,
  ArrowLeft,
  ArrowRight,
  ArrowUp,
  ArrowUpDown,
  ArrowUpRight,
  BarChart3,
  Bot,
  Braces,
  Bug,
  Calendar,
  CheckCircle2,
  Check,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  ChevronUp,
  Circle,
  CircleDashed,
  Clock,
  Cloud,
  CloudOff,
  Copy,
  Cpu,
  Crosshair,
  Database,
  Download,
  Eye,
  EyeOff,
  FileCode,
  FileText,
  Filter,
  Funnel,
  FunctionSquare,
  Gauge,
  Settings,
  Cog,
  GitBranch,
  GitCommit,
  Github,
  Hash,
  HelpCircle,
  IdCard,
  Info,
  Layers,
  LineChart,
  Link,
  ListChecks,
  List,
  Loader2,
  Lock,
  LogOut,
  Maximize,
  Maximize2,
  MessageSquare,
  Minimize,
  Minimize2,
  Minus,
  Monitor,
  Moon,
  MoreHorizontal,
  MoreVertical,
  Network,
  Package,
  PanelLeft,
  PauseCircle,
  Play,
  Plus,
  Quote,
  Radio,
  Rocket,
  RotateCcw,
  RefreshCw,
  Save,
  Scan,
  Search,
  Share2,
  Shield,
  ShieldCheck,
  SortAsc,
  SortDesc,
  Square,
  Star,
  Sun,
  Table,
  Tag,
  Target,
  Terminal,
  Timer,
  Trash2,
  TrendingDown,
  TrendingUp,
  Upload,
  User,
  UserCircle,
  Users,
  Wifi,
  WifiOff,
  Workflow,
  Wrench,
  X,
  XCircle,
  Zap,
  ArrowUpNarrowWide,
  LayoutGrid,
  HardDrive,
  Lightbulb,
  Server,
} from "lucide-react";

export type IconComponent = LucideIcon;
export type IconComponentProps = LucideProps;
export type IconWeight = "thin" | "light" | "regular" | "bold" | "fill" | "duotone";
export type { LucideIcon as Icon };

// Activity / Pulse / Heartbeat
export { Activity };
export { Activity as Pulse };
export { Activity as Heartbeat };

// Alerts / Warnings
export { AlertTriangle };
export { AlertCircle };
export { AlertTriangle as Warning };
export { AlertCircle as WarningCircle };
export { AlertCircle as WarningAlt };
export { AlertTriangle as WarningDiamond };
export { AlertTriangle as WarningOctagon };
export { AlertCircle as WarningAltFilled };
export { AlertTriangle as WarningFilled };

// Arrows
export { ArrowLeft };
export { ArrowDown };
export { ArrowUp };
export { ArrowUpDown };
export { ArrowRight };
export { ArrowUpRight };
export { RotateCcw as ArrowCounterClockwise };
export { RotateCcw as ArrowSquareOut };
export { Maximize as ArrowsOutSimple };
export { RefreshCw };
export { RefreshCw as ArrowsClockwise };
export { RefreshCw as ArrowClockwise };
export { RefreshCw as Renew };
export { RefreshCw as Restart };
export { RefreshCw as Reset };
export { RotateCcw as RotateCcw };
export { ArrowDown as CollapseAll };
export { ArrowUp as ExpandAll };

// Bot / Robot
export { Bot };

/**
 * Agent nodes — deployable agent processes (treat like microservices).
 * Use for fleet / node rows, nav, combobox node column — not for reasoners or skills.
 */
export { Server as AgentNodeIcon };

// Reasoners — LLM-capable / “thinking” API endpoints (non-deterministic).
export { Lightbulb };
export { Lightbulb as ReasonerIcon };

// Skills — deterministic callable APIs (ƒ); pair visually with ReasonerIcon in lists.

// Calendar / Time
export { Calendar };
export { Calendar as Events };
export { Clock };
export { Clock as Time };
export { Clock as RecentlyViewed };
export { Timer };
export { Timer as ClockClockwise };
export { RotateCcw as ClockCounterClockwise };

// Charts / Analytics
export { LineChart as Analytics };
export { BarChart3 };
export { TrendingUp };
export { TrendingDown };
export { TrendingUp as TrendingUp2 };
export { Gauge };
export { Gauge as GaugeCircle };

// Check / Checkmarks
export { Check };
export { Check as Checkmark };
export { CheckCircle2 as CheckmarkFilled };
export { CheckCircle2 as CheckCircle };
export { CheckCircle2 };
export { CheckCircle2 as CircleCheck };
export { ListChecks as Checklist };

// Chevrons / Carets
export { ChevronDown };
export { ChevronLeft };
export { ChevronRight };
export { ChevronUp };
export { ChevronDown as CaretDown };
export { ChevronRight as CaretRight };
export { ChevronUp as CaretUp };
export { ChevronLeft as CaretLeft };

// Circle variants
export { Circle };
export { CircleDashed as CircleDash };
export { CircleDashed };
export { Circle as CircleFilled };

// Code / Braces
export { Braces as Code };
export { Braces as BracketsCurly };
export { Braces };
export { FileCode as FileJson };

// Cloud
export { Cloud };
export { CloudOff as CloudOffline };

// Copy
export { Copy };
export { Copy as CopySimple };

// Cpu / Chip
export { Cpu };
export { Cpu as Chip };

// Close / X
export { X as Close };
export { XCircle as CloseFilled };
export { X };
export { XCircle };
export { XCircle as CircleX };
export { XCircle as ErrorFilled };

// Database
export { Database };
export { Database as DataBase };

// Dots / More
export { MoreHorizontal as DotsThree };
export { MoreVertical as DotsThreeVertical };
export { MoreHorizontal };

// Download / Upload
export { Download };
export { Download as DownloadSimple };
export { Upload };
export { Upload as UploadSimple };

// Eye / View
export { Eye };
export { Eye as View };
export { EyeOff };

// File / Document
export { FileText };
export { FileText as Document };
export { FileText as Article };
export { FileText as Note };

// Filter / Funnel / Sort
export { Filter };
export { Funnel as FunnelSimple };
export { SortAsc as SortAscending };
export { SortDesc as SortDescending };
export { ArrowUpNarrowWide as SortAscendingIcon };
export { ArrowDownNarrowWide as SortDescendingIcon };

// Flash / Zap / Lightning
export { Zap as Flash };
export { Zap };

// Focus / Crosshair
export { Crosshair as Focus };
export { Target as LocateFixed };
export { Target };
export { Scan };

export { FunctionSquare as Function };
export { FunctionSquare as SkillIcon };

// Git
export { GitBranch };
export { GitBranch as GitBranch2 };
export { GitBranch as TreeStructure };
export { GitCommit };
export { Github as GithubLogo };

// Globe / Earth
export { Network as Earth };
export { Network };
export { Network as Network_3 };
export { Share2 };
export { Share2 as ShareNetwork };

// Hash
export { Hash };

// ID / Identification
export { IdCard as Identification };

// Help / Question
export { HelpCircle };
export { HelpCircle as Question };
export { HelpCircle as QuestionMark };

// Info
export { Info };
export { Info as Information };

// Layers / Stack
export { Layers };
export { Layers as Stack };

// Launch / Rocket
export { Rocket as Launch };
export { Rocket };

// Link / External
export { Link };
export { ArrowUpRight as ExternalLink };

// List
export { List };
export { List as ListBullets };

// Loader / Spinner
export { Loader2 };
export { Loader2 as InProgress };
export { Loader2 as Loader };
export { Loader2 as SpinnerGap };

// Lock
export { Lock };

// Log Out / Sign Out
export { LogOut as SignOut };

// Maximize / Minimize / Corners
export { Maximize };
export { Maximize2 };
export { Minimize };
export { Minimize2 };
export { Minimize as CornersIn };

// MessageSquare / Chat
export { MessageSquare };
export { MessageSquare as Chat };

// Minus
export { Minus };

// Monitor / Desktop
export { Monitor };

// Moon / Sun
export { Moon };
export { Sun };

// Package
export { Package };
export { Package as PackageIcon };

// Panel / Sidebar
export { PanelLeft };

// Pause
export { PauseCircle };
export { PauseCircle as PauseFilled };

// Play
export { Play };

// Plus
export { Plus };

// Quote
export { Quote };

// Radio / Broadcast / RadioTower
export { Radio as RadioTower };
export { Radio as Broadcast };

// Save / Floppy
export { Save };
export { Save as FloppyDisk };

// Search / MagnifyingGlass
export { Search };

// Settings / Gear / Cog
export { Settings };
export { Cog };

// Shield / Security
export { Shield };
export { ShieldCheck };
export { ShieldCheck as Security };

// Sort
export { SortAsc };
export { SortDesc };
export { ArrowUpNarrowWide };
export { ArrowDownNarrowWide };

// Star
export { Star };

// Stop / Square
export { Square as Stop };

// Table
export { Table };

// Tag
export { Tag };

// Terminal
export { Terminal };

// Trash
export { Trash2 as Trash };
export { Trash2 as TrashCan };

// Type / Text
export { Hash as Type };

// User / Users
export { User };
export { Users };
export { UserCircle };

// Wifi
export { Wifi };
export { Wifi as WifiHigh };
export { WifiOff };
export { WifiOff as WifiSlash };

// Workflow / FlowArrow
export { Workflow as FlowArrow };

// Wrench / Bug / Tools
export { Wrench };
export { Bug };
export { Wrench as Tools };

// Layout / Grid / Dashboard
export { LayoutGrid as Dashboard };
export { LayoutGrid as Grid };
export { LayoutGrid as SquaresFour };
export { LayoutGrid as GridFour };

// Hard Drive / Cloud Server
export { HardDrive as ServerProxy };

// ListChecks
export { ListChecks };
