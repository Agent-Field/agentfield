import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import {
  Bell,
  CheckCheck,
  ChevronDown,
  ChevronRight,
  Trash2,
  X,
} from "lucide-react";

import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
// NOTE: we intentionally do NOT use <Badge variant="destructive"> for the
// unread count — that variant auto-injects an XCircle icon (see badge.tsx
// statusIcons map) and inherits px-2 py-0.5 + gap-1.5 padding that makes a
// tiny count pill look enormous and broken. The unread indicator below is a
// plain styled span using the destructive color tokens directly. The
// `secondary` Badge is still safe to use inside the popover header.
import { Separator } from "@/components/ui/separator";
import {
  getNotificationGlyph,
  useNotifications,
  type Notification,
} from "@/components/ui/notification";

/**
 * Notification bell + popover center.
 *
 * Lives in the sidebar header next to ModeToggle. Always visible, muted at
 * rest, with an unread count badge rendered via the shadcn <Badge>. Clicking
 * opens a Popover with the full persistent notification log.
 */
interface NotificationBellProps {
  className?: string;
}

/** Compact relative timestamp used in the bell popover: "now", "2m", "3h",
 * "4d". Prefers short glyphs over English phrases to save horizontal
 * space in the tight tree layout. */
function formatCompactRelative(createdAt: number, now: number): string {
  const diffSec = Math.max(0, Math.floor((now - createdAt) / 1000));
  if (diffSec < 10) return "now";
  if (diffSec < 60) return `${diffSec}s`;
  const diffMin = Math.floor(diffSec / 60);
  if (diffMin < 60) return `${diffMin}m`;
  const diffHr = Math.floor(diffMin / 60);
  if (diffHr < 24) return `${diffHr}h`;
  const diffDay = Math.floor(diffHr / 24);
  if (diffDay < 7) return `${diffDay}d`;
  const diffWk = Math.floor(diffDay / 7);
  return `${diffWk}w`;
}

function formatAbsoluteTime(createdAt: number): string {
  return new Date(createdAt).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

export function NotificationBell({ className }: NotificationBellProps) {
  const {
    notifications,
    unreadCount,
    markRead,
    markAllRead,
    removeNotification,
    clearAll,
  } = useNotifications();
  const [open, setOpen] = useState(false);
  // Re-render relative timestamps periodically while popover is open.
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    if (!open) return;
    setNow(Date.now());
    const id = window.setInterval(() => setNow(Date.now()), 30_000);
    return () => window.clearInterval(id);
  }, [open]);

  const hasNotifications = notifications.length > 0;
  const hasUnread = unreadCount > 0;
  const badgeLabel = unreadCount > 99 ? "99+" : String(unreadCount);

  /**
   * Flatten the list into an ordered stream of "sections" — each section
   * is either a run cluster (multiple notifications sharing a runId) or
   * a single orphan notification. We preserve the original chronology:
   * newest first, but consecutive entries with the same runId collapse
   * into a shared header.
   */
  const sections = useMemo(() => {
    type Section =
      | {
          kind: "run";
          runId: string;
          runLabel?: string;
          entries: Notification[];
          unread: number;
        }
      | { kind: "single"; entry: Notification };

    const result: Section[] = [];
    for (const n of notifications) {
      if (n.runId) {
        const last = result[result.length - 1];
        if (last && last.kind === "run" && last.runId === n.runId) {
          last.entries.push(n);
          if (!n.read) last.unread += 1;
          continue;
        }
        result.push({
          kind: "run",
          runId: n.runId,
          runLabel: n.runLabel,
          entries: [n],
          unread: n.read ? 0 : 1,
        });
      } else {
        result.push({ kind: "single", entry: n });
      }
    }
    return result;
  }, [notifications]);

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          className={cn(
            "relative size-8 shrink-0 rounded-md text-muted-foreground",
            "hover:bg-accent hover:text-foreground",
            "focus-visible:ring-2 focus-visible:ring-ring",
            className,
          )}
          aria-label={
            hasUnread
              ? `Notifications, ${unreadCount} unread`
              : "Notifications"
          }
        >
          <Bell
            className={cn(
              "size-4 transition-colors",
              hasUnread && "text-foreground",
            )}
            aria-hidden
          />
          {hasUnread ? (
            <span
              className={cn(
                "pointer-events-none absolute right-0.5 top-0.5",
                "flex h-3.5 min-w-[0.875rem] items-center justify-center",
                "rounded-full bg-destructive px-1",
                "text-[9px] font-semibold leading-none tabular-nums text-destructive-foreground",
                "ring-1 ring-background",
              )}
              aria-hidden
            >
              {badgeLabel}
            </span>
          ) : null}
        </Button>
      </PopoverTrigger>
      <PopoverContent
        align="end"
        side="bottom"
        sideOffset={8}
        className="w-[min(92vw,22rem)] p-0"
      >
        <TooltipProvider delayDuration={250}>
        <div className="flex items-center justify-between gap-2 px-3 py-2.5">
          <div className="flex min-w-0 items-center gap-2">
            <h3 className="text-sm font-semibold leading-none text-foreground">
              Notifications
            </h3>
            {hasUnread ? (
              <Badge
                variant="secondary"
                className="h-5 px-1.5 text-[10px] font-medium leading-none"
              >
                {unreadCount} new
              </Badge>
            ) : null}
          </div>
          <Button
            variant="ghost"
            size="sm"
            className="h-7 gap-1.5 px-2 text-xs text-muted-foreground hover:text-foreground disabled:opacity-40"
            onClick={markAllRead}
            disabled={!hasUnread}
            aria-label="Mark all as read"
          >
            <CheckCheck className="size-3.5" aria-hidden />
            Mark all read
          </Button>
        </div>
        <Separator />
        {hasNotifications ? (
          <>
            {/* Plain overflow div — Radix ScrollArea inside a Popover can
                swallow wheel events on some browsers, breaking scroll. */}
            <div className="max-h-[22rem] overflow-y-auto overscroll-contain">
              <ul className="flex flex-col">
                {sections.map((section, sectionIndex) => {
                  const key =
                    section.kind === "run"
                      ? `run-${section.runId}-${sectionIndex}`
                      : `single-${section.entry.id}`;
                  const isLast = sectionIndex === sections.length - 1;
                  return (
                    <li key={key}>
                      {section.kind === "run" ? (
                        <RunSection
                          runId={section.runId}
                          runLabel={section.runLabel}
                          entries={section.entries}
                          unread={section.unread}
                          now={now}
                          onMarkRead={markRead}
                          onRemove={removeNotification}
                          onNavigate={() => setOpen(false)}
                        />
                      ) : (
                        <NotificationRow
                          notification={section.entry}
                          now={now}
                          onMarkRead={() => markRead(section.entry.id)}
                          onRemove={() => removeNotification(section.entry.id)}
                        />
                      )}
                      {!isLast ? <Separator className="opacity-60" /> : null}
                    </li>
                  );
                })}
              </ul>
            </div>
            <Separator />
            <div className="flex items-center justify-between px-3 py-2">
              <span className="text-micro text-muted-foreground">
                {notifications.length}{" "}
                {notifications.length === 1 ? "notification" : "notifications"}
              </span>
              <Button
                variant="ghost"
                size="sm"
                className="h-7 gap-1.5 px-2 text-xs text-muted-foreground hover:text-destructive"
                onClick={clearAll}
                aria-label="Clear all notifications"
              >
                <Trash2 className="size-3.5" aria-hidden />
                Clear all
              </Button>
            </div>
          </>
        ) : (
          <NotificationEmptyState />
        )}
        </TooltipProvider>
      </PopoverContent>
    </Popover>
  );
}

/* ═══════════════════════════════════════════════════════════════
   Compact event row

   One-line summary of a single notification: icon + label + time.
   Hover/focus reveals the message in a Tooltip and the remove × on
   the right. No wrapped paragraphs, no uppercase headers — just a
   status line you can scan at a glance.
   ═══════════════════════════════════════════════════════════════ */

interface NotificationRowProps {
  notification: Notification;
  now: number;
  onMarkRead: () => void;
  onRemove: () => void;
  /** When true the row renders with a left connector line for the
   * timeline/tree visual used inside an expanded RunSection. */
  indent?: boolean;
}

function NotificationRow({
  notification,
  now,
  onMarkRead,
  onRemove,
  indent = false,
}: NotificationRowProps) {
  const { Icon, iconClass } = getNotificationGlyph(notification);
  const relative = formatCompactRelative(notification.createdAt, now);
  const absolute = formatAbsoluteTime(notification.createdAt);

  const content = (
    <div
      className={cn(
        "group/row relative flex w-full items-center gap-2 px-3 py-1.5 text-left transition-colors",
        "hover:bg-accent/60 focus-within:bg-accent/60",
        !notification.read && "bg-accent/20",
        indent && "pl-7",
      )}
      onClick={() => {
        if (!notification.read) onMarkRead();
      }}
    >
      <Icon className={cn("size-3.5 shrink-0", iconClass)} aria-hidden />
      <span
        className={cn(
          "min-w-0 flex-1 truncate text-xs leading-none",
          notification.read
            ? "font-normal text-foreground/85"
            : "font-medium text-foreground",
        )}
      >
        {notification.title}
      </span>
      <span
        className="shrink-0 text-[11px] tabular-nums text-muted-foreground"
        title={absolute}
      >
        {relative}
      </span>
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          onRemove();
        }}
        className={cn(
          "flex size-4 shrink-0 items-center justify-center rounded-sm text-muted-foreground/60 opacity-0 transition-opacity",
          "hover:bg-muted hover:text-foreground",
          "group-hover/row:opacity-100 focus-visible:opacity-100",
        )}
        aria-label="Dismiss notification"
      >
        <X className="size-3" aria-hidden />
      </button>
    </div>
  );

  // Only wrap in tooltip if we have a message to show — avoid empty bubbles.
  if (!notification.message) return content;

  return (
    <Tooltip delayDuration={250}>
      <TooltipTrigger asChild>{content}</TooltipTrigger>
      <TooltipContent
        side="left"
        align="center"
        className="max-w-[18rem] text-xs leading-snug"
      >
        {notification.message}
      </TooltipContent>
    </Tooltip>
  );
}

/* ═══════════════════════════════════════════════════════════════
   RunSection — collapsed-by-default run timeline

   Default: one line header (run label + latest event time) + one
   line latest-event summary ("Cancelled · 3 events"). ~44px.
   Expanded: header with chevron-down + vertical timeline of every
   event in the group, each ~28px.
   ═══════════════════════════════════════════════════════════════ */

interface RunSectionProps {
  runId: string;
  runLabel?: string;
  entries: Notification[];
  unread: number;
  now: number;
  onMarkRead: (id: string) => void;
  onRemove: (id: string) => void;
  onNavigate: () => void;
}

function RunSection({
  runId,
  runLabel,
  entries,
  unread,
  now,
  onMarkRead,
  onRemove,
  onNavigate,
}: RunSectionProps) {
  const [expanded, setExpanded] = useState(false);
  // Entries come in newest-first from the provider; latest is index 0.
  const latest = entries[0]!;
  const { Icon: LatestIcon, iconClass: latestIconClass } =
    getNotificationGlyph(latest);
  const latestRelative = formatCompactRelative(latest.createdAt, now);
  const shortId = runId.length > 10 ? `…${runId.slice(-6)}` : runId;
  const headerLabel = runLabel ?? shortId;
  const hasMultiple = entries.length > 1;

  return (
    <div className={cn(unread > 0 && "bg-accent/15")}>
      {/* Run header — single line, reasoner + chevron + time */}
      <div className="flex items-center gap-1.5 px-3 pt-2">
        {hasMultiple ? (
          <button
            type="button"
            onClick={() => setExpanded((v) => !v)}
            className="flex size-4 shrink-0 items-center justify-center rounded-sm text-muted-foreground/70 hover:bg-muted hover:text-foreground"
            aria-label={expanded ? "Collapse run events" : "Expand run events"}
          >
            {expanded ? (
              <ChevronDown className="size-3" aria-hidden />
            ) : (
              <ChevronRight className="size-3" aria-hidden />
            )}
          </button>
        ) : (
          <span className="size-4 shrink-0" aria-hidden />
        )}
        <Link
          to={`/runs/${runId}`}
          onClick={onNavigate}
          className="min-w-0 flex-1 truncate font-mono text-[11px] text-muted-foreground hover:text-foreground"
          title={runLabel ? `${runLabel} · ${runId}` : runId}
        >
          {headerLabel}
        </Link>
        {unread > 0 ? (
          <span
            className="size-1.5 shrink-0 rounded-full bg-sky-500"
            aria-label={`${unread} unread`}
          />
        ) : null}
        <span className="shrink-0 text-[11px] tabular-nums text-muted-foreground">
          {latestRelative}
        </span>
      </div>

      {/* Collapsed body: one-line latest-event summary */}
      {!expanded ? (
        <Tooltip delayDuration={250}>
          <TooltipTrigger asChild>
            <div
              role="button"
              tabIndex={0}
              onClick={() => {
                if (!latest.read) onMarkRead(latest.id);
                if (hasMultiple) setExpanded(true);
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  if (!latest.read) onMarkRead(latest.id);
                  if (hasMultiple) setExpanded(true);
                }
              }}
              className={cn(
                "group/latest mt-0.5 flex items-center gap-2 px-3 pb-2 pl-7 transition-colors",
                "hover:bg-accent/40",
              )}
            >
              <LatestIcon
                className={cn("size-3.5 shrink-0", latestIconClass)}
                aria-hidden
              />
              <span
                className={cn(
                  "min-w-0 flex-1 truncate text-xs leading-none",
                  latest.read
                    ? "font-normal text-foreground/85"
                    : "font-medium text-foreground",
                )}
              >
                {latest.title}
              </span>
              {hasMultiple ? (
                <span className="shrink-0 text-[10px] text-muted-foreground/70">
                  · {entries.length} events
                </span>
              ) : null}
            </div>
          </TooltipTrigger>
          {latest.message ? (
            <TooltipContent
              side="left"
              align="center"
              className="max-w-[18rem] text-xs leading-snug"
            >
              {latest.message}
            </TooltipContent>
          ) : null}
        </Tooltip>
      ) : (
        /* Expanded body: timeline of every event, tree-connected with a
           left border. Each row hover-reveals full message via Tooltip. */
        <div className="relative mt-0.5 pb-1">
          <div
            className="absolute bottom-2 left-[1.375rem] top-0 w-px bg-border/60"
            aria-hidden
          />
          <ul className="flex flex-col">
            {entries.map((entry) => (
              <li key={entry.id}>
                <NotificationRow
                  notification={entry}
                  now={now}
                  onMarkRead={() => onMarkRead(entry.id)}
                  onRemove={() => onRemove(entry.id)}
                  indent
                />
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function NotificationEmptyState() {
  return (
    <div className="flex flex-col items-center justify-center gap-1.5 px-4 py-8 text-center">
      <Bell className="size-6 text-muted-foreground/40" aria-hidden />
      <p className="text-xs font-medium text-muted-foreground">
        No notifications yet
      </p>
      <p className="text-[11px] text-muted-foreground/70">
        You&rsquo;ll see run events, errors, and actions here.
      </p>
    </div>
  );
}
