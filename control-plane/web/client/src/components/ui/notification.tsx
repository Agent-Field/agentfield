import type { ReactNode } from "react";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import {
  AlertTriangle,
  Ban,
  CheckCircle2,
  Info,
  PauseCircle,
  PlayCircle,
  Sparkles,
  type LucideIcon,
} from "lucide-react";
import { Toaster as SonnerToaster, toast as sonnerToast } from "sonner";
import { useTheme } from "next-themes";

import { statusTone } from "@/lib/theme";
import { cn } from "@/lib/utils";

/* ═══════════════════════════════════════════════════════════════
   Types
   ═══════════════════════════════════════════════════════════════ */

export type NotificationType = "success" | "error" | "warning" | "info";

/**
 * Narrower semantic for *what* happened. This drives the icon + tone
 * independently of the sonner `type` (which only knows success/info/etc).
 * A pause is not a "success" even though the API call succeeded — so we
 * pick a pause glyph and neutral/amber tone, not a green checkmark.
 */
export type NotificationEventKind =
  | "pause"
  | "resume"
  | "cancel"
  | "error"
  | "complete"
  | "start"
  | "info";

export interface Notification {
  id: string;
  type: NotificationType;
  /** What happened — overrides the type for icon/tone selection. */
  eventKind?: NotificationEventKind;
  title: string;
  message?: string;
  /** Toast auto-dismiss duration in ms. Set to 0 to keep the toast until clicked. */
  duration?: number;
  action?: {
    label: string;
    onClick: () => void;
  };
  /** If true, the toast will not auto-dismiss. Log entry is always persistent regardless. */
  persistent?: boolean;
  /** ms epoch, set automatically on creation. */
  createdAt: number;
  /** Whether the user has seen / acknowledged this in the bell popover. */
  read: boolean;
  /**
   * Optional run context. When set, the bell popover groups related
   * notifications under a single run header and the toast/row can render
   * a "View run" link automatically.
   */
  runId?: string;
  /** Display label for the run — e.g. `sec-af.audit` or `parallel_pipeline`. */
  runLabel?: string;
  /** Path to navigate to when the user clicks the notification or action. */
  href?: string;
}

interface NotificationContextType {
  /** All notifications, newest first. Acts as the persistent log. */
  notifications: Notification[];
  /** Count of unread notifications across the log. */
  unreadCount: number;
  addNotification: (
    notification: Omit<Notification, "id" | "createdAt" | "read">,
  ) => string;
  markRead: (id: string) => void;
  markAllRead: () => void;
  /** Removes the entry from the log and dismisses any visible toast. */
  removeNotification: (id: string) => void;
  /** Clear the entire log. Also dismisses any active toasts. */
  clearAll: () => void;
}

const NotificationContext = createContext<NotificationContextType | undefined>(
  undefined,
);

/** Keep the log bounded — older entries roll off the end. */
const MAX_LOG_SIZE = 50;

export function useNotifications() {
  const context = useContext(NotificationContext);
  if (!context) {
    throw new Error(
      "useNotifications must be used within a NotificationProvider",
    );
  }
  return context;
}

/* ═══════════════════════════════════════════════════════════════
   Provider
   ═══════════════════════════════════════════════════════════════ */

interface NotificationProviderProps {
  children: ReactNode;
}

export function NotificationProvider({ children }: NotificationProviderProps) {
  const { resolvedTheme } = useTheme();
  const [notifications, setNotifications] = useState<Notification[]>([]);

  const removeNotification = useCallback((id: string) => {
    setNotifications((prev) => prev.filter((n) => n.id !== id));
    sonnerToast.dismiss(id);
  }, []);

  const addNotification: NotificationContextType["addNotification"] =
    useCallback(
      (notification) => {
        const id =
          typeof crypto !== "undefined" && "randomUUID" in crypto
            ? crypto.randomUUID()
            : Math.random().toString(36).slice(2, 11);

        const entry: Notification = {
          ...notification,
          id,
          createdAt: Date.now(),
          read: false,
          duration: notification.duration ?? 5000,
        };

        // 1. Persist into the bell log (capped at MAX_LOG_SIZE).
        setNotifications((prev) => [entry, ...prev].slice(0, MAX_LOG_SIZE));

        // 2. Surface as a transient sonner toast — this is the source of
        //    truth for the live visual; the bell shows history.
        const toastFn =
          entry.type === "success"
            ? sonnerToast.success
            : entry.type === "error"
              ? sonnerToast.error
              : entry.type === "warning"
                ? sonnerToast.warning
                : sonnerToast.info;

        // If the entry has a runId but no explicit action, synthesize a
        // "View run" button pointing at the run detail page. Builds an
        // absolute path using the Vite base path so basename-aware routing
        // stays intact regardless of where the toast fires from.
        const synthAction =
          entry.action ??
          (entry.runId
            ? {
                label: "View run",
                onClick: () => {
                  const base =
                    (import.meta.env.VITE_BASE_PATH as string | undefined) ??
                    "/ui";
                  const trimmed = base.endsWith("/") ? base.slice(0, -1) : base;
                  window.location.assign(`${trimmed}/runs/${entry.runId}`);
                },
              }
            : undefined);

        // Pick the correct glyph for pause/resume/cancel/etc so the toast
        // icon matches the bell popover. Use sonner's generic toast() with
        // an `icon` option rather than type-specific calls — that way the
        // left border still gets set via classNames but the glyph reflects
        // the actual event instead of sonner's default checkmark.
        const { Icon: EventIcon, iconClass: eventIconClass } =
          getNotificationGlyph(entry);
        // React element for sonner's icon slot. We intentionally do not
        // import React.createElement here (already in scope via React 19
        // JSX transform).
        const iconElement = (
          <EventIcon className={`size-4 ${eventIconClass}`} aria-hidden />
        );

        toastFn(entry.title, {
          id,
          description: entry.message,
          duration: entry.persistent ? Infinity : entry.duration,
          icon: iconElement,
          action: synthAction
            ? {
                label: synthAction.label,
                onClick: synthAction.onClick,
              }
            : undefined,
        });

        return id;
      },
      [],
    );

  const markRead = useCallback((id: string) => {
    setNotifications((prev) =>
      prev.map((n) => (n.id === id ? { ...n, read: true } : n)),
    );
  }, []);

  const markAllRead = useCallback(() => {
    setNotifications((prev) =>
      prev.every((n) => n.read) ? prev : prev.map((n) => ({ ...n, read: true })),
    );
  }, []);

  const clearAll = useCallback(() => {
    setNotifications([]);
    sonnerToast.dismiss();
  }, []);

  const unreadCount = useMemo(
    () => notifications.reduce((count, n) => count + (n.read ? 0 : 1), 0),
    [notifications],
  );

  // Reflect unread count in the browser tab title so the user sees "(3)"
  // in the Chrome tab when a notification comes in while they're on
  // another tab. The base title is captured once on mount so we don't
  // accumulate "(N) (N) Title" if this effect runs repeatedly.
  useEffect(() => {
    const base = document.title.replace(/^\(\d+\)\s*/, "");
    document.title = unreadCount > 0 ? `(${unreadCount}) ${base}` : base;
  }, [unreadCount]);

  const contextValue = useMemo<NotificationContextType>(
    () => ({
      notifications,
      unreadCount,
      addNotification,
      markRead,
      markAllRead,
      removeNotification,
      clearAll,
    }),
    [
      notifications,
      unreadCount,
      addNotification,
      markRead,
      markAllRead,
      removeNotification,
      clearAll,
    ],
  );

  // Sonner reads its theme from the next-themes resolved value so the
  // toast colors match the rest of the app's dark/light mode.
  const sonnerTheme = resolvedTheme === "light" ? "light" : "dark";

  return (
    <NotificationContext.Provider value={contextValue}>
      {children}
      <SonnerToaster
        theme={sonnerTheme}
        position="bottom-right"
        closeButton
        visibleToasts={4}
        toastOptions={{
          // Neutral card background everywhere — only the type icon carries
          // colour. This matches the shadcn philosophy: the surface stays
          // quiet, the icon does the signalling. A thin left border picks
          // up the tone so the type is readable at a glance without
          // flooding the card.
          unstyled: false,
          classNames: {
            toast: cn(
              "group toast !bg-card !text-foreground !border !border-border !shadow-lg",
              "gap-3 rounded-md border-l-4",
            ),
            title: "text-sm font-medium leading-snug text-foreground",
            description: "text-xs leading-snug text-muted-foreground",
            icon: "size-4",
            success:
              "!border-l-status-success/70 [&>[data-icon]]:text-status-success",
            error:
              "!border-l-status-error/80 [&>[data-icon]]:text-status-error",
            warning:
              "!border-l-status-warning/70 [&>[data-icon]]:text-status-warning",
            info: "!border-l-status-info/70 [&>[data-icon]]:text-status-info",
            closeButton:
              "!bg-transparent !border-border/60 !text-muted-foreground hover:!bg-muted hover:!text-foreground",
            actionButton:
              "!bg-primary !text-primary-foreground hover:!bg-primary/90",
            cancelButton:
              "!bg-muted !text-muted-foreground hover:!bg-muted/80",
          },
        }}
      />
    </NotificationContext.Provider>
  );
}

/* ═══════════════════════════════════════════════════════════════
   Glyph + accent helpers — single source of truth

   Notifications are described in two dimensions:
     - `eventKind` (pause, resume, cancel, error, complete, start, info)
       picks the GLYPH and accent tone.
     - `type` (success, error, warning, info) is the fallback for any
       notification without an explicit eventKind, and is what sonner's
       built-in styling keys off.
   ═══════════════════════════════════════════════════════════════ */

const EVENT_ICON: Record<NotificationEventKind, LucideIcon> = {
  pause: PauseCircle,
  resume: PlayCircle,
  cancel: Ban,
  error: AlertTriangle,
  complete: CheckCircle2,
  start: Sparkles,
  info: Info,
};

const EVENT_ACCENT: Record<NotificationEventKind, { icon: string }> = {
  pause: { icon: statusTone.warning.accent },
  resume: { icon: statusTone.success.accent },
  cancel: { icon: statusTone.neutral.accent },
  error: { icon: statusTone.error.accent },
  complete: { icon: statusTone.success.accent },
  start: { icon: statusTone.info.accent },
  info: { icon: statusTone.info.accent },
};

const TYPE_FALLBACK_KIND: Record<NotificationType, NotificationEventKind> = {
  success: "complete",
  error: "error",
  warning: "info",
  info: "info",
};

export function resolveEventKind(
  notification: Pick<Notification, "eventKind" | "type">,
): NotificationEventKind {
  return notification.eventKind ?? TYPE_FALLBACK_KIND[notification.type];
}

export function getNotificationGlyph(
  notification: Pick<Notification, "eventKind" | "type">,
): { Icon: LucideIcon; iconClass: string } {
  const kind = resolveEventKind(notification);
  return { Icon: EVENT_ICON[kind], iconClass: EVENT_ACCENT[kind].icon };
}

/* ═══════════════════════════════════════════════════════════════
   Convenience hooks — backwards-compatible with existing callers
   ═══════════════════════════════════════════════════════════════ */

export function useSuccessNotification() {
  const { addNotification } = useNotifications();
  return useCallback(
    (title: string, message?: string, action?: Notification["action"]) =>
      addNotification({
        type: "success",
        title,
        message,
        action,
        duration: 4000,
      }),
    [addNotification],
  );
}

export function useErrorNotification() {
  const { addNotification } = useNotifications();
  return useCallback(
    (title: string, message?: string, action?: Notification["action"]) =>
      addNotification({
        type: "error",
        title,
        message,
        action,
        duration: 6000,
      }),
    [addNotification],
  );
}

export function useInfoNotification() {
  const { addNotification } = useNotifications();
  return useCallback(
    (title: string, message?: string, action?: Notification["action"]) =>
      addNotification({
        type: "info",
        title,
        message,
        action,
        duration: 5000,
      }),
    [addNotification],
  );
}

export function useWarningNotification() {
  const { addNotification } = useNotifications();
  return useCallback(
    (title: string, message?: string, action?: Notification["action"]) =>
      addNotification({
        type: "warning",
        title,
        message,
        action,
        duration: 5000,
      }),
    [addNotification],
  );
}

/**
 * Run-scoped notification hook. Emits a notification tagged with a runId
 * so the bell popover can group it under a shared run header. If `href`
 * is omitted it defaults to `/ui/runs/{runId}`, so a "View run" action is
 * automatically synthesized in the toast.
 */
export interface RunNotificationOptions {
  type?: NotificationType;
  /** What happened — drives icon + tone. Prefer this over `type`. */
  eventKind?: NotificationEventKind;
  title: string;
  message?: string;
  runId: string;
  runLabel?: string;
  href?: string;
  persistent?: boolean;
}

export function useRunNotification() {
  const { addNotification } = useNotifications();
  return useCallback(
    (opts: RunNotificationOptions) => {
      const defaultDuration =
        opts.type === "error" ? 6000 : opts.type === "warning" ? 5000 : 4000;
      const baseUrl = import.meta.env.VITE_BASE_PATH || "/ui";
      return addNotification({
        type: opts.type ?? "info",
        eventKind: opts.eventKind,
        title: opts.title,
        message: opts.message,
        runId: opts.runId,
        runLabel: opts.runLabel,
        href: opts.href ?? `${baseUrl}/runs/${opts.runId}`,
        duration: defaultDuration,
        persistent: opts.persistent,
      });
    },
    [addNotification],
  );
}

/* ═══════════════════════════════════════════════════════════════
   DID/VC specific helpers — unchanged public API
   ═══════════════════════════════════════════════════════════════ */

export function useDIDNotifications() {
  const success = useSuccessNotification();
  const error = useErrorNotification();
  const info = useInfoNotification();

  return {
    didCopied: (type: string = "DID") =>
      success(`${type} Copied`, `${type} has been copied to clipboard`),

    didRegistered: (nodeId: string) =>
      success("DID Registered", `DID identity registered for node ${nodeId}`),

    didError: (message: string) => error("DID Operation Failed", message),

    didRefreshed: () =>
      info("DID Data Refreshed", "DID information has been updated"),
  };
}

export function useVCNotifications() {
  const success = useSuccessNotification();
  const error = useErrorNotification();
  const info = useInfoNotification();

  return {
    vcCopied: () =>
      success("VC Copied", "Verifiable Credential copied to clipboard"),

    vcDownloaded: (filename?: string) =>
      success(
        "VC Downloaded",
        filename ? `Downloaded as ${filename}` : "VC document downloaded",
      ),

    vcVerified: (valid: boolean) =>
      valid
        ? success("VC Verified", "Verifiable Credential is valid and verified")
        : error(
            "VC Verification Failed",
            "Verifiable Credential verification failed",
          ),

    vcError: (message: string) => error("VC Operation Failed", message),

    vcChainLoaded: (count: number) =>
      info("VC Chain Loaded", `Loaded ${count} verification credentials`),
  };
}
