import { useState } from "react";
import type { FormEvent } from "react";
import { useAuth } from "../contexts/AuthContext";
import { HintIcon } from "@/components/authorization/HintIcon";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { TooltipProvider } from "@/components/ui/tooltip";

/**
 * Inline prompt shown on admin pages for managing the admin token.
 * Always visible: shows a form when no token is set, or a compact
 * status bar with change/clear actions when a token is active.
 */
export function AdminTokenPrompt({ onTokenSet }: { onTokenSet?: () => void }) {
  const { adminToken, setAdminToken } = useAuth();
  const [inputToken, setInputToken] = useState("");
  const [editing, setEditing] = useState(false);

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (!inputToken.trim()) return;
    setAdminToken(inputToken.trim());
    setInputToken("");
    setEditing(false);
    onTokenSet?.();
  };

  const handleClear = () => {
    setAdminToken(null);
    setInputToken("");
    setEditing(false);
  };

  // Token is set — show compact status with change/clear actions
  if (adminToken && !editing) {
    return (
      <TooltipProvider delayDuration={200}>
      <div className="flex items-center gap-3 text-sm text-muted-foreground px-1 py-1.5">
        <span className="inline-flex items-center gap-1">
          <span className="h-2 w-2 rounded-full bg-green-500" />
          Admin token saved in this browser
          <HintIcon label="About the admin token">
            Matches server <code className="font-mono">admin_token</code> or env. Not Settings.
            Unchanged repo default is often <code className="font-mono">admin-secret</code>.
          </HintIcon>
        </span>
        <Button variant="ghost" size="sm" className="h-6 px-2 text-xs" onClick={() => setEditing(true)}>
          Change
        </Button>
        <Button variant="ghost" size="sm" className="h-6 px-2 text-xs text-muted-foreground" onClick={handleClear}>
          Clear
        </Button>
      </div>
      </TooltipProvider>
    );
  }

  // No token or editing — show the input form
  return (
    <TooltipProvider delayDuration={200}>
    <Alert className="border-amber-500/30 bg-amber-500/5">
      <AlertDescription className="space-y-3">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
          <span className="inline-flex items-center gap-1 text-sm font-medium text-amber-700 dark:text-amber-500 shrink-0">
            Admin token
            <HintIcon label="About the admin token">
              Must match server <code className="font-mono">admin_token</code> or{" "}
              <code className="font-mono">AGENTFIELD_AUTHORIZATION_ADMIN_TOKEN</code>. This browser only.
              Default YAML is often <code className="font-mono">admin-secret</code>.
            </HintIcon>
          </span>
          <form onSubmit={handleSubmit} className="flex flex-1 flex-wrap items-center gap-2">
            <Input
              type="password"
              value={inputToken}
              onChange={(e) => setInputToken(e.target.value)}
              placeholder="Same value as on the server"
              className="h-8 max-w-xs"
              autoFocus={editing}
            />
            <Button type="submit" size="sm" className="h-8" disabled={!inputToken.trim()}>
              Save in browser
            </Button>
            {editing && (
              <Button type="button" variant="ghost" size="sm" className="h-8" onClick={() => setEditing(false)}>
                Cancel
              </Button>
            )}
          </form>
        </div>
      </AlertDescription>
    </Alert>
    </TooltipProvider>
  );
}
