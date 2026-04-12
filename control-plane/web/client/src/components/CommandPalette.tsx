import { useEffect, useState, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import {
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandSeparator,
} from "@/components/ui/command";
import {
  LayoutDashboard,
  Play,
  Server,
  FlaskConical,
  KeyRound,
  FileCheck2,
  Settings,
  Search,
} from "lucide-react";

// Navigation jumper — this palette does not search runs/workflows/nodes.
// The backing search API was never implemented; treat Cmd-K as Cmd-P.
export function CommandPalette() {
  const [open, setOpen] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        setOpen((o) => !o);
      }
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, []);

  const runAction = useCallback((action: () => void) => {
    setOpen(false);
    action();
  }, []);

  return (
    <CommandDialog open={open} onOpenChange={setOpen}>
      <CommandInput placeholder="Jump to page..." />
      <CommandList>
        <CommandEmpty>No results found.</CommandEmpty>

        <CommandGroup heading="Navigation">
          <CommandItem
            onSelect={() => runAction(() => navigate("/dashboard"))}
          >
            <LayoutDashboard className="mr-2 size-4" />
            Dashboard
          </CommandItem>
          <CommandItem onSelect={() => runAction(() => navigate("/runs"))}>
            <Play className="mr-2 size-4" />
            Runs
          </CommandItem>
          <CommandItem onSelect={() => runAction(() => navigate("/agents"))}>
            <Server className="mr-2 size-4" />
            Agent nodes
          </CommandItem>
          <CommandItem
            onSelect={() => runAction(() => navigate("/playground"))}
          >
            <FlaskConical className="mr-2 size-4" />
            Playground
          </CommandItem>
          <CommandItem
            onSelect={() => runAction(() => navigate("/access"))}
          >
            <KeyRound className="mr-2 size-4" />
            Access management
          </CommandItem>
          <CommandItem
            onSelect={() => runAction(() => navigate("/verify"))}
          >
            <FileCheck2 className="mr-2 size-4" />
            Audit
          </CommandItem>
          <CommandItem
            onSelect={() => runAction(() => navigate("/settings"))}
          >
            <Settings className="mr-2 size-4" />
            Settings
          </CommandItem>
        </CommandGroup>

        <CommandSeparator />

        <CommandGroup heading="Actions">
          <CommandItem
            onSelect={() =>
              runAction(() => navigate("/runs?status=failed"))
            }
          >
            <Search className="mr-2 size-4" />
            Show failed runs
          </CommandItem>
          <CommandItem
            onSelect={() =>
              runAction(() => navigate("/runs?status=running"))
            }
          >
            <Search className="mr-2 size-4" />
            Show active runs
          </CommandItem>
          <CommandItem
            onSelect={() => runAction(() => navigate("/settings"))}
          >
            <Settings className="mr-2 size-4" />
            Configure webhooks
          </CommandItem>
        </CommandGroup>
      </CommandList>
    </CommandDialog>
  );
}
