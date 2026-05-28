import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  SegmentedControl,
  type SegmentedControlOption,
} from "@/components/ui/segmented-control";
import { ChipInput } from "@/components/ui/chip-input";
import { CheckCircle, XCircle, CaretRight } from "@/components/ui/icon-bridge";
import { cn } from "@/lib/utils";
import type {
  AccessPolicy,
  AccessPolicyRequest,
} from "../../services/accessPoliciesApi";
import { listKnownTags } from "../../services/accessPoliciesApi";

const emptyPolicy: AccessPolicyRequest = {
  name: "",
  caller_tags: [],
  target_tags: [],
  allow_functions: [],
  deny_functions: [],
  action: "allow",
  priority: 0,
  description: "",
};

const actionOptions: ReadonlyArray<SegmentedControlOption> = [
  {
    value: "allow",
    label: "Allow",
    icon: ({ className }: { className?: string }) => (
      <CheckCircle size={16} className={cn("text-status-success", className)} />
    ),
  },
  {
    value: "deny",
    label: "Deny",
    icon: ({ className }: { className?: string }) => (
      <XCircle size={16} className={cn("text-status-error", className)} />
    ),
  },
] as const;

interface PolicyFormDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  editPolicy?: AccessPolicy | null;
  onSave: (req: AccessPolicyRequest, editId?: number) => Promise<void>;
}

export function PolicyFormDialog({
  open,
  onOpenChange,
  editPolicy,
  onSave,
}: PolicyFormDialogProps) {
  const [form, setForm] = useState<AccessPolicyRequest>({ ...emptyPolicy });
  const [saving, setSaving] = useState(false);
  const [suggestions, setSuggestions] = useState<string[]>([]);

  // Fetch known tags when dialog opens
  useEffect(() => {
    if (open) {
      listKnownTags()
        .then((data) => setSuggestions(data.tags))
        .catch(() => {});
    }
  }, [open]);

  // Populate form when editing
  useEffect(() => {
    if (editPolicy) {
      setForm({
        name: editPolicy.name,
        caller_tags: [...editPolicy.caller_tags],
        target_tags: [...editPolicy.target_tags],
        allow_functions: [...(editPolicy.allow_functions || [])],
        deny_functions: [...(editPolicy.deny_functions || [])],
        action: editPolicy.action,
        priority: editPolicy.priority,
        description: editPolicy.description || "",
      });
    } else {
      setForm({ ...emptyPolicy });
    }
  }, [editPolicy, open]);

  const handleSave = async () => {
    try {
      setSaving(true);
      await onSave(form, editPolicy?.id);
      onOpenChange(false);
    } finally {
      setSaving(false);
    }
  };

  const functionsLabel =
    form.action === "allow" ? "Allowed Functions" : "Denied Functions";
  const functionsValue =
    form.action === "allow"
      ? form.allow_functions || []
      : form.deny_functions || [];
  const onFunctionsChange = (fns: string[]) => {
    if (form.action === "allow") {
      setForm({ ...form, allow_functions: fns, deny_functions: [] });
    } else {
      setForm({ ...form, deny_functions: fns, allow_functions: [] });
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>
            {editPolicy ? "Edit Policy" : "Create Policy"}
          </DialogTitle>
          <DialogDescription>
            Define a tag-based access policy for cross-agent calls.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          {/* Action segmented control */}
          <div>
            <Label className="mb-2 block">Action</Label>
            <SegmentedControl
              value={form.action}
              onValueChange={(v) =>
                setForm({ ...form, action: v as "allow" | "deny" })
              }
              options={actionOptions}
              className="w-full"
            />
          </div>

          {/* Name */}
          <div>
            <Label>Name</Label>
            <Input
              value={form.name}
              onChange={(e) => setForm({ ...form, name: e.target.value })}
              placeholder="e.g. analytics-read-financial"
            />
          </div>

          {/* Description */}
          <div>
            <Label>Description</Label>
            <Input
              value={form.description || ""}
              onChange={(e) =>
                setForm({ ...form, description: e.target.value })
              }
              placeholder="Optional description"
            />
          </div>

          {/* Two-column: Tags (left) + Functions/Priority (right) */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
            {/* Left column: Caller → Target tags */}
            <div className="space-y-3">
              <div>
                <Label className="mb-1.5 block">Caller Tags</Label>
                <ChipInput
                  value={form.caller_tags}
                  onChange={(tags) =>
                    setForm({ ...form, caller_tags: tags })
                  }
                  suggestions={suggestions}
                  placeholder="e.g. analytics"
                />
              </div>

              <div className="flex justify-center py-1">
                <CaretRight
                  size={20}
                  className="text-muted-foreground rotate-90"
                />
              </div>

              <div>
                <Label className="mb-1.5 block">Target Tags</Label>
                <ChipInput
                  value={form.target_tags}
                  onChange={(tags) =>
                    setForm({ ...form, target_tags: tags })
                  }
                  suggestions={suggestions}
                  placeholder="e.g. financial"
                />
              </div>
            </div>

            {/* Right column: Functions + Priority */}
            <div className="space-y-4">
              <div>
                <Label className="mb-1.5 block">{functionsLabel}</Label>
                <ChipInput
                  value={functionsValue}
                  onChange={onFunctionsChange}
                  placeholder="e.g. get_report, list_*"
                />
                <p className="text-xs text-muted-foreground mt-1">
                  Supports wildcards (*). Leave empty for all functions.
                </p>
              </div>

              <div>
                <Label>Priority</Label>
                <Input
                  type="number"
                  value={form.priority ?? 0}
                  onChange={(e) =>
                    setForm({
                      ...form,
                      priority: parseInt(e.target.value) || 0,
                    })
                  }
                  placeholder="0"
                />
                <p className="text-xs text-muted-foreground mt-1">
                  Higher priority policies are evaluated first.
                </p>
              </div>
            </div>
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={handleSave} disabled={saving || !form.name.trim()}>
            {saving
              ? "Saving..."
              : editPolicy
                ? "Update"
                : "Create"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
