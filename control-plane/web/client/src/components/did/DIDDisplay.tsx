import { useState } from "react";
import { useDIDInfo } from "../../hooks/useDIDInfo";
import { copyDIDToClipboard } from "../../services/didApi";
import {
  Copy,
  CheckmarkFilled,
  Identification,
  Security,
  Cognitive,
  Flash
} from "@/components/ui/icon-bridge";

interface DIDDisplayProps {
  nodeId: string;
  variant?: "compact" | "full" | "inline";
  className?: string;
}

/**
 * Beautiful DID display component with world-class dark mode design
 * Inspired by Linear's premium interface aesthetics
 */
export function DIDDisplay({
  nodeId,
  variant = "compact",
  className = "",
}: DIDDisplayProps) {
  const { didInfo, loading } = useDIDInfo(nodeId);
  const [copied, setCopied] = useState(false);

  const handleCopyDID = async () => {
    if (!didInfo?.did) return;

    const success = await copyDIDToClipboard(didInfo.did);
    if (success) {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  if (loading || !didInfo?.did) {
    return null;
  }

  // Truncate DID for display
  const truncateDID = (did: string, length: number = 12) => {
    if (did.length <= length + 6) return did;
    return `${did.slice(0, length)}...${did.slice(-6)}`;
  };

  // Inline variant - minimal, for use within text
  if (variant === "inline") {
    return (
      <span className={`inline-flex items-center gap-1.5 ${className}`}>
        <span className="inline-flex items-center gap-1 text-body-small">
          <Identification size={12} className="text-accent-primary" />
          <span className="font-mono">{truncateDID(didInfo.did, 8)}</span>
        </span>
        <button
          onClick={handleCopyDID}
          className="inline-flex items-center justify-center w-4 h-4 rounded-sm hover:bg-muted transition-colors duration-150"
          title="Copy DID"
        >
          {copied ? (
            <CheckmarkFilled size={10} className="text-status-success" />
          ) : (
            <Copy size={10} className="text-muted-foreground hover:text-foreground" />
          )}
        </button>
      </span>
    );
  }

  // Compact variant - elegant card-like display
  if (variant === "compact") {
    return (
      <div className={`group relative ${className}`}>
        <div className="flex items-center gap-3 px-4 py-3 bg-card hover:bg-card-hover rounded-lg border border-card-border transition-all duration-200 shadow-sm hover:shadow-md">
          <div className="flex items-center gap-2 flex-1 min-w-0">
            <div className="flex items-center justify-center w-6 h-6 rounded-md bg-accent-primary/10 border border-accent-primary/20">
              <Identification size={14} className="text-accent-primary" />
            </div>
            <span className="text-sm font-mono text-foreground truncate">
              {truncateDID(didInfo.did, 16)}
            </span>
          </div>
          <button
            onClick={handleCopyDID}
            className="flex items-center justify-center w-8 h-8 rounded-md hover:bg-muted transition-colors duration-150 opacity-0 group-hover:opacity-100"
            title="Copy DID"
          >
            {copied ? (
              <CheckmarkFilled size={16} className="text-status-success" />
            ) : (
              <Copy size={16} className="text-muted-foreground hover:text-foreground" />
            )}
          </button>
        </div>

        {/* Enhanced tooltip with full DID on hover */}
        <div className="absolute bottom-full left-0 mb-3 px-3 py-2 bg-popover text-popover-foreground text-xs rounded-lg border border-border opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none z-50 max-w-xs break-all shadow-lg">
          <div className="font-medium mb-1">Full DID</div>
          <div className="font-mono text-muted-foreground">{didInfo.did}</div>
        </div>
      </div>
    );
  }

  // Full variant - detailed display with metadata
  const handleCopyWebDID = async () => {
    if (!didInfo?.did_web) return;

    const success = await copyDIDToClipboard(didInfo.did_web);
    if (success) {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <div className={`bg-card border border-card-border rounded-xl p-6 shadow-sm hover:shadow-md transition-shadow duration-200 ${className}`}>
      <div className="flex items-center justify-between mb-5">
        <div className="flex items-center gap-3">
          <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-accent-primary/10 border border-accent-primary/20">
            <Security size={20} className="text-accent-primary" />
          </div>
          <div>
            <h3 className="text-base font-semibold text-foreground">
              Decentralized Identity
            </h3>
          </div>
        </div>
      </div>

      <div className="space-y-4">
        {/* Cryptographic Identity (did:key) */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <div>
              <div className="text-sm font-medium text-foreground">Cryptographic Identity</div>
              <div className="text-xs text-muted-foreground">did:key — signing &amp; authentication</div>
            </div>
            <button
              onClick={handleCopyDID}
              className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs text-muted-foreground hover:text-foreground hover:bg-muted rounded-md transition-colors duration-150"
            >
              {copied ? (
                <>
                  <CheckmarkFilled size={12} className="text-status-success" />
                  <span className="text-status-success">Copied</span>
                </>
              ) : (
                <>
                  <Copy size={12} />
                  <span>Copy</span>
                </>
              )}
            </button>
          </div>
          <div className="p-3 bg-muted rounded-lg border border-border">
            <div className="font-mono text-sm text-foreground break-all leading-relaxed">
              {didInfo.did}
            </div>
          </div>
        </div>

        {/* Web Identity (did:web) */}
        {didInfo.did_web && (
          <div>
            <div className="flex items-center justify-between mb-2">
              <div>
                <div className="text-sm font-medium text-foreground">Web Identity</div>
                <div className="text-xs text-muted-foreground">did:web — JWT &amp; external integrations</div>
              </div>
              <button
                onClick={handleCopyWebDID}
                className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs text-muted-foreground hover:text-foreground hover:bg-muted rounded-md transition-colors duration-150"
              >
                <Copy size={12} />
                <span>Copy</span>
              </button>
            </div>
            <div className="p-3 bg-muted rounded-lg border border-border">
              <div className="font-mono text-sm text-foreground break-all leading-relaxed">
                {didInfo.did_web}
              </div>
            </div>
          </div>
        )}

        <div className="flex items-center gap-6 text-sm pt-1">
          <div className="flex items-center gap-2 text-muted-foreground">
            <div className="flex items-center justify-center w-5 h-5 rounded bg-muted border border-border">
              <Cognitive size={12} className="text-muted-foreground" />
            </div>
            <span>{Object.keys(didInfo.reasoners).length} reasoners</span>
          </div>
          <div className="flex items-center gap-2 text-muted-foreground">
            <div className="flex items-center justify-center w-5 h-5 rounded bg-muted border border-border">
              <Flash size={12} className="text-muted-foreground" />
            </div>
            <span>{Object.keys(didInfo.skills).length} skills</span>
          </div>
          <div className="flex items-center gap-2 text-muted-foreground">
            <span>Registered {new Date(didInfo.registered_at).toLocaleDateString()}</span>
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * Simple DID badge for showing identity status with DID preview
 */
interface DIDIdentityBadgeProps {
  nodeId: string;
  showDID?: boolean;
  className?: string;
}

export function DIDIdentityBadge({
  nodeId,
  showDID = true,
  className = "",
}: DIDIdentityBadgeProps) {
  const { didInfo, loading } = useDIDInfo(nodeId);
  const [copied, setCopied] = useState(false);

  const handleCopyDID = async (e: React.MouseEvent) => {
    e.stopPropagation();
    if (!didInfo?.did) return;

    const success = await copyDIDToClipboard(didInfo.did);
    if (success) {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  if (loading || !didInfo?.did) {
    return null;
  }

  const truncateDID = (did: string) => {
    return `${did.slice(0, 8)}...${did.slice(-4)}`;
  };

  return (
    <div className={`inline-flex items-center gap-3 ${className}`}>
      <div className="flex items-center gap-2">
        <div className="flex items-center justify-center w-4 h-4 rounded-sm bg-status-success/10">
          <Security size={10} className="text-status-success" />
        </div>
        <span className="text-xs font-medium text-status-success">Verified</span>
      </div>

      {showDID && (
        <div className="flex items-center gap-2 px-3 py-1.5 bg-muted hover:bg-card-hover rounded-md text-xs transition-colors duration-150 group border border-border">
          <Identification size={12} className="text-muted-foreground" />
          <span className="font-mono text-foreground">
            {truncateDID(didInfo.did)}
          </span>
          <button
            onClick={handleCopyDID}
            className="opacity-0 group-hover:opacity-100 transition-opacity duration-150"
            title="Copy DID"
          >
            {copied ? (
              <CheckmarkFilled size={12} className="text-status-success" />
            ) : (
              <Copy size={12} className="text-muted-foreground hover:text-foreground" />
            )}
          </button>
        </div>
      )}
    </div>
  );
}
