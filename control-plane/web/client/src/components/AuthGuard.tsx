import { useEffect, useState } from "react";
import type { FormEvent } from "react";
import { KeyRound, Loader2, ShieldCheck } from "lucide-react";
import { useAuth } from "../contexts/AuthContext";
import { setGlobalApiKey } from "../services/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import logoLight from "@/assets/logos/logo-short-light-v2.svg";
import logoDark from "@/assets/logos/logo-short-dark-v2.svg";

export function AuthGuard({ children }: { children: React.ReactNode }) {
  const { apiKey, setApiKey, setAdminToken, isAuthenticated, authRequired } = useAuth();
  const [inputKey, setInputKey] = useState("");
  const [inputAdminToken, setInputAdminToken] = useState("");
  const [error, setError] = useState("");
  const [validating, setValidating] = useState(false);

  useEffect(() => {
    setGlobalApiKey(apiKey);
  }, [apiKey]);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError("");
    setValidating(true);

    try {
      const response = await fetch("/api/ui/v1/dashboard/summary", {
        headers: { "X-API-Key": inputKey },
      });

      if (response.ok) {
        setApiKey(inputKey);
        setGlobalApiKey(inputKey);
        if (inputAdminToken.trim()) {
          setAdminToken(inputAdminToken.trim());
        }
      } else {
        setError("Invalid API key. Check the key and try again.");
      }
    } catch {
      setError("Unable to reach the control plane. Is the server running?");
    } finally {
      setValidating(false);
    }
  };

  if (!authRequired || isAuthenticated) {
    return <>{children}</>;
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-background p-4">
      <div className="w-full max-w-sm space-y-6">
        {/* Logo + branding */}
        <div className="flex flex-col items-center gap-3 text-center">
          <span className="relative size-12">
            <img src={logoLight} alt="" width={48} height={48} className="size-12 rounded-xl object-cover dark:hidden" />
            <img src={logoDark} alt="" width={48} height={48} className="hidden size-12 rounded-xl object-cover dark:block" />
          </span>
          <div>
            <h1 className="text-lg font-semibold tracking-tight">AgentField</h1>
            <p className="text-sm text-muted-foreground">Control Plane</p>
          </div>
        </div>

        {/* Sign-in card */}
        <Card variant="surface">
          <CardHeader className="space-y-1 pb-4">
            <CardTitle className="text-base">Authenticate</CardTitle>
            <CardDescription>
              Enter your API key to access the control plane dashboard.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="space-y-2">
                <label htmlFor="api-key" className="text-sm font-medium leading-none">
                  API Key
                </label>
                <div className="relative">
                  <KeyRound className="absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
                  <Input
                    id="api-key"
                    type="password"
                    value={inputKey}
                    onChange={(e) => setInputKey(e.target.value)}
                    placeholder="hax_live_…"
                    className="pl-9"
                    disabled={validating}
                    autoFocus
                  />
                </div>
              </div>

              <div className="space-y-2">
                <label htmlFor="admin-token" className="text-sm font-medium leading-none text-muted-foreground">
                  Admin Token <span className="font-normal">(optional)</span>
                </label>
                <div className="relative">
                  <ShieldCheck className="absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
                  <Input
                    id="admin-token"
                    type="password"
                    value={inputAdminToken}
                    onChange={(e) => setInputAdminToken(e.target.value)}
                    placeholder="For permission management"
                    className="pl-9"
                    disabled={validating}
                  />
                </div>
              </div>

              {error && (
                <Alert variant="destructive" className="py-2">
                  <AlertDescription className="text-sm">{error}</AlertDescription>
                </Alert>
              )}

              <Button type="submit" className="w-full" disabled={validating || !inputKey}>
                {validating ? (
                  <>
                    <Loader2 className="mr-2 size-4 animate-spin" />
                    Validating…
                  </>
                ) : (
                  "Connect"
                )}
              </Button>
            </form>
          </CardContent>
        </Card>

        <p className="text-center text-xs text-muted-foreground">
          Key is stored locally in this browser only.
        </p>
      </div>
    </div>
  );
}
