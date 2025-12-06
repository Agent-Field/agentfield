import { useEffect, useState } from "react";
import type { FormEvent } from "react";
import { useAuth } from "../contexts/AuthContext";
import { setGlobalApiKey } from "../services/api";

export function AuthGuard({ children }: { children: React.ReactNode }) {
  const { apiKey, setApiKey, isAuthenticated, authRequired } = useAuth();
  const [inputKey, setInputKey] = useState("");
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
      } else {
        setError("Invalid API key");
      }
    } catch {
      setError("Connection failed");
    } finally {
      setValidating(false);
    }
  };

  if (!authRequired || isAuthenticated) {
    return <>{children}</>;
  }

  return (
    <div className="flex items-center justify-center min-h-screen bg-background">
      <form onSubmit={handleSubmit} className="p-8 bg-card rounded-lg shadow-lg max-w-md w-full">
        <h2 className="text-2xl font-semibold mb-2">AgentField Control Plane</h2>
        <p className="text-muted-foreground mb-6">Enter your API key to continue</p>

        <input
          type="password"
          value={inputKey}
          onChange={(e) => setInputKey(e.target.value)}
          placeholder="API Key"
          className="w-full p-3 border rounded-md mb-4 bg-background"
          disabled={validating}
          autoFocus
        />

        {error && <p className="text-destructive mb-4">{error}</p>}

        <button
          type="submit"
          className="w-full bg-primary text-primary-foreground p-3 rounded-md font-medium disabled:opacity-50"
          disabled={validating || !inputKey}
        >
          {validating ? "Validating..." : "Connect"}
        </button>
      </form>
    </div>
  );
}
