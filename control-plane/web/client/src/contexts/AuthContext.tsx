import { createContext, useContext, useEffect, useState } from "react";
import type { ReactNode } from "react";
import { setGlobalApiKey } from "../services/api";

interface AuthContextType {
  apiKey: string | null;
  setApiKey: (key: string | null) => void;
  isAuthenticated: boolean;
  authRequired: boolean;
  clearAuth: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);
const STORAGE_KEY = "af_api_key";

// Simple obfuscation for localStorage; not meant as real security.
const encryptKey = (key: string): string => btoa(key.split("").reverse().join(""));
const decryptKey = (value: string): string => {
  try {
    return atob(value).split("").reverse().join("");
  } catch {
    return "";
  }
};

// Initialize global API key from localStorage BEFORE any React rendering
// This ensures API calls made during initial render have the key
const initStoredKey = (() => {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      const key = decryptKey(stored);
      if (key) {
        setGlobalApiKey(key);
        return key;
      }
    }
  } catch {
    // localStorage might not be available
  }
  return null;
})();

export function AuthProvider({ children }: { children: ReactNode }) {
  // Initialize with pre-loaded key so it's available immediately
  const [apiKey, setApiKeyState] = useState<string | null>(initStoredKey);
  const [authRequired, setAuthRequired] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const checkAuth = async () => {
      try {
        // Check for stored key first
        const stored = localStorage.getItem(STORAGE_KEY);
        const storedKey = stored ? decryptKey(stored) : null;

        // Clean up invalid stored key
        if (stored && !storedKey) {
          localStorage.removeItem(STORAGE_KEY);
        }

        // Make a single request, with stored key if available
        const headers: HeadersInit = {};
        if (storedKey) {
          headers["X-API-Key"] = storedKey;
        }

        const response = await fetch("/api/ui/v1/dashboard/summary", { headers });

        if (response.ok) {
          // Success - either no auth required, or stored key is valid
          if (storedKey) {
            setApiKeyState(storedKey);
            setGlobalApiKey(storedKey); // Set immediately so API calls work
            setAuthRequired(true); // Auth is configured, we just have a valid key
          } else {
            setAuthRequired(false); // No auth required on server
          }
        } else if (response.status === 401) {
          // Auth required and key (if any) is invalid
          setAuthRequired(true);
          setGlobalApiKey(null); // Clear any stale key
          if (stored) {
            localStorage.removeItem(STORAGE_KEY);
          }
        }
      } catch (err) {
        // Network error - assume auth might be required, prompt user
        console.error("Auth check failed:", err);
        setAuthRequired(true);
      } finally {
        setLoading(false);
      }
    };

    void checkAuth();
  }, []);

  const setApiKey = (key: string | null) => {
    setApiKeyState(key);
    setGlobalApiKey(key); // Set immediately so API calls work
    if (key) {
      localStorage.setItem(STORAGE_KEY, encryptKey(key));
    } else {
      localStorage.removeItem(STORAGE_KEY);
    }
  };

  const clearAuth = () => {
    setApiKeyState(null);
    setGlobalApiKey(null);
    localStorage.removeItem(STORAGE_KEY);
  };

  if (loading) {
    return <div className="flex items-center justify-center min-h-screen">Loading...</div>;
  }

  return (
    <AuthContext.Provider
      value={{
        apiKey,
        setApiKey,
        isAuthenticated: !authRequired || !!apiKey,
        authRequired,
        clearAuth,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error("useAuth must be used within AuthProvider");
  }
  return ctx;
}
