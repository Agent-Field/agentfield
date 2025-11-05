import type { ReactNode } from "react";
import { createContext, useContext, useEffect, useState } from "react";

export type AppMode = "developer" | "user";

interface ModeContextType {
  mode: AppMode;
  setMode: (mode: AppMode) => void;
  toggleMode: () => void;
}

const ModeContext = createContext<ModeContextType | undefined>(undefined);

interface ModeProviderProps {
  children: ReactNode;
}

export function ModeProvider({ children }: ModeProviderProps) {
  const [mode, setModeState] = useState<AppMode>(() => {
    // Load mode from localStorage on initialization
    const savedMode = localStorage.getItem("haxen-app-mode");
    return savedMode === "developer" || savedMode === "user"
      ? savedMode
      : "developer";
  });

  const setMode = (newMode: AppMode) => {
    setModeState(newMode);
    localStorage.setItem("haxen-app-mode", newMode);
  };

  const toggleMode = () => {
    const newMode = mode === "developer" ? "user" : "developer";
    setMode(newMode);
  };

  // Persist mode changes to localStorage
  useEffect(() => {
    localStorage.setItem("haxen-app-mode", mode);
  }, [mode]);

  return (
    <ModeContext.Provider value={{ mode, setMode, toggleMode }}>
      {children}
    </ModeContext.Provider>
  );
}

export function useMode() {
  const context = useContext(ModeContext);
  if (context === undefined) {
    throw new Error("useMode must be used within a ModeProvider");
  }
  return context;
}
