import { Code, User } from '@/components/ui/icon-bridge';
import { useMode } from '../contexts/ModeContext';

export function ModeToggle() {
  const { mode, toggleMode } = useMode();

  return (
    <div className="flex items-center gap-2">
      <span className="text-sm hidden sm:inline">Mode:</span>
      <button
        onClick={toggleMode}
        className="flex items-center gap-2 px-3 py-1.5 rounded-full border border-border bg-muted hover:bg-card transition-colors duration-200"
        title={`Switch to ${mode === 'developer' ? 'user' : 'developer'} mode`}
      >
        {mode === 'developer' ? (
          <>
            <Code className="h-4 w-4 text-blue-400" />
            <span className="text-sm font-medium text-blue-400 hidden sm:inline">Developer</span>
          </>
        ) : (
          <>
            <User className="h-4 w-4 text-green-400" />
            <span className="text-sm font-medium text-green-400 hidden sm:inline">User</span>
          </>
        )}
      </button>
    </div>
  );
}
