package skillkit

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
)

// These seams keep reconciliation failure tests independent of platform file
// permissions. They deliberately cover every filesystem operation used while
// removing an obsolete recorded integration.
var (
	reconcileLstat     = os.Lstat
	reconcileRemove    = os.Remove
	reconcileRemoveAll = os.RemoveAll
	reconcileReadFile  = os.ReadFile
	reconcileWriteFile = os.WriteFile
	reconcileRename    = os.Rename
	reconcileSaveState = SaveState
)

// reconcileAliasOrphans removes state entries for old standalone skills which
// are now aliases of a catalog skill. It operates only on paths recorded in
// state; it must never resolve the alias to its canonical skill.
func reconcileAliasOrphans() error {
	state, err := LoadState()
	if err != nil {
		return err
	}
	orphans := aliasOrphanNames(state)
	if len(orphans) == 0 {
		return nil
	}

	root, err := CanonicalRoot()
	if err != nil {
		return err
	}
	for _, orphanName := range orphans {
		installed := state.Skills[orphanName]
		for _, targetName := range installed.SortedTargetNames() {
			if err := removeRecordedTarget(orphanName, targetName, installed.Targets[targetName]); err != nil {
				return fmt.Errorf("reconcile alias orphan %q target %q: %w", orphanName, targetName, err)
			}
		}
		if err := removeRecordedPath(filepath.Join(root, orphanName)); err != nil {
			return fmt.Errorf("reconcile alias orphan %q target %q: %w", orphanName, "canonical directory", err)
		}
		delete(state.Skills, orphanName)
	}
	if err := reconcileSaveState(state); err != nil {
		return fmt.Errorf("reconcile alias orphans save state: %w", err)
	}
	return nil
}

func aliasOrphanNames(state *State) []string {
	canonical := make(map[string]struct{}, len(Catalog))
	aliases := map[string]struct{}{}
	for _, skill := range Catalog {
		canonical[skill.Name] = struct{}{}
		for _, alias := range skill.Aliases {
			aliases[alias] = struct{}{}
		}
	}
	var names []string
	for name := range state.Skills {
		if _, isCanonical := canonical[name]; isCanonical {
			continue
		}
		if _, isAlias := aliases[name]; isAlias {
			names = append(names, name)
		}
	}
	sort.Strings(names)
	return names
}

func removeRecordedTarget(orphanName, targetName string, installed InstalledTarget) error {
	switch installed.Method {
	case "symlink":
		return removeRecordedPath(installed.Path)
	case "marker-block":
		// The empty version is intentional: marker matching is name based.
		return uninstallMarkerBlock(Skill{Name: orphanName}, installed.Path)
	default:
		return fmt.Errorf("unsupported install method %q", installed.Method)
	}
}

func removeRecordedPath(path string) error {
	info, err := reconcileLstat(path)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return err
	}
	if info.Mode()&os.ModeSymlink != 0 || !info.IsDir() {
		return reconcileRemove(path)
	}
	return reconcileRemoveAll(path)
}
