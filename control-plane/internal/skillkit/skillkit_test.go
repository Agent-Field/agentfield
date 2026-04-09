package skillkit

import (
	"errors"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

type fakeTarget struct {
	name         string
	displayName  string
	method       string
	detected     bool
	path         string
	installErr   error
	uninstallErr error
	installCalls int
}

func (f *fakeTarget) Name() string       { return f.name }
func (f *fakeTarget) DisplayName() string { return f.displayName }
func (f *fakeTarget) Method() string     { return f.method }
func (f *fakeTarget) Detected() bool     { return f.detected }
func (f *fakeTarget) TargetPath() (string, error) {
	if f.path == "" {
		return "", errors.New("missing path")
	}
	return f.path, nil
}
func (f *fakeTarget) Install(skill Skill, _ string) (InstalledTarget, error) {
	f.installCalls++
	if f.installErr != nil {
		return InstalledTarget{}, f.installErr
	}
	return InstalledTarget{
		TargetName:  f.name,
		Method:      f.method,
		Path:        f.path,
		Version:     skill.Version,
		InstalledAt: time.Now().UTC(),
	}, nil
}
func (f *fakeTarget) Uninstall() error { return f.uninstallErr }
func (f *fakeTarget) Status() (bool, string, error) {
	return false, "", nil
}

func withEnv(t *testing.T, key, value string) {
	t.Helper()
	t.Setenv(key, value)
}

func withTempHome(t *testing.T) string {
	t.Helper()
	home := t.TempDir()
	withEnv(t, "HOME", home)
	withEnv(t, "AGENTFIELD_HOME", filepath.Join(home, ".agentfield-home"))
	withEnv(t, "PATH", filepath.Join(home, "bin"))
	return home
}

func captureStdout(t *testing.T, fn func()) string {
	t.Helper()
	orig := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("pipe: %v", err)
	}
	os.Stdout = w
	done := make(chan []byte, 1)
	go func() {
		data, _ := io.ReadAll(r)
		done <- data
	}()
	fn()
	_ = w.Close()
	os.Stdout = orig
	out := <-done
	_ = r.Close()
	return string(out)
}

func TestCatalogFunctions(t *testing.T) {
	skill, err := CatalogByName(Catalog[0].Name)
	if err != nil {
		t.Fatalf("CatalogByName success: %v", err)
	}
	if skill.Name != Catalog[0].Name {
		t.Fatalf("unexpected skill name: %q", skill.Name)
	}

	if _, err := CatalogByName("missing"); err == nil || !strings.Contains(err.Error(), "available:") {
		t.Fatalf("CatalogByName missing error = %v", err)
	}

	files, err := skill.EnumerateFiles()
	if err != nil {
		t.Fatalf("EnumerateFiles: %v", err)
	}
	if len(files) == 0 || len(files["SKILL.md"]) == 0 {
		t.Fatalf("EnumerateFiles missing SKILL.md: %v", mapKeys(files))
	}

	entry, err := skill.EntryContent()
	if err != nil {
		t.Fatalf("EntryContent: %v", err)
	}
	if !strings.Contains(string(entry), "AgentField") {
		t.Fatalf("EntryContent missing expected content")
	}

	if _, err := (Skill{Name: "bad", EmbedRoot: "skill_data/missing", EntryFile: "SKILL.md"}).EnumerateFiles(); err == nil {
		t.Fatal("EnumerateFiles expected error for missing root")
	}

	if _, err := relativeUnderEmbed(skill.EmbedRoot, filepath.Join(skill.EmbedRoot, "SKILL.md")); err != nil {
		t.Fatalf("relativeUnderEmbed success: %v", err)
	}
	if _, err := relativeUnderEmbed(skill.EmbedRoot, "outside/SKILL.md"); err == nil {
		t.Fatal("relativeUnderEmbed expected error")
	}
}

func TestStateFunctions(t *testing.T) {
	home := withTempHome(t)

	root, err := CanonicalRoot()
	if err != nil {
		t.Fatalf("CanonicalRoot: %v", err)
	}
	if !strings.Contains(root, ".agentfield-home") {
		t.Fatalf("CanonicalRoot = %q", root)
	}

	state, err := LoadState()
	if err != nil {
		t.Fatalf("LoadState empty: %v", err)
	}
	if state.Version != stateFileVersion || len(state.Skills) != 0 {
		t.Fatalf("unexpected empty state: %+v", state)
	}

	s := &State{
		Skills: map[string]InstalledSkill{
			Catalog[0].Name: {
				CurrentVersion:    Catalog[0].Version,
				InstalledAt:       time.Now().UTC(),
				AvailableVersions: []string{Catalog[0].Version},
				Targets: map[string]InstalledTarget{
					"zeta": {TargetName: "zeta"},
					"alpha": {TargetName: "alpha"},
				},
			},
		},
	}
	if err := SaveState(s); err != nil {
		t.Fatalf("SaveState: %v", err)
	}

	loaded, err := LoadState()
	if err != nil {
		t.Fatalf("LoadState saved: %v", err)
	}
	if loaded.Version != stateFileVersion || loaded.Skills[Catalog[0].Name].CurrentVersion != Catalog[0].Version {
		t.Fatalf("unexpected loaded state: %+v", loaded)
	}

	names := loaded.Skills[Catalog[0].Name].SortedTargetNames()
	if strings.Join(names, ",") != "alpha,zeta" {
		t.Fatalf("SortedTargetNames = %v", names)
	}

	statePath := filepath.Join(root, ".state.json")
	if err := os.WriteFile(statePath, []byte("{"), 0o644); err != nil {
		t.Fatalf("write invalid state: %v", err)
	}
	if _, err := LoadState(); err == nil || !strings.Contains(err.Error(), "parse state file") {
		t.Fatalf("LoadState invalid error = %v", err)
	}

	missingErr := os.ErrNotExist
	if !errIsNotExist(missingErr) || errIsNotExist(nil) {
		t.Fatalf("errIsNotExist unexpected results")
	}

	if !strings.Contains(homeDir(), home) {
		t.Fatalf("homeDir = %q want under %q", homeDir(), home)
	}
}

func TestMarkerBlockFunctions(t *testing.T) {
	dir := t.TempDir()
	targetPath := filepath.Join(dir, "AGENTS.md")
	skillV1 := Skill{Name: Catalog[0].Name, Version: "1.0.0", Description: Catalog[0].Description, EntryFile: "SKILL.md"}
	skillV2 := skillV1
	skillV2.Version = "2.0.0"

	if err := os.WriteFile(targetPath, []byte("prefix\n"), 0o644); err != nil {
		t.Fatalf("seed target: %v", err)
	}

	inst, err := installMarkerBlock(skillV1, "/canonical/current", targetPath)
	if err != nil {
		t.Fatalf("installMarkerBlock first: %v", err)
	}
	if inst.Method != "marker-block" || inst.Version != "1.0.0" {
		t.Fatalf("unexpected install result: %+v", inst)
	}
	if got := readMarkerVersion(skillV1, targetPath); got != "1.0.0" {
		t.Fatalf("readMarkerVersion = %q", got)
	}

	if _, err := installMarkerBlock(skillV2, "/canonical/current", targetPath); err != nil {
		t.Fatalf("installMarkerBlock replace: %v", err)
	}
	content, err := os.ReadFile(targetPath)
	if err != nil {
		t.Fatalf("read target: %v", err)
	}
	text := string(content)
	if strings.Count(text, markerStartPattern(skillV1)) != 1 || !strings.Contains(text, "v2.0.0") {
		t.Fatalf("unexpected marker content:\n%s", text)
	}

	multi := "before\n" + renderPointerBlock(skillV1, "/x") + "\n" + renderPointerBlock(skillV1, "/y") + "\nafter\n"
	stripped := stripMarkerBlock(multi, skillV1)
	if !strings.Contains(stripped, "before") || !strings.Contains(stripped, "after") || strings.Contains(stripped, markerStartPattern(skillV1)) {
		t.Fatalf("stripMarkerBlock multi = %q", stripped)
	}

	malformed := "before\n" + markerStart(skillV1) + "\npartial"
	if got := stripMarkerBlock(malformed, skillV1); got != "before" {
		t.Fatalf("stripMarkerBlock malformed = %q", got)
	}

	if err := uninstallMarkerBlock(skillV1, targetPath); err != nil {
		t.Fatalf("uninstallMarkerBlock: %v", err)
	}
	content, err = os.ReadFile(targetPath)
	if err != nil {
		t.Fatalf("read cleaned target: %v", err)
	}
	if strings.TrimSpace(string(content)) != "prefix" {
		t.Fatalf("unexpected cleaned content: %q", string(content))
	}

	if err := os.WriteFile(targetPath, []byte(renderPointerBlock(skillV1, "/z")+"\n"), 0o644); err != nil {
		t.Fatalf("seed marker-only file: %v", err)
	}
	if err := uninstallMarkerBlock(skillV1, targetPath); err != nil {
		t.Fatalf("uninstall marker-only: %v", err)
	}
	if _, err := os.Stat(targetPath); !os.IsNotExist(err) {
		t.Fatalf("expected target removal, stat err = %v", err)
	}

	if got := readMarkerVersion(skillV1, targetPath); got != "" {
		t.Fatalf("readMarkerVersion missing = %q", got)
	}
}

func TestHelpersAndTargets(t *testing.T) {
	home := withTempHome(t)
	if commandAvailable("definitely-not-installed-binary") {
		t.Fatal("commandAvailable unexpectedly true")
	}
	if dirExists("") || fileExists("") {
		t.Fatal("empty path should not exist")
	}

	dirPath := filepath.Join(home, "dir")
	filePath := filepath.Join(home, "file.txt")
	if err := os.MkdirAll(dirPath, 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(filePath, []byte("x"), 0o644); err != nil {
		t.Fatalf("write file: %v", err)
	}
	if !dirExists(dirPath) || !fileExists(filePath) {
		t.Fatal("dirExists/fileExists false")
	}

	skill := Catalog[0]
	if !strings.Contains(markerStart(skill), skill.Name) || !strings.Contains(markerEnd(skill), skill.Name) {
		t.Fatal("marker helpers missing skill name")
	}
	block := renderPointerBlock(skill, "/canonical/current")
	if !strings.Contains(block, filepath.Join("/canonical/current", skill.EntryFile)) || !strings.Contains(block, skill.Version) {
		t.Fatalf("renderPointerBlock = %q", block)
	}
	if platformInfo() != runtime.GOOS {
		t.Fatalf("platformInfo = %q want %q", platformInfo(), runtime.GOOS)
	}

	targetNames := map[string]bool{}
	for _, target := range AllTargets() {
		targetNames[target.Name()] = true
	}
	for _, name := range []string{"aider", "claude-code", "codex", "cursor", "gemini", "opencode", "windsurf"} {
		if !targetNames[name] {
			t.Fatalf("missing registered target %q", name)
		}
	}
	if _, err := TargetByName("missing"); err == nil {
		t.Fatal("TargetByName missing expected error")
	}

	if err := ensureLineInFile(filepath.Join(home, ".aider.conf.yml"), "read: /tmp/skill"); err != nil {
		t.Fatalf("ensureLineInFile create: %v", err)
	}
	if err := ensureLineInFile(filepath.Join(home, ".aider.conf.yml"), "read: /tmp/skill"); err != nil {
		t.Fatalf("ensureLineInFile idempotent: %v", err)
	}
	confData, err := os.ReadFile(filepath.Join(home, ".aider.conf.yml"))
	if err != nil {
		t.Fatalf("read conf: %v", err)
	}
	if strings.Count(string(confData), "read: /tmp/skill") != 1 {
		t.Fatalf("unexpected conf content: %q", string(confData))
	}

	if err := os.MkdirAll(filepath.Join(home, ".codex"), 0o755); err != nil {
		t.Fatalf("mkdir codex dir: %v", err)
	}
	codex := codexTarget{}
	if codex.DisplayName() != "Codex (OpenAI)" || codex.Method() != "marker-block" {
		t.Fatalf("unexpected codex metadata: %q %q", codex.DisplayName(), codex.Method())
	}
	if !codex.Detected() {
		t.Fatal("codex target should be detected")
	}
	codexPath, _ := codex.TargetPath()
	if _, err := codex.Install(skill, filepath.Join(home, "canonical", "current")); err != nil {
		t.Fatalf("codex install: %v", err)
	}
	if installed, version, err := codex.Status(); err != nil || !installed || version != skill.Version {
		t.Fatalf("codex status = %v %q %v", installed, version, err)
	}
	if err := codex.Uninstall(); err != nil {
		t.Fatalf("codex uninstall: %v", err)
	}
	if installed, _, _ := codex.Status(); installed {
		t.Fatal("codex should not be installed after uninstall")
	}
	if !strings.HasSuffix(codexPath, filepath.Join(".codex", "AGENTS.override.md")) {
		t.Fatalf("unexpected codex path: %q", codexPath)
	}

	if err := os.MkdirAll(filepath.Join(home, ".aider"), 0o755); err != nil {
		t.Fatalf("mkdir aider dir: %v", err)
	}
	aider := aiderTarget{}
	if aider.DisplayName() != "Aider" || aider.Method() != "marker-block" {
		t.Fatalf("unexpected aider metadata: %q %q", aider.DisplayName(), aider.Method())
	}
	if !aider.Detected() {
		t.Fatal("aider target should be detected")
	}
	aiderPath, _ := aider.TargetPath()
	if _, err := aider.Install(skill, filepath.Join(home, "canonical", "current")); err != nil {
		t.Fatalf("aider install: %v", err)
	}
	aiderConf, err := os.ReadFile(filepath.Join(home, ".aider.conf.yml"))
	if err != nil || !strings.Contains(string(aiderConf), "read: "+aiderPath) {
		t.Fatalf("aider conf not updated: %q err=%v", string(aiderConf), err)
	}
	if err := aider.Uninstall(); err != nil {
		t.Fatalf("aider uninstall: %v", err)
	}
	if installed, _, err := aider.Status(); err != nil || installed {
		t.Fatalf("aider status after uninstall = %v %v", installed, err)
	}

	if err := os.MkdirAll(filepath.Join(home, ".claude"), 0o755); err != nil {
		t.Fatalf("mkdir claude dir: %v", err)
	}
	claude := claudeCodeTarget{}
	if claude.DisplayName() != "Claude Code" || claude.Method() != "symlink" {
		t.Fatalf("unexpected claude metadata: %q %q", claude.DisplayName(), claude.Method())
	}
	if !claude.Detected() {
		t.Fatal("claude target should be detected")
	}
	canonicalCurrentDir := filepath.Join(home, "canonical", "current")
	if err := os.MkdirAll(canonicalCurrentDir, 0o755); err != nil {
		t.Fatalf("mkdir canonical current: %v", err)
	}
	inst, err := claude.Install(skill, canonicalCurrentDir)
	if err != nil {
		t.Fatalf("claude install: %v", err)
	}
	if inst.Method != "symlink" {
		t.Fatalf("claude install method = %q", inst.Method)
	}
	if installed, version, err := claude.Status(); err != nil || !installed || version != "current" {
		t.Fatalf("claude status = %v %q %v", installed, version, err)
	}
	if err := claude.Uninstall(); err != nil {
		t.Fatalf("claude uninstall: %v", err)
	}
	if installed, _, _ := claude.Status(); installed {
		t.Fatal("claude should not be installed after uninstall")
	}

	if err := os.MkdirAll(filepath.Join(home, ".gemini"), 0o755); err != nil {
		t.Fatalf("mkdir gemini dir: %v", err)
	}
	gemini := geminiTarget{}
	if gemini.DisplayName() != "Gemini CLI" || gemini.Method() != "marker-block" {
		t.Fatalf("unexpected gemini metadata: %q %q", gemini.DisplayName(), gemini.Method())
	}
	if !gemini.Detected() {
		t.Fatal("gemini target should be detected")
	}
	if _, err := gemini.Install(skill, filepath.Join(home, "canonical", "current")); err != nil {
		t.Fatalf("gemini install: %v", err)
	}
	if installed, version, err := gemini.Status(); err != nil || !installed || version != skill.Version {
		t.Fatalf("gemini status = %v %q %v", installed, version, err)
	}
	if err := gemini.Uninstall(); err != nil {
		t.Fatalf("gemini uninstall: %v", err)
	}

	if err := os.MkdirAll(filepath.Join(home, ".config", "opencode"), 0o755); err != nil {
		t.Fatalf("mkdir opencode dir: %v", err)
	}
	opencode := opencodeTarget{}
	if opencode.DisplayName() != "OpenCode" || opencode.Method() != "marker-block" {
		t.Fatalf("unexpected opencode metadata: %q %q", opencode.DisplayName(), opencode.Method())
	}
	if !opencode.Detected() {
		t.Fatal("opencode target should be detected")
	}
	if _, err := opencode.Install(skill, filepath.Join(home, "canonical", "current")); err != nil {
		t.Fatalf("opencode install: %v", err)
	}
	if installed, version, err := opencode.Status(); err != nil || !installed || version != skill.Version {
		t.Fatalf("opencode status = %v %q %v", installed, version, err)
	}
	if err := opencode.Uninstall(); err != nil {
		t.Fatalf("opencode uninstall: %v", err)
	}

	if err := os.MkdirAll(filepath.Join(home, ".codeium"), 0o755); err != nil {
		t.Fatalf("mkdir windsurf dir: %v", err)
	}
	windsurf := windsurfTarget{}
	if windsurf.DisplayName() != "Windsurf" || windsurf.Method() != "marker-block" {
		t.Fatalf("unexpected windsurf metadata: %q %q", windsurf.DisplayName(), windsurf.Method())
	}
	if !windsurf.Detected() {
		t.Fatal("windsurf target should be detected")
	}
	if _, err := windsurf.Install(skill, filepath.Join(home, "canonical", "current")); err != nil {
		t.Fatalf("windsurf install: %v", err)
	}
	if installed, version, err := windsurf.Status(); err != nil || !installed || version != skill.Version {
		t.Fatalf("windsurf status = %v %q %v", installed, version, err)
	}
	if err := windsurf.Uninstall(); err != nil {
		t.Fatalf("windsurf uninstall: %v", err)
	}

	detected := map[string]bool{}
	for _, target := range DetectedTargets() {
		detected[target.Name()] = true
	}
	for _, name := range []string{"aider", "claude-code", "codex", "gemini", "opencode", "windsurf"} {
		if !detected[name] {
			t.Fatalf("DetectedTargets missing %q: %+v", name, detected)
		}
	}

	out := captureStdout(t, func() {
		cursor := cursorTarget{}
		if cursor.DisplayName() != "Cursor" || cursor.Method() != "manual" {
			t.Fatalf("unexpected cursor metadata: %q %q", cursor.DisplayName(), cursor.Method())
		}
		if cursor.Detected() {
			t.Fatal("cursor should not be detected in temp home")
		}
		if _, err := cursor.TargetPath(); err == nil {
			t.Fatal("cursor TargetPath expected error")
		}
		inst, err := cursor.Install(skill, canonicalCurrentDir)
		if err != nil {
			t.Fatalf("cursor install: %v", err)
		}
		if inst.Method != "manual" || inst.Version != skill.Version {
			t.Fatalf("unexpected cursor install: %+v", inst)
		}
		if err := cursor.Uninstall(); err != nil {
			t.Fatalf("cursor uninstall: %v", err)
		}
		if installed, version, err := cursor.Status(); err != nil || installed || version != "" {
			t.Fatalf("cursor status = %v %q %v", installed, version, err)
		}
	})
	if !strings.Contains(out, "Cursor manual install required") || !strings.Contains(out, "manual uninstall") {
		t.Fatalf("unexpected cursor output: %q", out)
	}
}

func TestResolveTargetsAndInstallLifecycle(t *testing.T) {
	withTempHome(t)

	origTargets := allTargets
	t.Cleanup(func() { allTargets = origTargets })

	success := &fakeTarget{name: "success", displayName: "Success", method: "marker-block", detected: true, path: "/tmp/success"}
	failing := &fakeTarget{name: "failing", displayName: "Failing", method: "marker-block", detected: true, path: "/tmp/failing", installErr: errors.New("boom")}
	skippedTarget := &fakeTarget{name: "skipped", displayName: "Skipped", method: "marker-block", detected: false, path: "/tmp/skipped"}
	allTargets = []Target{success, failing, skippedTarget}

	selected, skipped, err := resolveTargets(InstallOptions{AllDetected: true})
	if err != nil {
		t.Fatalf("resolveTargets all detected: %v", err)
	}
	if len(selected) != 2 || len(skipped) != 1 || skipped[0].TargetName != "skipped" {
		t.Fatalf("unexpected resolveTargets result: selected=%d skipped=%+v", len(selected), skipped)
	}

	selected, skipped, err = resolveTargets(InstallOptions{})
	if err != nil || len(selected) != 2 || len(skipped) != 1 || skipped[0].Reason != "not detected" {
		t.Fatalf("default resolveTargets = %d %+v %v", len(selected), skipped, err)
	}

	selected, skipped, err = resolveTargets(InstallOptions{AllRegistered: true})
	if err != nil || len(selected) != 3 || len(skipped) != 0 {
		t.Fatalf("all registered resolveTargets = %d %+v %v", len(selected), skipped, err)
	}

	if _, _, err := resolveTargets(InstallOptions{Targets: []string{"missing"}}); err == nil {
		t.Fatal("resolveTargets expected missing target error")
	}

	report, err := Install(InstallOptions{SkillName: Catalog[0].Name, AllDetected: true})
	if err != nil {
		t.Fatalf("Install: %v", err)
	}
	if !report.WroteCanonical || len(report.TargetsInstalled) != 1 || len(report.TargetsFailed) != 1 || len(report.TargetsSkipped) != 1 {
		t.Fatalf("unexpected install report: %+v", report)
	}
	if success.installCalls != 1 || failing.installCalls != 1 {
		t.Fatalf("unexpected install call counts: success=%d failing=%d", success.installCalls, failing.installCalls)
	}
	if _, err := os.Stat(filepath.Join(report.CanonicalDir, "SKILL.md")); err != nil {
		t.Fatalf("canonical SKILL.md missing: %v", err)
	}
	if linkDest, err := os.Readlink(report.CurrentLink); err != nil || linkDest != Catalog[0].Version {
		t.Fatalf("current symlink = %q err=%v", linkDest, err)
	}

	state, err := LoadState()
	if err != nil {
		t.Fatalf("LoadState after install: %v", err)
	}
	if listed, err := ListInstalled(); err != nil || len(listed.Skills) != 1 {
		t.Fatalf("ListInstalled = %+v err=%v", listed, err)
	}
	if len(state.Skills[Catalog[0].Name].Targets) != 1 {
		t.Fatalf("unexpected saved targets: %+v", state.Skills[Catalog[0].Name].Targets)
	}

	report, err = Install(InstallOptions{SkillName: Catalog[0].Name, Targets: []string{"success"}})
	if err != nil {
		t.Fatalf("Install second pass: %v", err)
	}
	if len(report.TargetsInstalled) != 0 || len(report.TargetsSkipped) == 0 || !strings.Contains(report.TargetsSkipped[0].Reason, "already installed") {
		t.Fatalf("unexpected second install report: %+v", report)
	}

	report, err = Install(InstallOptions{SkillName: Catalog[0].Name, Targets: []string{"success"}, Force: true, DryRun: true})
	if err != nil {
		t.Fatalf("Install dry run: %v", err)
	}
	if report.WroteCanonical || len(report.TargetsInstalled) != 1 {
		t.Fatalf("unexpected dry-run report: %+v", report)
	}

	report, err = Update(Catalog[0].Name)
	if err != nil {
		t.Fatalf("Update: %v", err)
	}
	if len(report.TargetsInstalled) != 1 || success.installCalls < 2 {
		t.Fatalf("unexpected update report: %+v", report)
	}

	if err := Uninstall(UninstallOptions{SkillName: Catalog[0].Name, Targets: []string{"success"}}); err != nil {
		t.Fatalf("Uninstall target: %v", err)
	}
	state, err = LoadState()
	if err != nil {
		t.Fatalf("LoadState after uninstall target: %v", err)
	}
	if _, ok := state.Skills[Catalog[0].Name]; ok {
		t.Fatalf("skill should be removed after last target uninstall: %+v", state.Skills)
	}

	if _, err := Update(Catalog[0].Name); err == nil {
		t.Fatal("Update expected not-installed error")
	}
	if err := Uninstall(UninstallOptions{SkillName: Catalog[0].Name}); err == nil {
		t.Fatal("Uninstall expected not-installed error")
	}

	if _, err := resolveSkill(Catalog[0].Name, "999.0.0"); err == nil {
		t.Fatal("resolveSkill expected version mismatch error")
	}

	origCatalog := Catalog
	Catalog = nil
	if _, err := resolveSkill("", ""); err == nil {
		t.Fatal("resolveSkill expected empty catalog error")
	}
	Catalog = origCatalog

	linkPath := filepath.Join(t.TempDir(), "current")
	targetDir := filepath.Join(filepath.Dir(linkPath), "v1")
	if err := os.MkdirAll(targetDir, 0o755); err != nil {
		t.Fatalf("mkdir targetDir: %v", err)
	}
	if err := updateCurrentLink(linkPath, targetDir); err != nil {
		t.Fatalf("updateCurrentLink first: %v", err)
	}
	if dest, err := os.Readlink(linkPath); err != nil || dest != "v1" {
		t.Fatalf("updateCurrentLink symlink = %q err=%v", dest, err)
	}
	if err := os.Remove(linkPath); err != nil {
		t.Fatalf("remove link: %v", err)
	}
	if err := os.WriteFile(linkPath, []byte("x"), 0o644); err != nil {
		t.Fatalf("seed file: %v", err)
	}
	if err := updateCurrentLink(linkPath, targetDir); err != nil {
		t.Fatalf("updateCurrentLink replace file: %v", err)
	}
}

func mapKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}
