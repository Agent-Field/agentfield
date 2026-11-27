package harness

import (
	"context"
	"log"
	"os"
	"os/exec"
	"regexp"
	"strings"
	"sync"
)

// codeAFProbeTimeoutSeconds bounds `codeaf chat --help`, which only prints its
// flag list. A binary that cannot do that quickly is not worth waiting for: the
// run goes ahead unwatched.
const codeAFProbeTimeoutSeconds = 10

// codeAFLiveFlags are the flags a watched run passes to `codeaf chat --once`.
// A build that does not list every one of them refuses the command line at
// once (v0.8.3: "codeaf chat has no --json flag"), so a watched run on such a
// build is dead before it starts.
var codeAFLiveFlags = []string{"json", "events", "steer", "session"}

// codeAFLiveFlagListed reads one flag's line out of a `chat --help` listing.
// The help prints each flag on a line of its own, "  --json" or "  --events
// string" (a stock Go flag package would print "  -json"), so the flag must
// open the line. A usage line such as "[--session path]" mentions flags
// without listing them and must not count.
func codeAFLiveFlagListed(help, name string) bool {
	return regexp.MustCompile(`(?m)^[ \t]+--?` + regexp.QuoteMeta(name) + `([ \t]|$)`).MatchString(help)
}

// codeAFMissingLiveFlags names the live flags a `chat --help` output leaves out.
func codeAFMissingLiveFlags(help string) []string {
	var missing []string
	for _, name := range codeAFLiveFlags {
		if !codeAFLiveFlagListed(help, name) {
			missing = append(missing, "--"+name)
		}
	}
	return missing
}

// codeAFLiveFlagRejection recognises a run that died up front because codeaf
// does not know one of the watched run's flags: the stderr of a build that
// refuses the command line, before any work started.
var codeAFLiveFlagRejection = regexp.MustCompile(
	`has no --(json|events|steer|session) flag|flag provided but not defined: -{1,2}(json|events|steer|session|once)\b`)

// codeAFRejectedLiveFlags reports whether a failed watched run was refused for
// an unknown flag, with nothing run: it exited non-zero and printed no
// envelope. The older spelling, "-events ... not defined", still counts.
func codeAFRejectedLiveFlags(cli *CLIResult) bool {
	if cli == nil || cli.ReturnCode == 0 || parseCodeAFEnvelope(cli.Stdout) != nil {
		return false
	}
	return codeAFLiveFlagRejection.MatchString(cli.Stderr) ||
		(strings.Contains(cli.Stderr, "-events") && strings.Contains(cli.Stderr, "not defined"))
}

// codeAFProbeKey identifies one build of one binary. A binary replaced in
// place (an upgrade) changes its size or modification time and so is probed
// again.
type codeAFProbeKey struct {
	path  string
	mtime int64
	size  int64
}

// codeAFLiveProbes remembers what each binary answered, for the life of the
// process. The mutex is held across the probe on purpose: runs that start
// together share one `chat --help` instead of each running their own.
var codeAFLiveProbes = struct {
	sync.Mutex
	cache map[codeAFProbeKey]bool
}{cache: make(map[codeAFProbeKey]bool)}

func resetCodeAFLiveProbes() {
	codeAFLiveProbes.Lock()
	defer codeAFLiveProbes.Unlock()
	codeAFLiveProbes.cache = make(map[codeAFProbeKey]bool)
}

// codeAFProbeKeyFor resolves the binary on PATH when it is a bare name and
// stats it; ok is false when it cannot be, and then nothing is cached.
func codeAFProbeKeyFor(bin string) (codeAFProbeKey, bool) {
	path := bin
	if !strings.ContainsAny(path, `/\`) {
		resolved, err := exec.LookPath(path)
		if err != nil {
			return codeAFProbeKey{}, false
		}
		path = resolved
	}
	info, err := os.Stat(path)
	if err != nil {
		return codeAFProbeKey{}, false
	}
	return codeAFProbeKey{path: path, mtime: info.ModTime().UnixNano(), size: info.Size()}, true
}

// supportsLiveChat reports whether `bin chat --once` takes every flag a watched
// run passes, found by reading `bin chat --help` rather than by guessing at a
// version number. The answer is remembered per binary (path, modification time
// and size), so the probe runs once per process per binary, and so does the
// warning a "no" logs. A probe that could not run (a timeout, a missing binary)
// says no for this run and is not remembered, so a flaky start does not turn
// watching off for good.
func (p *CodeAFProvider) supportsLiveChat(ctx context.Context, bin string, env map[string]string) bool {
	key, keyed := codeAFProbeKeyFor(bin)
	if keyed {
		codeAFLiveProbes.Lock()
		defer codeAFLiveProbes.Unlock()
		if supported, ok := codeAFLiveProbes.cache[key]; ok {
			return supported
		}
	}
	probe := p.probeCLI
	if probe == nil {
		probe = runCodeAFCLI
	}
	cli, err := probe(ctx, []string{bin, "chat", "--help"}, env, "", codeAFProbeTimeoutSeconds, 0, nil)
	// Both streams: this build prints the flag list on stdout, and a stock
	// flag package prints it on stderr.
	help := ""
	if cli != nil {
		help = cli.Stdout + "\n" + cli.Stderr
	}
	if err != nil || cli == nil || cli.ReturnCode < 0 || (cli.ReturnCode != 0 && strings.TrimSpace(help) == "") {
		reason := "did not finish"
		if err != nil {
			reason = err.Error()
		}
		log.Printf("WARNING CodeAF at %s could not be probed for watched-run support (`chat --help` %s); running unwatched with `exec`", bin, reason)
		return false
	}
	missing := codeAFMissingLiveFlags(help)
	supported := len(missing) == 0
	if keyed {
		codeAFLiveProbes.cache[key] = supported
	}
	if !supported {
		log.Printf("WARNING CodeAF at %s is older than this SDK expects: `chat --help` does not list %s, so a watched run would be refused at once. Running unwatched with `exec`; upgrade CodeAF to watch runs live", bin, strings.Join(missing, ", "))
	}
	return supported
}
