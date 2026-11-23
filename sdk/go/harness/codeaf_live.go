package harness

import (
	"bytes"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// Live watches and steers one codeaf run while it works.
//
// A codeaf run is otherwise silent until its final envelope. With Live set, the
// provider passes codeaf two files (`--events` and `--steer`), tails the first
// and hands every line to Event VERBATIM — the provider never parses or reshapes
// codeaf's entries, so whatever codeaf streams is what a watcher receives — and
// writes every note read from Steer to the second, where codeaf reads it at its
// next turn boundary.
//
// Begin is called once the run is about to start and End once it is over, so a
// caller can say "this execution is steerable now" only while it really is. A
// run that never reaches the binary (no key, no binary) calls neither.
type Live struct {
	Event func(line json.RawMessage)
	Steer <-chan string
	Begin func()
	End   func()
}

// livePoll is how often the events file is read. A quarter second keeps a
// transcript visibly live without spinning.
const livePoll = 250 * time.Millisecond

// liveRun is one armed run: its two files and the goroutines that serve them.
type liveRun struct {
	live       *Live
	dir        string
	eventsPath string
	steerPath  string
	// sessionPath is the run's own conversation (--session), never created
	// beforehand so codeaf starts it fresh.
	sessionPath string
	stop        chan struct{}
	wg          sync.WaitGroup
	offset      int64
	partial     []byte
}

// startLive makes the run's two files and starts tailing and steering. The
// caller passes the returned flags to codeaf and calls finish when it exits.
func startLive(live *Live) (*liveRun, []string, error) {
	if live == nil {
		return nil, nil, nil
	}
	dir, err := os.MkdirTemp("", "codeaf-live-")
	if err != nil {
		return nil, nil, err
	}
	r := &liveRun{live: live, dir: dir, eventsPath: filepath.Join(dir, "events.ndjson"), steerPath: filepath.Join(dir, "steer.ndjson"),
		sessionPath: filepath.Join(dir, "transcript.jsonl"), stop: make(chan struct{})}
	// Both files exist before codeaf starts, so neither side races a create.
	for _, p := range []string{r.eventsPath, r.steerPath} {
		f, err := os.OpenFile(p, os.O_CREATE|os.O_WRONLY, 0o600)
		if err != nil {
			_ = os.RemoveAll(dir)
			return nil, nil, err
		}
		_ = f.Close()
	}
	if live.Begin != nil {
		live.Begin()
	}
	r.wg.Add(1)
	go r.tail()
	if live.Steer != nil {
		r.wg.Add(1)
		go r.steer()
	}
	// EVERY WATCHED RUN IS ITS OWN CONVERSATION. A `chat --once` that names no
	// session resumes the newest one in its directory, and a node reuses its
	// clone of a repository from run to run — so the second ticket's step
	// opened on the first ticket's whole turn. The session file is named in
	// the run's own directory and not created, so codeaf starts it fresh; the
	// watcher (WorkAF) keeps the record, and the run's directory goes with it.
	return r, []string{"--events", r.eventsPath, "--steer", r.steerPath, "--session", r.sessionPath}, nil
}

func (r *liveRun) tail() {
	defer r.wg.Done()
	t := time.NewTicker(livePoll)
	defer t.Stop()
	for {
		select {
		case <-r.stop:
			r.drain(true)
			return
		case <-t.C:
			r.drain(false)
		}
	}
}

// drain hands over every complete line appended since the last read; at the
// end it also hands over a last line codeaf wrote without its newline.
func (r *liveRun) drain(final bool) {
	f, err := os.Open(r.eventsPath)
	if err != nil {
		return
	}
	defer f.Close()
	if _, err := f.Seek(r.offset, io.SeekStart); err != nil {
		return
	}
	raw, _ := io.ReadAll(f)
	r.offset += int64(len(raw))
	buf := append(r.partial, raw...)
	for {
		i := bytes.IndexByte(buf, '\n')
		if i < 0 {
			break
		}
		r.emit(buf[:i])
		buf = buf[i+1:]
	}
	r.partial = append([]byte(nil), buf...)
	if final && len(bytes.TrimSpace(r.partial)) > 0 {
		r.emit(r.partial)
		r.partial = nil
	}
}

func (r *liveRun) emit(line []byte) {
	line = bytes.TrimSpace(line)
	if len(line) == 0 || r.live.Event == nil || !json.Valid(line) {
		return
	}
	r.live.Event(json.RawMessage(append([]byte(nil), line...)))
}

func (r *liveRun) steer() {
	defer r.wg.Done()
	for {
		select {
		case <-r.stop:
			return
		case note, ok := <-r.live.Steer:
			if !ok {
				return
			}
			note = strings.TrimSpace(note)
			if note == "" {
				continue
			}
			line, _ := json.Marshal(map[string]string{"text": note})
			f, err := os.OpenFile(r.steerPath, os.O_WRONLY|os.O_APPEND, 0o600)
			if err != nil {
				continue
			}
			_, _ = f.Write(append(line, '\n'))
			_ = f.Close()
		}
	}
}

// finish stops tailing after one last read, calls End and removes the files.
func (r *liveRun) finish() {
	if r == nil {
		return
	}
	close(r.stop)
	r.wg.Wait()
	if r.live.End != nil {
		r.live.End()
	}
	_ = os.RemoveAll(r.dir)
}
