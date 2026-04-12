//go:build !windows

package agent

import (
	"bufio"
	"io"
	"os"
	"strings"
)

func installProcessStdioCapture(ring *processLogRing) {
	if ring == nil {
		return
	}
	hookStream("stdout", os.Stdout, ring)
	hookStream("stderr", os.Stderr, ring)
}

func hookStream(name string, orig *os.File, ring *processLogRing) {
	r, w, err := os.Pipe()
	if err != nil {
		return
	}
	switch name {
	case "stdout":
		os.Stdout = w
	case "stderr":
		os.Stderr = w
	default:
		_ = r.Close()
		_ = w.Close()
		return
	}
	go func() {
		defer func() { _ = r.Close() }()
		br := bufio.NewReader(r)
		maxLB := processLogsMaxLineBytes()
		for {
			line, err := br.ReadString('\n')
			if len(line) > 0 {
				s := strings.TrimSuffix(line, "\n")
				trunc := false
				if len(s) > maxLB {
					s = s[:maxLB]
					trunc = true
				}
				ring.appendLine(name, s, trunc)
				_, _ = io.WriteString(orig, s+"\n")
			}
			if err != nil {
				if err == io.EOF {
					return
				}
				return
			}
		}
	}()
}
