//go:build unix

package codeaf

import (
	"bytes"
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestInstalledEnsureDoesNotWaitForLock(t *testing.T) {
	opts, pins, _ := fixture(t)
	if err := ensure(opts, pins); err != nil {
		t.Fatal(err)
	}
	unlock, err := lockCodeaf(context.Background(), filepath.Join(opts.Home, "bin", ".codeaf.lock"))
	if err != nil {
		t.Fatal(err)
	}
	defer unlock()
	done := make(chan error, 1)
	go func() { done <- Ensure(opts) }()
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(200 * time.Millisecond):
		t.Fatal("installed binary waited for lock")
	}
}

func TestEnsureLockWaitBoundAndCancellation(t *testing.T) {
	for _, auto := range []bool{false, true} {
		t.Run(map[bool]string{false: "explicit", true: "automatic"}[auto], func(t *testing.T) {
			noSkip(t)
			resetBestEffort()
			t.Cleanup(resetBestEffort)
			home := t.TempDir()
			dir := filepath.Join(home, "bin")
			if err := os.MkdirAll(dir, 0755); err != nil {
				t.Fatal(err)
			}
			unlock, err := lockCodeaf(context.Background(), filepath.Join(dir, ".codeaf.lock"))
			if err != nil {
				t.Fatal(err)
			}
			defer unlock()
			ctx, cancel := context.WithTimeout(context.Background(), 80*time.Millisecond)
			defer cancel()
			opts := Options{Home: home, Context: ctx}
			var out bytes.Buffer
			started := time.Now()
			if auto {
				err = EnsureBestEffort(opts, &out)
			} else {
				err = Ensure(opts)
			}
			if time.Since(started) > time.Second {
				t.Fatal("lock wait exceeded context")
			}
			if auto {
				if err != nil || strings.Count(out.String(), "warning:") != 1 || strings.Contains(out.String(), "Installing") {
					t.Fatalf("automatic result=%v output=%q", err, out.String())
				}
			} else if !errors.Is(err, context.DeadlineExceeded) {
				t.Fatalf("context not honored: %v", err)
			}
			opts.Context = context.Background()
			opts.LockWait = 80 * time.Millisecond
			if err := Ensure(opts); !errors.Is(err, context.DeadlineExceeded) {
				t.Fatalf("lock budget not honored: %v", err)
			}
		})
	}
}
