package packages

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// What an archive is allowed to contain, and what happens to the rest.
//
// A Go toolchain archive is downloaded from the network and unpacked onto the
// user's machine, so the entry policy is a security control, not a formality:
// only directories and regular files are written, everything else is refused
// before a single byte lands. These tests pin that policy directly — the
// happy-path provisioning tests never exercise it, because a well-formed Go
// archive contains nothing unusual.
//
// The real go1.26.5.linux-amd64 tarball is 15026 regular files and 1667
// directories with zero links, so refusing links costs nothing on the genuine
// artifact and closes the door on a tampered one.

// tarEntry is one entry to write into a test tarball.
type tarEntry struct {
	name     string
	typeflag byte
	body     string
	linkname string
	mode     int64
}

func writeTarGz(t *testing.T, entries []tarEntry) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "fixture.tar.gz")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	gz := gzip.NewWriter(f)
	tw := tar.NewWriter(gz)
	for _, e := range entries {
		mode := e.mode
		if mode == 0 {
			mode = 0o644
		}
		h := &tar.Header{Name: e.name, Typeflag: e.typeflag, Mode: mode, Linkname: e.linkname}
		if e.typeflag == tar.TypeReg {
			h.Size = int64(len(e.body))
		}
		if err := tw.WriteHeader(h); err != nil {
			t.Fatal(err)
		}
		if e.typeflag == tar.TypeReg {
			if _, err := tw.Write([]byte(e.body)); err != nil {
				t.Fatal(err)
			}
		}
	}
	for _, c := range []func() error{tw.Close, gz.Close, f.Close} {
		if err := c(); err != nil {
			t.Fatal(err)
		}
	}
	return path
}

// Contract: directories and regular files extract, preserving the file mode —
// the toolchain is unusable if `go/bin/go` arrives without its execute bit.
func TestExtractGoTarGz_WritesDirsAndFilesPreservingMode(t *testing.T) {
	archive := writeTarGz(t, []tarEntry{
		{name: "go", typeflag: tar.TypeDir, mode: 0o755},
		{name: "go/bin", typeflag: tar.TypeDir, mode: 0o755},
		{name: "go/bin/go", typeflag: tar.TypeReg, body: "#!/bin/sh\n", mode: 0o755},
		{name: "go/VERSION", typeflag: tar.TypeReg, body: "go1.26.5", mode: 0o644},
	})
	dst := t.TempDir()
	if err := extractGoTarGz(archive, dst); err != nil {
		t.Fatalf("extract: %v", err)
	}

	info, err := os.Stat(filepath.Join(dst, "go", "bin", "go"))
	if err != nil {
		t.Fatalf("go/bin/go missing: %v", err)
	}
	if info.Mode().Perm()&0o111 == 0 {
		t.Fatalf("go/bin/go is not executable: %v", info.Mode())
	}
	if body, err := os.ReadFile(filepath.Join(dst, "go", "VERSION")); err != nil || string(body) != "go1.26.5" {
		t.Fatalf("VERSION = %q, err = %v", body, err)
	}
}

// Contract: a tar entry that is not a directory or a regular file is refused.
// Links are the dangerous ones — a symlink to /etc plus a later write through
// it is the classic way an archive escapes its destination — but anything we do
// not positively understand is refused too, rather than silently skipped.
func TestExtractGoTarGz_RefusesLinksAndExoticEntries(t *testing.T) {
	for _, tc := range []struct {
		name  string
		entry tarEntry
		want  string
	}{
		{
			name:  "symlink escaping the destination",
			entry: tarEntry{name: "go/bin/go", typeflag: tar.TypeSymlink, linkname: "../../../../etc/passwd"},
			want:  "links are not allowed",
		},
		{
			name:  "symlink pointing somewhere harmless",
			entry: tarEntry{name: "go/bin/gofmt", typeflag: tar.TypeSymlink, linkname: "go"},
			want:  "links are not allowed",
		},
		{
			name:  "hard link",
			entry: tarEntry{name: "go/bin/gofmt", typeflag: tar.TypeLink, linkname: "go/bin/go"},
			want:  "links are not allowed",
		},
		{
			name:  "character device",
			entry: tarEntry{name: "go/bin/tty", typeflag: tar.TypeChar},
			want:  "unsupported archive entry",
		},
		{
			name:  "fifo",
			entry: tarEntry{name: "go/bin/pipe", typeflag: tar.TypeFifo},
			want:  "unsupported archive entry",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			archive := writeTarGz(t, []tarEntry{
				{name: "go", typeflag: tar.TypeDir, mode: 0o755},
				{name: "go/bin", typeflag: tar.TypeDir, mode: 0o755},
				tc.entry,
			})
			dst := t.TempDir()
			err := extractGoTarGz(archive, dst)
			if err == nil {
				t.Fatal("extraction accepted an entry it must refuse")
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error = %v, want it to mention %q", err, tc.want)
			}
			// Nothing outside dst, and no link left behind inside it.
			if _, err := os.Lstat(filepath.Join(dst, "go", "bin", "gofmt")); err == nil {
				t.Fatal("a refused entry was still created")
			}
		})
	}
}

// Contract: a tar entry naming a path outside the destination is refused, so a
// tampered archive cannot write over the user's filesystem.
func TestExtractGoTarGz_RefusesEscapingNames(t *testing.T) {
	for _, name := range []string{
		"../escaped",
		"go/../../escaped",
		"/etc/passwd",
	} {
		t.Run(name, func(t *testing.T) {
			archive := writeTarGz(t, []tarEntry{
				{name: name, typeflag: tar.TypeReg, body: "owned", mode: 0o644},
			})
			dst := t.TempDir()
			outside := filepath.Join(filepath.Dir(dst), "escaped")
			if err := extractGoTarGz(archive, dst); err == nil {
				t.Fatal("extraction accepted an escaping path")
			}
			if _, err := os.Stat(outside); err == nil {
				t.Fatalf("archive wrote outside the destination: %s", outside)
			}
		})
	}
}

// Contract: the zip path — used on Windows, where Go ships a .zip — applies the
// same policy as the tarball path.
func TestExtractGoZip_AppliesTheSameEntryPolicy(t *testing.T) {
	writeZip := func(t *testing.T, build func(*zip.Writer)) string {
		t.Helper()
		path := filepath.Join(t.TempDir(), "fixture.zip")
		f, err := os.Create(path)
		if err != nil {
			t.Fatal(err)
		}
		zw := zip.NewWriter(f)
		build(zw)
		if err := zw.Close(); err != nil {
			t.Fatal(err)
		}
		if err := f.Close(); err != nil {
			t.Fatal(err)
		}
		return path
	}

	t.Run("dirs and files extract", func(t *testing.T) {
		archive := writeZip(t, func(zw *zip.Writer) {
			if _, err := zw.Create("go/bin/"); err != nil {
				t.Fatal(err)
			}
			h := &zip.FileHeader{Name: "go/bin/go.exe"}
			h.SetMode(0o755)
			w, err := zw.CreateHeader(h)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := w.Write([]byte("binary")); err != nil {
				t.Fatal(err)
			}
		})
		dst := t.TempDir()
		if err := extractGoZip(archive, dst); err != nil {
			t.Fatalf("extract: %v", err)
		}
		if body, err := os.ReadFile(filepath.Join(dst, "go", "bin", "go.exe")); err != nil || string(body) != "binary" {
			t.Fatalf("go.exe = %q, err = %v", body, err)
		}
	})

	t.Run("symlink refused", func(t *testing.T) {
		archive := writeZip(t, func(zw *zip.Writer) {
			h := &zip.FileHeader{Name: "go/bin/go"}
			h.SetMode(os.ModeSymlink | 0o777)
			w, err := zw.CreateHeader(h)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := w.Write([]byte("../../../../etc/passwd")); err != nil {
				t.Fatal(err)
			}
		})
		err := extractGoZip(archive, t.TempDir())
		if err == nil || !strings.Contains(err.Error(), "links are not allowed") {
			t.Fatalf("err = %v, want a refusal naming links", err)
		}
	})

	t.Run("escaping name refused", func(t *testing.T) {
		archive := writeZip(t, func(zw *zip.Writer) {
			w, err := zw.Create("../escaped")
			if err != nil {
				t.Fatal(err)
			}
			if _, err := w.Write([]byte("owned")); err != nil {
				t.Fatal(err)
			}
		})
		dst := t.TempDir()
		if err := extractGoZip(archive, dst); err == nil {
			t.Fatal("extraction accepted an escaping path")
		}
		if _, err := os.Stat(filepath.Join(filepath.Dir(dst), "escaped")); err == nil {
			t.Fatal("archive wrote outside the destination")
		}
	})
}

// Contract: provisioning degrades quietly when the machine cannot host a
// toolchain, so the caller keeps its actionable "install Go" guidance instead of
// surfacing a plumbing error the user cannot act on. These are the paths a
// locked-down or unusual environment actually takes.
func TestProvisionGoToolchain_DegradesWhenTheMachineCannotHostIt(t *testing.T) {
	goodArchive := goArchiveFixture(t, map[string]string{
		"go/bin/go": "#!/bin/sh\nexit 0\n",
	})
	sum := sha256.Sum256(goodArchive)
	checksum := hex.EncodeToString(sum[:])

	t.Run("no resolvable AgentField home", func(t *testing.T) {
		configureGoProvisionFixture(t, goodArchive, checksum)
		t.Setenv("AGENTFIELD_HOME", "")
		// os.UserHomeDir fails with no HOME, so there is nowhere to cache a
		// toolchain and provisioning must decline rather than error.
		t.Setenv("HOME", "")
		if runtime.GOOS == "windows" {
			t.Setenv("USERPROFILE", "")
		}
		goCmd, cached, err := provisionGoToolchain()
		if err != nil || goCmd != "" || cached {
			t.Fatalf("goCmd=%q cached=%v err=%v, want a quiet decline", goCmd, cached, err)
		}
	})

	t.Run("toolchains path is not a directory", func(t *testing.T) {
		configureGoProvisionFixture(t, goodArchive, checksum)
		home := t.TempDir()
		t.Setenv("AGENTFIELD_HOME", home)
		// A file where the toolchains directory belongs: the cache cannot be
		// created, so provisioning declines instead of failing the install with
		// a filesystem error.
		if err := os.WriteFile(filepath.Join(home, "toolchains"), []byte("not a dir"), 0o644); err != nil {
			t.Fatal(err)
		}
		goCmd, cached, err := provisionGoToolchain()
		if err != nil || goCmd != "" || cached {
			t.Fatalf("goCmd=%q cached=%v err=%v, want a quiet decline", goCmd, cached, err)
		}
	})

	t.Run("archive host is unreachable", func(t *testing.T) {
		// The index answers, so a release is selected — but the archive itself
		// cannot be fetched. Still a decline, not an error.
		filename := fmt.Sprintf("go9.9.9.%s-%s.tar.gz", runtime.GOOS, runtime.GOARCH)
		index := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			fmt.Fprintf(w, `[{"version":"go9.9.9","stable":true,"files":[{"filename":%q,"os":%q,"arch":%q,"kind":"archive","sha256":%q}]}]`,
				filename, runtime.GOOS, runtime.GOARCH, checksum)
		}))
		t.Cleanup(index.Close)
		dead := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
		deadURL := dead.URL
		dead.Close() // nothing is listening there now

		oldClient, oldIndex, oldBase := goProvisionHTTPClient, goProvisionIndexURL, goProvisionArchiveBaseURL
		goProvisionHTTPClient = index.Client()
		goProvisionIndexURL = index.URL
		goProvisionArchiveBaseURL = deadURL + "/"
		t.Cleanup(func() {
			goProvisionHTTPClient, goProvisionIndexURL, goProvisionArchiveBaseURL = oldClient, oldIndex, oldBase
		})
		t.Setenv("AGENTFIELD_HOME", t.TempDir())

		goCmd, cached, err := provisionGoToolchain()
		if err != nil || goCmd != "" || cached {
			t.Fatalf("goCmd=%q cached=%v err=%v, want a quiet decline", goCmd, cached, err)
		}
	})
}

// Contract: a download that fails its integrity check fails the install. It must
// NOT degrade into the "install Go yourself" message — that would tell the user
// their machine is missing a toolchain when what actually happened is that the
// one we fetched could not be trusted.
func TestResolveGoToolchain_SurfacesACorruptDownload(t *testing.T) {
	archive := goArchiveFixture(t, map[string]string{"go/bin/go": "#!/bin/sh\nexit 0\n"})
	configureGoProvisionFixture(t, archive, strings.Repeat("0", 64)) // wrong checksum
	t.Setenv("AGENTFIELD_HOME", t.TempDir())
	t.Setenv("PATH", t.TempDir()) // no ambient go

	_, err := resolveGoToolchain(t.TempDir())
	if err == nil {
		t.Fatal("a corrupt toolchain download must fail the install")
	}
	if !strings.Contains(err.Error(), "SHA256 mismatch") {
		t.Fatalf("error = %v, want the integrity failure, not generic install guidance", err)
	}
	if strings.Contains(err.Error(), "no `go` toolchain was found") {
		t.Fatal("integrity failure was masked as a missing toolchain")
	}
}

// Contract: an existing but unwritable toolchains directory declines rather than
// erroring — the user can still install Go themselves, which the caller's
// message tells them how to do.
func TestProvisionGoToolchain_DeclinesWhenTheCacheIsUnwritable(t *testing.T) {
	if os.Geteuid() == 0 {
		t.Skip("root ignores directory permissions")
	}
	archive := goArchiveFixture(t, map[string]string{"go/bin/go": "#!/bin/sh\nexit 0\n"})
	sum := sha256.Sum256(archive)
	configureGoProvisionFixture(t, archive, hex.EncodeToString(sum[:]))

	home := t.TempDir()
	t.Setenv("AGENTFIELD_HOME", home)
	toolchains := filepath.Join(home, "toolchains")
	if err := os.Mkdir(toolchains, 0o555); err != nil { // exists, but nothing can be created in it
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chmod(toolchains, 0o755) })

	goCmd, cached, err := provisionGoToolchain()
	if err != nil || goCmd != "" || cached {
		t.Fatalf("goCmd=%q cached=%v err=%v, want a quiet decline", goCmd, cached, err)
	}
}

// Contract: a download that dies partway through declines rather than erroring,
// and leaves nothing cached — the next install retries from scratch instead of
// finding a truncated archive masquerading as a toolchain.
func TestProvisionGoToolchain_DeclinesOnATruncatedDownload(t *testing.T) {
	filename := fmt.Sprintf("go9.9.9.%s-%s.tar.gz", runtime.GOOS, runtime.GOARCH)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/index") {
			fmt.Fprintf(w, `[{"version":"go9.9.9","stable":true,"files":[{"filename":%q,"os":%q,"arch":%q,"kind":"archive","sha256":%q}]}]`,
				filename, runtime.GOOS, runtime.GOARCH, strings.Repeat("a", 64))
			return
		}
		// Promise a long body, deliver a few bytes, then drop the connection.
		conn, buf, err := w.(http.Hijacker).Hijack()
		if err != nil {
			t.Errorf("hijack: %v", err)
			return
		}
		fmt.Fprint(buf, "HTTP/1.1 200 OK\r\nContent-Length: 1048576\r\n\r\ntruncated")
		_ = buf.Flush()
		_ = conn.Close()
	}))
	t.Cleanup(srv.Close)

	oldClient, oldIndex, oldBase := goProvisionHTTPClient, goProvisionIndexURL, goProvisionArchiveBaseURL
	goProvisionHTTPClient = srv.Client()
	goProvisionIndexURL = srv.URL + "/index"
	goProvisionArchiveBaseURL = srv.URL + "/"
	t.Cleanup(func() {
		goProvisionHTTPClient, goProvisionIndexURL, goProvisionArchiveBaseURL = oldClient, oldIndex, oldBase
	})

	home := t.TempDir()
	t.Setenv("AGENTFIELD_HOME", home)

	goCmd, cached, err := provisionGoToolchain()
	if err != nil || goCmd != "" || cached {
		t.Fatalf("goCmd=%q cached=%v err=%v, want a quiet decline", goCmd, cached, err)
	}
	if entries, _ := os.ReadDir(filepath.Join(home, "toolchains")); len(entries) != 0 {
		t.Fatalf("a truncated download left %d entries cached", len(entries))
	}
}

// Contract: a download size is rendered the way a person would say it, and an
// index that omits the size does not claim "0 B".
func TestHumanBytes(t *testing.T) {
	for _, tc := range []struct {
		in   int64
		want string
	}{
		{0, "unknown size"},
		{-1, "unknown size"},
		{512, "512 B"},
		{2048, "2 KB"},
		{66041589, "63 MB"},
		{5 * 1024 * 1024 * 1024, "5 GB"},
	} {
		if got := humanBytes(tc.in); got != tc.want {
			t.Errorf("humanBytes(%d) = %q, want %q", tc.in, got, tc.want)
		}
	}
}
