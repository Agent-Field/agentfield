package workspace

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// CAS is a local content-addressed store. Blobs are whole files keyed by the
// hex sha256 of their content and laid out sharded by the first two hex digits
// (cas/ab/cdef...) to keep directory sizes manageable.
//
// POC note: the CLI and the control plane run on the same machine and share
// ~/.agentfield. The CLI seals directly into the same CAS that the control
// plane reads when uploading blobs to a node, so no blob copy happens between
// them — they are literally the same directory on disk.
type CAS struct {
	root string
}

// NewCAS returns a CAS rooted at the given directory. The directory is created
// lazily on the first write.
func NewCAS(root string) *CAS {
	return &CAS{root: root}
}

// Root returns the CAS root directory.
func (c *CAS) Root() string { return c.root }

func (c *CAS) blobPath(sha string) string {
	if len(sha) < 2 {
		return filepath.Join(c.root, sha)
	}
	return filepath.Join(c.root, sha[:2], sha[2:])
}

// Has reports whether a blob with the given sha256 is present.
func (c *CAS) Has(sha string) bool {
	if sha == "" {
		return false
	}
	_, err := os.Stat(c.blobPath(sha))
	return err == nil
}

// Get returns the bytes of a stored blob.
func (c *CAS) Get(sha string) ([]byte, error) {
	if sha == "" {
		return nil, fmt.Errorf("empty sha")
	}
	return os.ReadFile(c.blobPath(sha))
}

// Put stores content under the given sha256. The write is atomic (temp file plus
// rename) and idempotent: an existing identical blob is left untouched.
func (c *CAS) Put(sha string, data []byte) error {
	if sha == "" {
		return fmt.Errorf("empty sha")
	}
	if c.Has(sha) {
		return nil
	}
	dst := c.blobPath(sha)
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return fmt.Errorf("create cas shard: %w", err)
	}
	tmp, err := os.CreateTemp(filepath.Dir(dst), ".tmp-"+sha[:min(len(sha), 8)]+"-*")
	if err != nil {
		return fmt.Errorf("create cas temp: %w", err)
	}
	tmpName := tmp.Name()
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		_ = os.Remove(tmpName)
		return fmt.Errorf("write cas temp: %w", err)
	}
	if err := tmp.Close(); err != nil {
		_ = os.Remove(tmpName)
		return fmt.Errorf("close cas temp: %w", err)
	}
	if err := os.Rename(tmpName, dst); err != nil {
		_ = os.Remove(tmpName)
		return fmt.Errorf("commit cas blob: %w", err)
	}
	return nil
}

// PutVerified stores content only if its sha256 matches the expected key,
// rejecting content that does not hash to the claimed address. Used on the
// receiving side of a blob upload.
func (c *CAS) PutVerified(sha string, data []byte) error {
	actual := hashBytes(data)
	if actual != sha {
		return fmt.Errorf("blob content hash %s does not match expected %s", actual, sha)
	}
	return c.Put(sha, data)
}

// PutBytes hashes content, stores it, and returns the sha256 key.
func (c *CAS) PutBytes(data []byte) (string, error) {
	sha := hashBytes(data)
	if err := c.Put(sha, data); err != nil {
		return "", err
	}
	return sha, nil
}

// DefaultCASDir returns the shared content store directory (~/.agentfield/cas),
// honoring AGENTFIELD_HOME the same way the rest of the CLI does.
func DefaultCASDir() string {
	return filepath.Join(HomeDir(), "cas")
}

// HomeDir returns the AgentField home directory, honoring AGENTFIELD_HOME and
// falling back to ~/.agentfield.
func HomeDir() string {
	if custom := strings.TrimSpace(os.Getenv("AGENTFIELD_HOME")); custom != "" {
		return custom
	}
	home, err := os.UserHomeDir()
	if err != nil || strings.TrimSpace(home) == "" {
		// Last-resort fallback keeps sealing usable in constrained environments.
		return filepath.Join(os.TempDir(), ".agentfield")
	}
	return filepath.Join(home, ".agentfield")
}
