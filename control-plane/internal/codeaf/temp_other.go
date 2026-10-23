//go:build !unix

package codeaf

import "os"

func trustTempDirectory(_ string, _ os.FileInfo) error { return nil }
