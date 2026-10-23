//go:build !unix

package codeaf

func lockCodeaf(string) (func() error, error) {
	return func() error { return nil }, nil
}
