//go:build !unix

package aforge

func lockAforge(string) (func() error, error) {
	return func() error { return nil }, nil
}
