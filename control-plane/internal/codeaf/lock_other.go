//go:build !unix

package codeaf

import "context"

func lockCodeaf(context.Context, string) (func() error, error) {
	return func() error { return nil }, nil
}
