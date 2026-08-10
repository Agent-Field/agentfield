//go:build !unix

package furrow

func lockFurrow(string) (func() error, error) {
	return func() error { return nil }, nil
}
