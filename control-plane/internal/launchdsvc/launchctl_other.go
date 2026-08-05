//go:build !darwin

package launchdsvc

import "errors"

// ErrUnsupported is returned by every mutating helper off macOS. launchd is a
// macOS facility; the CLI surfaces this as a clear "macOS-only" message rather
// than pretending to manage a service.
var ErrUnsupported = errors.New("service management is macOS-only")

func Supported() bool { return false }

func GUIDomain() string         { return "" }
func SvcTarget(l string) string { return l }

func KickstartArgs(label string, kill bool) []string { return nil }

func Bootstrap(plistPath string) error        { return ErrUnsupported }
func Bootout(label string) error              { return ErrUnsupported }
func Kickstart(label string, kill bool) error { return ErrUnsupported }
func SignalAgent(label, signal string) error  { return ErrUnsupported }

func AgentLoaded(label string) bool { return false }

func Reload(plistPath, label string) {}
