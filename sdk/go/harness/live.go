package harness

import "encoding/json"

// LiveInfo describes a live run in its started relay entry.
type LiveInfo struct {
	Harness   string `json:"harness"`
	Steerable bool   `json:"steerable"`
}

// LiveProvider implements live execution. When it receives Options.Live, it
// passes its LiveInfo to Begin once before streaming events and calls End once
// when the run ends. A steerable provider also reads notes from Steer.
type LiveProvider interface {
	Provider
	LiveInfo() LiveInfo
}

// Live carries a run's events and steer notes between a provider and its caller.
type Live struct {
	Event func(line json.RawMessage)
	Steer <-chan string
	Begin func(LiveInfo)
	End   func()
}
