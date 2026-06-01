// Package cron implements a schedule-based LoopSource. Each trigger holds a
// 5-field cron expression (minute hour day-of-month month day-of-week); the
// Source ticks once per scheduled fire and emits a synthetic event of type
// "tick" to the dispatcher.
//
// The expression parser supports the common subset: '*', integers, ranges
// (a-b), lists (a,b,c), and steps ('*/n' or 'a-b/n'). It does NOT support
// names, special strings ('@hourly', '@daily'), seconds, or year fields.
// This subset is sufficient for first-party use and avoids pulling in an
// external cron dependency.
package cron

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/sources"
)

type source struct{}

func init() {
	sources.Register(&source{})
}

func (s *source) Name() string         { return "cron" }
func (s *source) Kind() sources.Kind   { return sources.KindLoop }
func (s *source) SecretRequired() bool { return false }

func (s *source) ConfigSchema() json.RawMessage {
	return json.RawMessage(`{
        "type":"object",
        "properties":{
          "expression":{"type":"string","description":"5-field cron expression (minute hour dom month dow)"},
          "timezone":{"type":"string","default":"UTC","description":"IANA timezone name"}
        },
        "required":["expression"],
        "additionalProperties": false
    }`)
}

type config struct {
	Expression string `json:"expression"`
	Timezone   string `json:"timezone"`
}

func parseConfig(cfg json.RawMessage) (config, error) {
	var c config
	if err := json.Unmarshal(cfg, &c); err != nil {
		return c, fmt.Errorf("cron: invalid config: %w", err)
	}
	if c.Expression == "" {
		return c, errors.New("cron: expression is required")
	}
	if c.Timezone == "" {
		c.Timezone = "UTC"
	}
	return c, nil
}

func (s *source) Validate(cfg json.RawMessage) error {
	c, err := parseConfig(cfg)
	if err != nil {
		return err
	}
	if _, err := parseExpression(c.Expression); err != nil {
		return err
	}
	if _, err := time.LoadLocation(c.Timezone); err != nil {
		return fmt.Errorf("cron: invalid timezone %q: %w", c.Timezone, err)
	}
	return nil
}

func (s *source) Run(ctx context.Context, cfg json.RawMessage, secret string, emit func(sources.Event)) error {
	c, err := parseConfig(cfg)
	if err != nil {
		return err
	}
	schedule, err := parseExpression(c.Expression)
	if err != nil {
		return err
	}
	loc, err := time.LoadLocation(c.Timezone)
	if err != nil {
		return err
	}

	for {
		now := time.Now().In(loc)
		next := schedule.Next(now)
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(time.Until(next)):
			// Use the actual scheduled minute as the idempotency key so a single
			// fire is only persisted once even if the manager restarts mid-tick.
			key := fmt.Sprintf("%s@%s", c.Expression, next.UTC().Format("2006-01-02T15:04Z"))
			normalized, _ := json.Marshal(map[string]any{
				"fired_at":   next.UTC().Format(time.RFC3339),
				"expression": c.Expression,
				"timezone":   c.Timezone,
			})
			emit(sources.Event{
				Type:           "tick",
				IdempotencyKey: key,
				Raw:            normalized,
				Normalized:     normalized,
			})
		}
	}
}

// schedule is the parsed cron expression. Each field is a bitmask of allowed
// values. Next() walks forward until all fields match.
type schedule struct {
	minute    uint64 // 0-59
	hour      uint64 // 0-23
	dom       uint64 // 1-31
	month     uint64 // 1-12
	dow       uint64 // 0-6 (Sunday=0)
	domStar   bool   // true when dom field is unrestricted '*'
	dowStar   bool   // true when dow field is unrestricted '*'
}

func parseExpression(expr string) (*schedule, error) {
	fields := strings.Fields(expr)
	if len(fields) != 5 {
		return nil, fmt.Errorf("cron: expected 5 fields, got %d", len(fields))
	}
	s := &schedule{}
	var err error
	if s.minute, err = parseField(fields[0], 0, 59); err != nil {
		return nil, fmt.Errorf("cron: minute: %w", err)
	}
	if s.hour, err = parseField(fields[1], 0, 23); err != nil {
		return nil, fmt.Errorf("cron: hour: %w", err)
	}
	if s.dom, err = parseField(fields[2], 1, 31); err != nil {
		return nil, fmt.Errorf("cron: day-of-month: %w", err)
	}
	if s.month, err = parseField(fields[3], 1, 12); err != nil {
		return nil, fmt.Errorf("cron: month: %w", err)
	}
	if s.dow, err = parseField(fields[4], 0, 6); err != nil {
		return nil, fmt.Errorf("cron: day-of-week: %w", err)
	}
	s.domStar = fields[2] == "*"
	s.dowStar = fields[4] == "*"
	return s, nil
}

func parseField(field string, min, max int) (uint64, error) {
	var mask uint64
	for _, part := range strings.Split(field, ",") {
		step := 1
		if idx := strings.Index(part, "/"); idx >= 0 {
			s, err := strconv.Atoi(part[idx+1:])
			if err != nil || s <= 0 {
				return 0, fmt.Errorf("invalid step in %q", part)
			}
			step = s
			part = part[:idx]
		}
		var lo, hi int
		switch {
		case part == "*":
			lo, hi = min, max
		case strings.Contains(part, "-"):
			ab := strings.SplitN(part, "-", 2)
			a, err1 := strconv.Atoi(ab[0])
			b, err2 := strconv.Atoi(ab[1])
			if err1 != nil || err2 != nil {
				return 0, fmt.Errorf("invalid range %q", part)
			}
			lo, hi = a, b
		default:
			v, err := strconv.Atoi(part)
			if err != nil {
				return 0, fmt.Errorf("invalid value %q", part)
			}
			lo, hi = v, v
		}
		if lo < min || hi > max || lo > hi {
			return 0, fmt.Errorf("value %d-%d out of range %d-%d", lo, hi, min, max)
		}
		for v := lo; v <= hi; v += step {
			mask |= 1 << uint(v)
		}
	}
	return mask, nil
}

func (s *schedule) Next(after time.Time) time.Time {
	// Move to the next minute boundary (cron doesn't fire at second granularity).
	t := after.Add(time.Minute - time.Duration(after.Second())*time.Second - time.Duration(after.Nanosecond())*time.Nanosecond)

	// Cap iterations so a malformed schedule cannot spin forever.
	for i := 0; i < 366*24*60; i++ {
		if s.month&(1<<uint(t.Month())) == 0 {
			t = time.Date(t.Year(), t.Month()+1, 1, 0, 0, 0, 0, t.Location())
			continue
		}
		domMatch := s.dom&(1<<uint(t.Day())) != 0
		dowMatch := s.dow&(1<<uint(t.Weekday())) != 0
		var dayOK bool
		switch {
		case s.domStar && s.dowStar:
			dayOK = true
		case s.domStar:
			dayOK = dowMatch
		case s.dowStar:
			dayOK = domMatch
		default:
			// Standard cron OR-semantics when both fields are restricted.
			dayOK = domMatch || dowMatch
		}
		if !dayOK {
			t = time.Date(t.Year(), t.Month(), t.Day()+1, 0, 0, 0, 0, t.Location())
			continue
		}
		if s.hour&(1<<uint(t.Hour())) == 0 {
			t = time.Date(t.Year(), t.Month(), t.Day(), t.Hour()+1, 0, 0, 0, t.Location())
			continue
		}
		if s.minute&(1<<uint(t.Minute())) == 0 {
			t = t.Add(time.Minute)
			continue
		}
		return t
	}
	// Fallback: schedule never matches. Return far future to avoid hot loop.
	return after.Add(24 * time.Hour)
}
