package cron

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/sources"
)

func TestCron_MetadataAndSchema(t *testing.T) {
	s := &source{}
	if s.Name() != "cron" {
		t.Fatalf("Name() = %q, want cron", s.Name())
	}
	if s.Kind() != sources.KindLoop {
		t.Fatalf("Kind() = %v, want loop", s.Kind())
	}
	if s.SecretRequired() {
		t.Fatal("cron must not require a secret")
	}
	var schema map[string]any
	if err := json.Unmarshal(s.ConfigSchema(), &schema); err != nil {
		t.Fatalf("schema is not valid JSON: %v", err)
	}
}

func TestParseExpression_Basic(t *testing.T) {
	cases := []struct {
		expr    string
		wantErr bool
	}{
		{"* * * * *", false},
		{"*/5 * * * *", false},
		{"0 0 * * 0", false},
		{"0 9-17 * * 1-5", false},
		{"60 * * * *", true},
		{"a b c d e", true},
		{"* * * *", true},
	}
	for _, c := range cases {
		_, err := parseExpression(c.expr)
		gotErr := err != nil
		if gotErr != c.wantErr {
			t.Errorf("parseExpression(%q): wantErr=%v gotErr=%v (err=%v)", c.expr, c.wantErr, gotErr, err)
		}
	}
}

func TestSchedule_NextEveryFiveMinutes(t *testing.T) {
	s, err := parseExpression("*/5 * * * *")
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	loc := time.UTC
	now := time.Date(2026, 4, 27, 12, 7, 30, 0, loc)
	next := s.Next(now)
	want := time.Date(2026, 4, 27, 12, 10, 0, 0, loc)
	if !next.Equal(want) {
		t.Fatalf("Next(%v) = %v, want %v", now, next, want)
	}
}

func TestSchedule_NextHourBoundary(t *testing.T) {
	s, err := parseExpression("0 * * * *")
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	now := time.Date(2026, 4, 27, 12, 30, 0, 0, time.UTC)
	want := time.Date(2026, 4, 27, 13, 0, 0, 0, time.UTC)
	if got := s.Next(now); !got.Equal(want) {
		t.Fatalf("Next = %v, want %v", got, want)
	}
}

func TestSchedule_NextWeekday(t *testing.T) {
	// 9am Mon-Fri.
	s, err := parseExpression("0 9 * * 1-5")
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	// Saturday 10am UTC -> next fire is Monday 9am.
	sat := time.Date(2026, 4, 25, 10, 0, 0, 0, time.UTC) // Apr 25 2026 is a Saturday
	want := time.Date(2026, 4, 27, 9, 0, 0, 0, time.UTC)
	if got := s.Next(sat); !got.Equal(want) {
		t.Fatalf("Next = %v, want %v", got, want)
	}
}

func TestSchedule_NextSkipsBadMonth(t *testing.T) {
	// Only May.
	s, err := parseExpression("0 0 1 5 *")
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	// April -> next is May 1 00:00.
	apr := time.Date(2026, 4, 27, 12, 0, 0, 0, time.UTC)
	want := time.Date(2026, 5, 1, 0, 0, 0, 0, time.UTC)
	if got := s.Next(apr); !got.Equal(want) {
		t.Fatalf("Next = %v, want %v", got, want)
	}
}

func TestParseExpression_StepWithRange(t *testing.T) {
	// Every 2 minutes within first 10 minutes of every hour.
	s, err := parseExpression("0-10/2 * * * *")
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	// At 12:01 -> next match is 12:02.
	now := time.Date(2026, 4, 27, 12, 1, 30, 0, time.UTC)
	want := time.Date(2026, 4, 27, 12, 2, 0, 0, time.UTC)
	if got := s.Next(now); !got.Equal(want) {
		t.Fatalf("Next = %v, want %v", got, want)
	}
}

func TestParseConfig_DefaultsTimezoneAndRequiresExpression(t *testing.T) {
	if _, err := parseConfig([]byte(`{}`)); err == nil {
		t.Fatal("expected error for missing expression")
	}
	c, err := parseConfig([]byte(`{"expression":"* * * * *"}`))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if c.Timezone != "UTC" {
		t.Errorf("default timezone = %q, want UTC", c.Timezone)
	}
}

func TestValidate_RejectsBadTimezone(t *testing.T) {
	err := (&source{}).Validate([]byte(`{"expression":"* * * * *","timezone":"Mars/Olympus_Mons"}`))
	if err == nil {
		t.Fatal("expected error for bogus timezone")
	}
}

func TestCronValidateAndRunErrorBranches(t *testing.T) {
	s := &source{}
	if err := s.Validate([]byte(`{`)); err == nil || !strings.Contains(err.Error(), "invalid config") {
		t.Fatalf("expected invalid config error, got %v", err)
	}
	if err := s.Validate([]byte(`{"expression":"* * *"}`)); err == nil || !strings.Contains(err.Error(), "expected 5 fields") {
		t.Fatalf("expected expression error, got %v", err)
	}

	ctx := context.Background()
	if err := s.Run(ctx, []byte(`{`), "", func(sources.Event) {}); err == nil || !strings.Contains(err.Error(), "invalid config") {
		t.Fatalf("expected run config error, got %v", err)
	}
	if err := s.Run(ctx, []byte(`{"expression":"bad"}`), "", func(sources.Event) {}); err == nil || !strings.Contains(err.Error(), "expected 5 fields") {
		t.Fatalf("expected run expression error, got %v", err)
	}
	if err := s.Run(ctx, []byte(`{"expression":"* * * * *","timezone":"Mars/Olympus_Mons"}`), "", func(sources.Event) {}); err == nil {
		t.Fatal("expected run timezone error")
	}

	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	err := s.Run(cancelled, []byte(`{"expression":"* * * * *"}`), "", func(sources.Event) {
		t.Fatal("cancelled cron source should not emit")
	})
	if err == nil || !strings.Contains(err.Error(), "context canceled") {
		t.Fatalf("expected context canceled, got %v", err)
	}
}

func TestParseFieldRejectsMalformedParts(t *testing.T) {
	for _, field := range []string{"*/0", "*/nope", "5-2", "x-y", "x", "61"} {
		if _, err := parseField(field, 0, 59); err == nil {
			t.Fatalf("parseField(%q) expected error", field)
		}
	}
}

func TestScheduleNextUsesCronDaySemantics(t *testing.T) {
	// Restricted DOM and DOW use standard OR semantics. This should match
	// Wednesday April 29 because the day-of-month is 29 even though DOW asks
	// for Sunday.
	s, err := parseExpression("0 8 29 * 0")
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	now := time.Date(2026, 4, 28, 9, 0, 0, 0, time.UTC)
	want := time.Date(2026, 4, 29, 8, 0, 0, 0, time.UTC)
	if got := s.Next(now); !got.Equal(want) {
		t.Fatalf("Next = %v, want %v", got, want)
	}
}
