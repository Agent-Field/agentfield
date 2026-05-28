package storage

import (
	"testing"
	"time"
)

func TestParseTimeString(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		wantErr bool
	}{
		{
			name:    "PostgreSQL timestamp with timezone and microseconds",
			input:   "2025-11-11 00:00:15.535675+00:00",
			wantErr: false,
		},
		{
			name:    "PostgreSQL timestamp with negative timezone offset",
			input:   "2025-11-10 21:30:59.083244-05:00",
			wantErr: false,
		},
		{
			name:    "PostgreSQL timestamp without microseconds",
			input:   "2025-11-11 00:00:15+00:00",
			wantErr: false,
		},
		{
			name:    "RFC3339 format",
			input:   "2025-11-11T00:00:15.535675Z",
			wantErr: false,
		},
		{
			name:    "SQLite format without timezone",
			input:   "2025-11-11 00:00:15.535675",
			wantErr: false,
		},
		{
			name:    "Empty string",
			input:   "",
			wantErr: false,
		},
		{
			name:    "Invalid format",
			input:   "not-a-timestamp",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := parseTimeString(tt.input)
			if (err != nil) != tt.wantErr {
				t.Errorf("parseTimeString() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && tt.input != "" {
				if result.IsZero() {
					t.Errorf("parseTimeString() returned zero time for valid input %q", tt.input)
				}
				// Verify the result is in UTC
				if result.Location() != time.UTC {
					t.Errorf("parseTimeString() returned non-UTC time: %v", result.Location())
				}
			}
		})
	}
}

func TestParseDBTime(t *testing.T) {
	now := time.Now().UTC()

	tests := []struct {
		name    string
		input   interface{}
		wantErr bool
	}{
		{
			name:    "time.Time value",
			input:   now,
			wantErr: false,
		},
		{
			name:    "PostgreSQL string timestamp",
			input:   "2025-11-11 00:00:15.535675+00:00",
			wantErr: false,
		},
		{
			name:    "Byte slice timestamp",
			input:   []byte("2025-11-11 00:00:15.535675+00:00"),
			wantErr: false,
		},
		{
			name:    "nil value",
			input:   nil,
			wantErr: false,
		},
		{
			name:    "Unsupported type",
			input:   123,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := parseDBTime(tt.input)
			if (err != nil) != tt.wantErr {
				t.Errorf("parseDBTime() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && tt.input != nil {
				if result.IsZero() && tt.name != "nil value" {
					t.Errorf("parseDBTime() returned zero time for valid input")
				}
			}
		})
	}
}
