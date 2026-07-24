package inputs

import (
	"encoding/json"
	"testing"
)

func TestRequiredString(t *testing.T) {
	tests := []struct {
		name    string
		input   map[string]any
		key     string
		want    string
		wantErr bool
	}{
		{
			name:    "present and non-empty",
			input:   map[string]any{"name": "test"},
			key:     "name",
			want:    "test",
			wantErr: false,
		},
		{
			name:    "present with whitespace",
			input:   map[string]any{"name": "  test  "},
			key:     "name",
			want:    "test",
			wantErr: false,
		},
		{
			name:    "missing key",
			input:   map[string]any{},
			key:     "name",
			want:    "",
			wantErr: true,
		},
		{
			name:    "empty string",
			input:   map[string]any{"name": ""},
			key:     "name",
			want:    "",
			wantErr: true,
		},
		{
			name:    "only whitespace",
			input:   map[string]any{"name": "   "},
			key:     "name",
			want:    "",
			wantErr: true,
		},
		{
			name:    "nil input",
			input:   nil,
			key:     "name",
			want:    "",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := RequiredString(tt.input, tt.key)
			if (err != nil) != tt.wantErr {
				t.Errorf("RequiredString() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("RequiredString() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestString(t *testing.T) {
	tests := []struct {
		name  string
		input map[string]any
		key   string
		want  string
	}{
		{
			name:  "present string",
			input: map[string]any{"name": "test"},
			key:   "name",
			want:  "test",
		},
		{
			name:  "present with whitespace",
			input: map[string]any{"name": "  test  "},
			key:   "name",
			want:  "test",
		},
		{
			name:  "missing key",
			input: map[string]any{},
			key:   "name",
			want:  "",
		},
		{
			name:  "nil input",
			input: nil,
			key:   "name",
			want:  "",
		},
		{
			name:  "non-string value",
			input: map[string]any{"name": 42},
			key:   "name",
			want:  "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := String(tt.input, tt.key)
			if got != tt.want {
				t.Errorf("String() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestInt(t *testing.T) {
	tests := []struct {
		name  string
		input map[string]any
		key   string
		want  int
	}{
		{
			name:  "native int",
			input: map[string]any{"count": 42},
			key:   "count",
			want:  42,
		},
		{
			name:  "float64 from JSON",
			input: map[string]any{"count": float64(42.0)},
			key:   "count",
			want:  42,
		},
		{
			name:  "float64 truncated",
			input: map[string]any{"count": float64(42.7)},
			key:   "count",
			want:  42,
		},
		{
			name: "json.Number",
			input: func() map[string]any {
				d := json.NewDecoder(nil)
				d.UseNumber()
				return map[string]any{"count": json.Number("42")}
			}(),
			key:  "count",
			want: 42,
		},
		{
			name:  "missing key",
			input: map[string]any{},
			key:   "count",
			want:  0,
		},
		{
			name:  "nil input",
			input: nil,
			key:   "count",
			want:  0,
		},
		{
			name:  "non-numeric value",
			input: map[string]any{"count": "abc"},
			key:   "count",
			want:  0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Int(tt.input, tt.key)
			if got != tt.want {
				t.Errorf("Int() = %d, want %d", got, tt.want)
			}
		})
	}
}

func TestObject(t *testing.T) {
	tests := []struct {
		name    string
		input   map[string]any
		key     string
		want    map[string]any
		wantErr bool
	}{
		{
			name:    "present object",
			input:   map[string]any{"config": map[string]any{"key": "value"}},
			key:     "config",
			want:    map[string]any{"key": "value"},
			wantErr: false,
		},
		{
			name:    "missing key",
			input:   map[string]any{},
			key:     "config",
			want:    nil,
			wantErr: true,
		},
		{
			name:    "empty object",
			input:   map[string]any{"config": map[string]any{}},
			key:     "config",
			want:    nil,
			wantErr: true,
		},
		{
			name:    "nil input",
			input:   nil,
			key:     "config",
			want:    nil,
			wantErr: true,
		},
		{
			name:    "non-object value",
			input:   map[string]any{"config": "not an object"},
			key:     "config",
			want:    nil,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Object(tt.input, tt.key)
			if (err != nil) != tt.wantErr {
				t.Errorf("Object() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if tt.wantErr {
				if got != nil {
					t.Errorf("Object() = %v, want nil", got)
				}
			} else {
				if got["key"] != tt.want["key"] {
					t.Errorf("Object() = %v, want %v", got, tt.want)
				}
			}
		})
	}
}

func TestFirstNonBlank(t *testing.T) {
	tests := []struct {
		name   string
		values []string
		want   string
	}{
		{
			name:   "first is non-blank",
			values: []string{"first", "second"},
			want:   "first",
		},
		{
			name:   "second is non-blank",
			values: []string{"", "second", "third"},
			want:   "second",
		},
		{
			name:   "first with whitespace",
			values: []string{"  first  ", "second"},
			want:   "first",
		},
		{
			name:   "all empty",
			values: []string{"", "", ""},
			want:   "",
		},
		{
			name:   "no values",
			values: []string{},
			want:   "",
		},
		{
			name:   "single non-blank",
			values: []string{"only"},
			want:   "only",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := FirstNonBlank(tt.values...)
			if got != tt.want {
				t.Errorf("FirstNonBlank() = %q, want %q", got, tt.want)
			}
		})
	}
}
