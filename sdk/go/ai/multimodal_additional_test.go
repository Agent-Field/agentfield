package ai

import "testing"

func TestDetectMIMEType(t *testing.T) {
	tests := []struct {
		path string
		want string
	}{
		{path: "image.png", want: "image/png"},
		{path: "image.JPG", want: "image/jpeg"},
		{path: "image.jpeg", want: "image/jpeg"},
		{path: "image.gif", want: "image/gif"},
		{path: "image.webp", want: "image/webp"},
		{path: "image.bin", want: "application/octet-stream"},
	}

	for _, tt := range tests {
		if got := detectMIMEType(tt.path); got != tt.want {
			t.Fatalf("detectMIMEType(%q) = %q, want %q", tt.path, got, tt.want)
		}
	}
}
