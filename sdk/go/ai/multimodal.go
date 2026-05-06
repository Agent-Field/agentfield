package ai

import (
	"encoding/base64"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

// audioURLClient bounds remote audio fetches so a slow or oversized
// response can't hang the caller or exhaust memory.
var audioURLClient = &http.Client{Timeout: 30 * time.Second}

// maxAudioURLBytes caps the response body read by WithAudioURL.
// Declared as a var so tests can shrink it without streaming 50 MiB.
var maxAudioURLBytes int64 = 50 * 1024 * 1024

func detectMIMEType(path string) string {
	lower := strings.ToLower(path)
	switch {
	case strings.HasSuffix(lower, ".png"):
		return "image/png"
	case strings.HasSuffix(lower, ".jpg"), strings.HasSuffix(lower, ".jpeg"):
		return "image/jpeg"
	case strings.HasSuffix(lower, ".gif"):
		return "image/gif"
	case strings.HasSuffix(lower, ".webp"):
		return "image/webp"
	default:
		return "application/octet-stream"
	}
}

// WithAudioFile reads a local audio file, base64 encodes it, and attaches it to the request.
// The format string must be explicitly provided (e.g., "mp3", "wav").
func WithAudioFile(path string, mediaType string) Option{
	return func(r *Request) error{
		data, err := os.ReadFile(path)
		if err != nil {
			return fmt.Errorf("read audio file: %w", err)
		}

		encoded := base64.StdEncoding.EncodeToString(data)
		if len(r.Messages) == 0{
			r.Messages = append(r.Messages, Message{
				Role: "user",
				Content: []ContentPart{},
			})
		}

		last := &r.Messages[len(r.Messages)-1]
		last.Content = append(last.Content, ContentPart{
			Type: "input_audio",
			InputAudio: &InputAudioData{
				Data: encoded,
				Format: mediaType,
			},
		})

		return nil
	}
}

// WithAudioURL downloads an audio file from the provided URL, base64 encodes it,
// and attaches it to the request. The network connection is safely closed after reading.
// The format string must be explicitly provided (e.g., "mp3", "wav").
func WithAudioURL(url string, mediaType string) Option{
	return func(r* Request) error {
		response, err := audioURLClient.Get(url)
		if err != nil{
			return fmt.Errorf("fetch audio file: %w", err)
		}

		defer response.Body.Close()

		if response.StatusCode != 200{
			return fmt.Errorf("fetch audio file: HTTP %d", response.StatusCode)
		}

		cap := maxAudioURLBytes
		data, err := io.ReadAll(io.LimitReader(response.Body, cap+1))
		if err != nil{
			return fmt.Errorf("read audio response: %w", err)
		}
		if int64(len(data)) > cap {
			return fmt.Errorf("fetch audio file: response exceeds %d bytes", cap)
		}

		encoded := base64.StdEncoding.EncodeToString(data)
		if len(r.Messages) == 0 {
			r.Messages = append(r.Messages, Message{
				Role: "user",
				Content: []ContentPart{},
			})
		}

		last := &r.Messages[len(r.Messages)-1]
		last.Content = append(last.Content, ContentPart{
			Type: "input_audio",
			InputAudio: &InputAudioData{
				Data: encoded,
				Format: mediaType,
			},
		})

		return nil
	}
}

// WithFile attaches a generic file to the request and base64 encodes it.
// The mediaType must be explicitly provided by the caller.
// 
// Common supported MediaTypes for AI models include:
//   - "application/pdf" (PDF documents)
//   - "text/plain" (Standard text files)
//   - "text/csv" (CSV data)
//   - "text/markdown" (Markdown files)
//   - "application/json" (JSON files)
//   - "text/html" (HTML documents)
//   - "application/vnd.openxmlformats-officedocument.wordprocessingml.document" (Word .docx)
//   - "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" (Excel .xlsx)
func WithFile(path string, mediaType string) Option{
	return func(r *Request) error{
		data, err := os.ReadFile(path)
		if err!= nil{
			return fmt.Errorf("read file: %w", err)
		}
		
		encoded := base64.StdEncoding.EncodeToString(data)
		fileData := fmt.Sprintf("data:%s;base64,%s", mediaType, encoded)

		if len(r.Messages) == 0{
			r.Messages = append(r.Messages, Message{
				Role: "user",
				Content: []ContentPart{},
			})
		}

		last := &r.Messages[len(r.Messages)-1]
		last.Content = append(last.Content, ContentPart{
			Type: "file",
			InputFile: &InputFileData{
				FileData: fileData,
			},
		})

		return nil;
	}
}