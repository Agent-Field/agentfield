package ai

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestWithAudioFile(t *testing.T) {
	tempFile, err := os.CreateTemp("", "test_audio_*.mp3")
	assert.NoError(t, err)
	defer os.Remove(tempFile.Name())

	dummyData := []byte("audio-data")
	_, err = tempFile.Write(dummyData)
	assert.NoError(t, err)
	tempFile.Close()

	req := &Request{}
	err = WithAudioFile(tempFile.Name(), "mp3")(req)

	assert.NoError(t, err)

	assert.Len(t, req.Messages, 1)
	assert.Len(t, req.Messages[0].Content, 1)

	part := req.Messages[0].Content[0]

	assert.Equal(t, "input_audio", part.Type)
	assert.NotNil(t, part.InputAudio)

	assert.Equal(t, "mp3", part.InputAudio.Format)

	expectedBase64 := base64.StdEncoding.EncodeToString(dummyData)
	assert.Equal(t, expectedBase64, part.InputAudio.Data)

	// Validate JSON serialization 
    jsonData, err := json.Marshal(req)
    assert.NoError(t, err)

    var parsed map[string]interface{}
    err = json.Unmarshal(jsonData, &parsed)
    assert.NoError(t, err)

    messages := parsed["messages"].([]interface{})
    msg := messages[0].(map[string]interface{})
    content := msg["content"].([]interface{})
    contentPart := content[0].(map[string]interface{})

    assert.Equal(t, "input_audio", contentPart["type"])
    inputAudio := contentPart["input_audio"].(map[string]interface{})
    assert.NotNil(t, inputAudio["data"])
    assert.Equal(t, "mp3", inputAudio["format"])
}

func TestWithAudioURL(t *testing.T) {
	dummyData := []byte("downloaded-audio")

	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write(dummyData)
	}))

	defer mockServer.Close()

	req := &Request{}

	err := WithAudioURL(mockServer.URL, "wav")(req)

	assert.NoError(t, err)
	assert.Len(t, req.Messages, 1)
	assert.Len(t, req.Messages[0].Content, 1)

	part := req.Messages[0].Content[0]
	assert.Equal(t, "input_audio", part.Type)
	assert.NotNil(t, part.InputAudio)
	assert.Equal(t, "wav", part.InputAudio.Format)

	expectedBase64 := base64.StdEncoding.EncodeToString(dummyData)
	assert.Equal(t, expectedBase64, part.InputAudio.Data)

	// Validate JSON serialization 
    jsonData, err := json.Marshal(req)
    assert.NoError(t, err)

    var parsed map[string]interface{}
    err = json.Unmarshal(jsonData, &parsed)
    assert.NoError(t, err)

    messages := parsed["messages"].([]interface{})
    msg := messages[0].(map[string]interface{})
    content := msg["content"].([]interface{})
    contentPart := content[0].(map[string]interface{})

    assert.Equal(t, "input_audio", contentPart["type"])
    inputAudio := contentPart["input_audio"].(map[string]interface{})
    assert.NotNil(t, inputAudio["data"])
    assert.Equal(t, "wav", inputAudio["format"])
}

func TestWithFile(t *testing.T) {
    testCases := []struct {
        filename string
        mimeType string
    }{
        {"report.pdf", "application/pdf"},
        {"document.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"},
        {"data.csv", "text/csv"},
        {"config.json", "application/json"},
        {"notes.txt", "text/plain"},
		{"file.html", "text/html"},
    }

    for _, tc := range testCases {
        t.Run(tc.filename, func(t *testing.T) {
            tempFile, err := os.CreateTemp("", tc.filename)
            assert.NoError(t, err)
            defer os.Remove(tempFile.Name())

			_, err = tempFile.Write([]byte("test-data"))
			assert.NoError(t, err)            
			tempFile.Close()

            req := &Request{}
            err = WithFile(tempFile.Name(), tc.mimeType)(req)

            assert.NoError(t, err)
            assert.Len(t, req.Messages, 1)
            assert.Len(t, req.Messages[0].Content, 1)

            part := req.Messages[0].Content[0]
            assert.Equal(t, "file", part.Type)
            assert.NotNil(t, part.InputFile)

            expectedBase64 := base64.StdEncoding.EncodeToString([]byte("test-data"))
            expectedFileData := fmt.Sprintf("data:%s;base64,%s", tc.mimeType, expectedBase64)
            assert.Equal(t, expectedFileData, part.InputFile.FileData)

			// Validate JSON serialization 
            jsonData, err := json.Marshal(req)
            assert.NoError(t, err)

            var parsed map[string]interface{}
            err = json.Unmarshal(jsonData, &parsed)
            assert.NoError(t, err)

            messages := parsed["messages"].([]interface{})
            msg := messages[0].(map[string]interface{})
            content := msg["content"].([]interface{})
            contentPart := content[0].(map[string]interface{})

            assert.Equal(t, "file", contentPart["type"])
            file := contentPart["file"].(map[string]interface{})
            assert.Contains(t, file["file_data"].(string), fmt.Sprintf("data:%s;base64,", tc.mimeType))
        })
    }
}

func TestWithAudioFile_ReadError(t *testing.T) {
	req := &Request{}
	err := WithAudioFile("/nonexistent/path/to/audio.mp3", "mp3")(req)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "read audio file")
	assert.Empty(t, req.Messages)
}

func TestWithAudioURL_FetchError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	url := server.URL
	server.Close()

	req := &Request{}
	err := WithAudioURL(url, "mp3")(req)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "fetch audio file")
	assert.Empty(t, req.Messages)
}

func TestWithAudioURL_HTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer server.Close()

	req := &Request{}
	err := WithAudioURL(server.URL, "mp3")(req)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "HTTP 500")
	assert.Empty(t, req.Messages)
}

func TestWithAudioURL_ExceedsSizeCap(t *testing.T) {
	prev := maxAudioURLBytes
	maxAudioURLBytes = 8
	defer func() { maxAudioURLBytes = prev }()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("oversized-payload"))
	}))
	defer server.Close()

	req := &Request{}
	err := WithAudioURL(server.URL, "mp3")(req)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "exceeds")
	assert.Empty(t, req.Messages)
}

func TestWithFile_ReadError(t *testing.T) {
	req := &Request{}
	err := WithFile("/nonexistent/path/to/file.pdf", "application/pdf")(req)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "read file")
	assert.Empty(t, req.Messages)
}