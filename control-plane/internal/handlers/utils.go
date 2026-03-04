package handlers

import (
	"encoding/json"
	"fmt"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
)

// marshalDataWithLogging marshals data to JSON with proper error handling and logging
func marshalDataWithLogging(data interface{}, fieldName string) ([]byte, error) {
	if data == nil {
		logger.Logger.Debug().Str("field", fieldName).Msg("marshaling nil data, returning null")
		return []byte("null"), nil
	}

	// Log the type and content of data being marshaled
	logger.Logger.Debug().Str("field", fieldName).Type("data_type", data).Msg("marshaling data")

	// Attempt to marshal with detailed error reporting
	jsonData, err := json.Marshal(data)
	if err != nil {
		logger.Logger.Error().Err(err).Str("field", fieldName).Msg("failed to marshal data")
		return nil, fmt.Errorf("failed to marshal %s: %w", fieldName, err)
	}

	logger.Logger.Debug().Str("field", fieldName).Int("bytes", len(jsonData)).Msg("data marshaled successfully")
	return jsonData, nil
}
