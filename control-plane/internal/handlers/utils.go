package handlers

import (
	"encoding/json"
	"fmt"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
)

// marshalDataWithLogging marshals data to JSON with proper error handling and logging
func marshalDataWithLogging(data interface{}, fieldName string) ([]byte, error) {
	if data == nil {
		logger.Logger.Debug().Msgf("🔍 MARSHAL_DEBUG: %s is nil, returning null", fieldName)
		return []byte("null"), nil
	}

	// Log the type and content of data being marshaled
	logger.Logger.Debug().Msgf("🔍 MARSHAL_DEBUG: Marshaling %s (type: %T)", fieldName, data)

	// Attempt to marshal with detailed error reporting
	jsonData, err := json.Marshal(data)
	if err != nil {
		logger.Logger.Error().Err(err).Msgf("❌ MARSHAL_ERROR: Failed to marshal %s (type: %T): %v", fieldName, data, data)
		return nil, fmt.Errorf("failed to marshal %s: %w", fieldName, err)
	}

	logger.Logger.Debug().Msgf("✅ MARSHAL_SUCCESS: Successfully marshaled %s (%d bytes): %s", fieldName, len(jsonData), string(jsonData))
	return jsonData, nil
}
