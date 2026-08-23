package ai

import (
	"bytes"
	"errors"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestAPIErrorCapsUnparseableBody(t *testing.T) {
	body := bytes.Repeat([]byte("x"), maxAPIErrorBody+1)
	err := newAPIError(502, body)

	var apiErr *APIError
	require.ErrorAs(t, err, &apiErr)
	require.Len(t, apiErr.Body, maxAPIErrorBody)
	require.True(t, errors.Is(err, &APIError{StatusCode: 502}))
}
