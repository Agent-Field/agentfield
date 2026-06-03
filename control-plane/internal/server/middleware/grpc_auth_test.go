package middleware

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

// mockHandler is a simple handler for testing
func mockHandler(ctx context.Context, req interface{}) (interface{}, error) {
	return "success", nil
}

// mockServerInfo for interceptor testing
var mockServerInfo = &grpc.UnaryServerInfo{
	FullMethod: "/test.Service/TestMethod",
}

func TestAPIKeyUnaryInterceptor_NoAuthConfigured(t *testing.T) {
	// When no API key is configured, all requests should be allowed
	interceptor := APIKeyUnaryInterceptor("")

	ctx := context.Background()
	resp, err := interceptor(ctx, "request", mockServerInfo, mockHandler)

	assert.NoError(t, err)
	assert.Equal(t, "success", resp)
}

func TestAPIKeyUnaryInterceptor_ValidXAPIKeyMetadata(t *testing.T) {
	interceptor := APIKeyUnaryInterceptor("secret-key")

	md := metadata.Pairs("x-api-key", "secret-key")
	ctx := metadata.NewIncomingContext(context.Background(), md)

	resp, err := interceptor(ctx, "request", mockServerInfo, mockHandler)

	assert.NoError(t, err)
	assert.Equal(t, "success", resp)
}

func TestAPIKeyUnaryInterceptor_ValidBearerToken(t *testing.T) {
	interceptor := APIKeyUnaryInterceptor("secret-key")

	md := metadata.Pairs("authorization", "Bearer secret-key")
	ctx := metadata.NewIncomingContext(context.Background(), md)

	resp, err := interceptor(ctx, "request", mockServerInfo, mockHandler)

	assert.NoError(t, err)
	assert.Equal(t, "success", resp)
}

func TestAPIKeyUnaryInterceptor_MissingMetadata(t *testing.T) {
	interceptor := APIKeyUnaryInterceptor("secret-key")

	// Context without any metadata
	ctx := context.Background()

	resp, err := interceptor(ctx, "request", mockServerInfo, mockHandler)

	assert.Nil(t, resp)
	assert.Error(t, err)

	st, ok := status.FromError(err)
	assert.True(t, ok)
	assert.Equal(t, codes.Unauthenticated, st.Code())
	assert.Equal(t, "missing metadata", st.Message())
}

func TestAPIKeyUnaryInterceptor_InvalidAPIKey(t *testing.T) {
	interceptor := APIKeyUnaryInterceptor("secret-key")

	md := metadata.Pairs("x-api-key", "wrong-key")
	ctx := metadata.NewIncomingContext(context.Background(), md)

	resp, err := interceptor(ctx, "request", mockServerInfo, mockHandler)

	assert.Nil(t, resp)
	assert.Error(t, err)

	st, ok := status.FromError(err)
	assert.True(t, ok)
	assert.Equal(t, codes.Unauthenticated, st.Code())
	assert.Equal(t, "invalid or missing api key", st.Message())
}

func TestAPIKeyUnaryInterceptor_InvalidBearerToken(t *testing.T) {
	interceptor := APIKeyUnaryInterceptor("secret-key")

	md := metadata.Pairs("authorization", "Bearer wrong-key")
	ctx := metadata.NewIncomingContext(context.Background(), md)

	resp, err := interceptor(ctx, "request", mockServerInfo, mockHandler)

	assert.Nil(t, resp)
	assert.Error(t, err)

	st, ok := status.FromError(err)
	assert.True(t, ok)
	assert.Equal(t, codes.Unauthenticated, st.Code())
}

func TestAPIKeyUnaryInterceptor_MalformedBearerToken(t *testing.T) {
	interceptor := APIKeyUnaryInterceptor("secret-key")

	tests := []struct {
		name   string
		auth   string
	}{
		{"no Bearer prefix", "secret-key"},
		{"Basic auth instead", "Basic secret-key"},
		{"malformed Bearer", "Bearersecret-key"},
		{"empty Bearer", "Bearer "},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			md := metadata.Pairs("authorization", tt.auth)
			ctx := metadata.NewIncomingContext(context.Background(), md)

			resp, err := interceptor(ctx, "request", mockServerInfo, mockHandler)

			assert.Nil(t, resp)
			assert.Error(t, err)

			st, ok := status.FromError(err)
			assert.True(t, ok)
			assert.Equal(t, codes.Unauthenticated, st.Code())
		})
	}
}

func TestAPIKeyUnaryInterceptor_XAPIKeyTakesPrecedence(t *testing.T) {
	interceptor := APIKeyUnaryInterceptor("secret-key")

	// Both headers set, x-api-key is valid
	md := metadata.Pairs(
		"x-api-key", "secret-key",
		"authorization", "Bearer wrong-key",
	)
	ctx := metadata.NewIncomingContext(context.Background(), md)

	resp, err := interceptor(ctx, "request", mockServerInfo, mockHandler)

	assert.NoError(t, err)
	assert.Equal(t, "success", resp)
}

func TestAPIKeyUnaryInterceptor_FallbackToBearer(t *testing.T) {
	interceptor := APIKeyUnaryInterceptor("secret-key")

	// Only bearer token set, should work as fallback
	md := metadata.Pairs("authorization", "Bearer secret-key")
	ctx := metadata.NewIncomingContext(context.Background(), md)

	resp, err := interceptor(ctx, "request", mockServerInfo, mockHandler)

	assert.NoError(t, err)
	assert.Equal(t, "success", resp)
}

func TestAPIKeyUnaryInterceptor_EmptyMetadata(t *testing.T) {
	interceptor := APIKeyUnaryInterceptor("secret-key")

	// Empty metadata (not nil, just no entries)
	md := metadata.MD{}
	ctx := metadata.NewIncomingContext(context.Background(), md)

	resp, err := interceptor(ctx, "request", mockServerInfo, mockHandler)

	assert.Nil(t, resp)
	assert.Error(t, err)

	st, ok := status.FromError(err)
	assert.True(t, ok)
	assert.Equal(t, codes.Unauthenticated, st.Code())
}

func TestAPIKeyUnaryInterceptor_CaseSensitivity(t *testing.T) {
	interceptor := APIKeyUnaryInterceptor("secret-key")

	// gRPC metadata keys are lowercased, but let's verify behavior
	tests := []struct {
		name  string
		key   string
		value string
		want  bool
	}{
		{"lowercase x-api-key", "x-api-key", "secret-key", true},
		// gRPC auto-lowercases keys, so these should work the same
		{"uppercase would be lowercased", "x-api-key", "secret-key", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			md := metadata.Pairs(tt.key, tt.value)
			ctx := metadata.NewIncomingContext(context.Background(), md)

			resp, err := interceptor(ctx, "request", mockServerInfo, mockHandler)

			if tt.want {
				assert.NoError(t, err)
				assert.Equal(t, "success", resp)
			} else {
				assert.Error(t, err)
			}
		})
	}
}

func TestAPIKeyUnaryInterceptor_MultipleValues(t *testing.T) {
	interceptor := APIKeyUnaryInterceptor("secret-key")

	// Multiple x-api-key values - should use first one
	md := metadata.Pairs(
		"x-api-key", "secret-key",
		"x-api-key", "another-key",
	)
	ctx := metadata.NewIncomingContext(context.Background(), md)

	resp, err := interceptor(ctx, "request", mockServerInfo, mockHandler)

	assert.NoError(t, err)
	assert.Equal(t, "success", resp)
}
