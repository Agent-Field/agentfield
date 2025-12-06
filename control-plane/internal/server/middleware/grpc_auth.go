package middleware

import (
	"context"
	"strings"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

// APIKeyUnaryInterceptor enforces API key authentication on gRPC calls.
func APIKeyUnaryInterceptor(apiKey string) grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
		// No auth configured, allow everything.
		if apiKey == "" {
			return handler(ctx, req)
		}

		md, ok := metadata.FromIncomingContext(ctx)
		if !ok {
			return nil, status.Error(codes.Unauthenticated, "missing metadata")
		}

		// Check x-api-key
		if keys := md.Get("x-api-key"); len(keys) > 0 && keys[0] == apiKey {
			return handler(ctx, req)
		}

		// Check Authorization: Bearer
		if auths := md.Get("authorization"); len(auths) > 0 && strings.HasPrefix(auths[0], "Bearer ") {
			if strings.TrimPrefix(auths[0], "Bearer ") == apiKey {
				return handler(ctx, req)
			}
		}

		return nil, status.Error(codes.Unauthenticated, "invalid or missing api key")
	}
}
