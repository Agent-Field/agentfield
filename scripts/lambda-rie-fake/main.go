// Command lambda-rie-fake is a minimal local stand-in for AWS Lambda, used to
// test the Go SDK's Lambda adapter path (examples/go_agent_nodes/cmd/lambda)
// without Docker or an AWS account. It exposes two listeners:
//
//   - The Lambda Runtime API (the subset aws-lambda-go's lambda.Start()
//     drives - see aws/aws-lambda-go's lambda/runtime_api_client.go):
//     GET  /2018-06-01/runtime/invocation/next
//     POST /2018-06-01/runtime/invocation/{id}/response
//     POST /2018-06-01/runtime/invocation/{id}/error
//     The compiled Lambda binary connects to this one via AWS_LAMBDA_RUNTIME_API,
//     exactly as it would against the real service.
//
//   - A public "Function URL" front end: any HTTP request received here is
//     translated into an API Gateway HTTP API v2 payload event, sent through
//     the runtime API to the function, and the v2 response event is
//     translated back into a real HTTP response. This is what lets the
//     control plane hit this process with an ordinary POST /execute exactly
//     as it would a real Lambda Function URL - the control plane never knows
//     it's talking to Lambda.
package main

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/aws/aws-lambda-go/events"
)

type invocation struct {
	id    string
	event []byte
}

type outcome struct {
	kind string // "response" or "error"
	body []byte
}

func main() {
	runtimePort := envOr("PORT", "9001")
	publicPort := envOr("PUBLIC_PORT", "9000")

	invokeCh := make(chan invocation)
	outcomeCh := make(chan outcome)
	var counter int64

	invokeSync := func(event []byte) outcome {
		id := fmt.Sprintf("req-%d", atomic.AddInt64(&counter, 1))
		invokeCh <- invocation{id: id, event: event}
		return <-outcomeCh
	}

	runtimeMux := http.NewServeMux()
	runtimeMux.HandleFunc("/admin/invoke", func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		out := invokeSync(body)
		w.Header().Set("X-Lambda-Outcome", out.kind)
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(out.body)
	})
	runtimeMux.HandleFunc("/2018-06-01/runtime/invocation/", func(w http.ResponseWriter, r *http.Request) {
		const prefix = "/2018-06-01/runtime/invocation/"
		rest := r.URL.Path[len(prefix):]

		switch {
		case r.Method == http.MethodGet && rest == "next":
			// A redeployed function reconnects with a fresh /next long-poll
			// while the old process's connection may still be around from
			// the server's point of view for a moment. Without watching
			// r.Context().Done(), a stale handler goroutine whose client
			// already died could still win the race on invokeCh and
			// silently swallow the next invocation. Bail out cleanly if
			// this connection goes away first.
			select {
			case inv := <-invokeCh:
				w.Header().Set("Lambda-Runtime-Aws-Request-Id", inv.id)
				w.Header().Set("Lambda-Runtime-Deadline-Ms", strconv.FormatInt(time.Now().Add(30*time.Second).UnixMilli(), 10))
				w.Header().Set("Lambda-Runtime-Invoked-Function-Arn", "arn:aws:lambda:local:000000000000:function:agentfield-local-test")
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write(inv.event)
			case <-r.Context().Done():
				return
			}

		case r.Method == http.MethodPost && strings.HasSuffix(rest, "/response"):
			body, _ := io.ReadAll(r.Body)
			outcomeCh <- outcome{kind: "response", body: body}
			w.WriteHeader(http.StatusAccepted)

		case r.Method == http.MethodPost && strings.HasSuffix(rest, "/error"):
			body, _ := io.ReadAll(r.Body)
			outcomeCh <- outcome{kind: "error", body: body}
			w.WriteHeader(http.StatusAccepted)

		default:
			http.NotFound(w, r)
		}
	})

	go func() {
		log.Printf("fake Lambda Runtime API listening on :%s", runtimePort)
		log.Fatal(http.ListenAndServe(":"+runtimePort, runtimeMux))
	}()

	publicMux := http.NewServeMux()
	publicMux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		eventJSON, err := requestToV2Event(r)
		if err != nil {
			http.Error(w, "failed to build lambda event: "+err.Error(), http.StatusInternalServerError)
			return
		}

		out := invokeSync(eventJSON)
		if out.kind == "error" {
			http.Error(w, "function invocation error: "+string(out.body), http.StatusInternalServerError)
			return
		}

		var resp events.APIGatewayV2HTTPResponse
		if err := json.Unmarshal(out.body, &resp); err != nil {
			http.Error(w, "bad lambda response envelope: "+err.Error(), http.StatusBadGateway)
			return
		}
		writeHTTPResponse(w, resp)
	})

	log.Printf("fake Lambda Function URL listening on :%s (proxies to :%s via the runtime API)", publicPort, runtimePort)
	log.Fatal(http.ListenAndServe(":"+publicPort, publicMux))
}

func requestToV2Event(r *http.Request) ([]byte, error) {
	bodyBytes, err := io.ReadAll(r.Body)
	if err != nil {
		return nil, err
	}

	headers := make(map[string]string, len(r.Header))
	for k, vs := range r.Header {
		headers[strings.ToLower(k)] = strings.Join(vs, ",")
	}

	event := events.APIGatewayV2HTTPRequest{
		Version:        "2.0",
		RouteKey:       "$default",
		RawPath:        r.URL.Path,
		RawQueryString: r.URL.RawQuery,
		Headers:        headers,
		Body:           string(bodyBytes),
		RequestContext: events.APIGatewayV2HTTPRequestContext{
			HTTP: events.APIGatewayV2HTTPRequestContextHTTPDescription{
				Method: r.Method,
				Path:   r.URL.Path,
			},
		},
	}
	return json.Marshal(event)
}

func writeHTTPResponse(w http.ResponseWriter, resp events.APIGatewayV2HTTPResponse) {
	for k, v := range resp.Headers {
		w.Header().Set(k, v)
	}
	for k, vs := range resp.MultiValueHeaders {
		for _, v := range vs {
			w.Header().Add(k, v)
		}
	}
	status := resp.StatusCode
	if status == 0 {
		status = http.StatusOK
	}
	w.WriteHeader(status)

	if resp.IsBase64Encoded {
		if decoded, err := base64.StdEncoding.DecodeString(resp.Body); err == nil {
			_, _ = w.Write(decoded)
			return
		}
	}
	_, _ = w.Write([]byte(resp.Body))
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
