package observability

import (
	"context"
	"fmt"
	"net/url"
	"strings"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.26.0"
	"go.opentelemetry.io/otel/trace"
)

const (
	instrumentationName = "github.com/Agent-Field/agentfield/control-plane"
)

// TracerConfig holds configuration for the OTel tracer.
type TracerConfig struct {
	Enabled     bool   `yaml:"enabled" mapstructure:"enabled"`
	Exporter    string `yaml:"exporter" mapstructure:"exporter"`         // "otlp-http" or "otlp-grpc"
	Endpoint    string `yaml:"endpoint" mapstructure:"endpoint"`         // e.g. "localhost:4318"
	ServiceName string `yaml:"service_name" mapstructure:"service_name"` // defaults to "agentfield"
	Insecure    bool   `yaml:"insecure" mapstructure:"insecure"`         // skip TLS verification
}

// Tracer wraps the OTel tracer and provides AgentField-specific span helpers.
type Tracer struct {
	tracer   trace.Tracer
	provider *sdktrace.TracerProvider
}

// InitTracer creates and registers an OTel TracerProvider with an OTLP HTTP exporter.
// Returns a Tracer that can create spans, and a shutdown function to flush on exit.
func InitTracer(ctx context.Context, cfg TracerConfig) (*Tracer, func(context.Context) error, error) {
	if !cfg.Enabled {
		return nil, func(context.Context) error { return nil }, nil
	}

	serviceName := cfg.ServiceName
	if serviceName == "" {
		serviceName = "agentfield"
	}

	endpoint := resolveTraceEndpoint(cfg.Exporter, cfg.Endpoint)

	exporter, err := newTraceExporter(ctx, cfg.Exporter, endpoint, cfg.Insecure)
	if err != nil {
		return nil, nil, fmt.Errorf("create OTLP trace exporter: %w", err)
	}

	res, err := resource.New(ctx,
		resource.WithAttributes(
			semconv.ServiceName(serviceName),
			attribute.String("agentfield.component", "control-plane"),
		),
	)
	if err != nil {
		return nil, nil, fmt.Errorf("create OTel resource: %w", err)
	}

	provider := sdktrace.NewTracerProvider(
		sdktrace.WithBatcher(exporter),
		sdktrace.WithResource(res),
	)

	otel.SetTracerProvider(provider)

	tracer := provider.Tracer(instrumentationName)

	t := &Tracer{
		tracer:   tracer,
		provider: provider,
	}

	return t, provider.Shutdown, nil
}

func resolveTraceEndpoint(exporterName, endpoint string) string {
	if endpoint != "" {
		return endpoint
	}
	if exporterName == "otlp-grpc" {
		return "localhost:4317"
	}
	return "localhost:4318"
}

func newTraceExporter(ctx context.Context, exporterName, endpoint string, insecure bool) (sdktrace.SpanExporter, error) {
	hasScheme := strings.Contains(endpoint, "://")
	if hasScheme {
		parsed, err := url.Parse(endpoint)
		if err != nil || parsed.Scheme == "" || parsed.Host == "" || (parsed.Scheme != "http" && parsed.Scheme != "https") {
			return nil, fmt.Errorf("invalid OTLP endpoint %q", endpoint)
		}
	}

	if exporterName == "otlp-grpc" {
		opts := []otlptracegrpc.Option{}
		if hasScheme {
			opts = append(opts, otlptracegrpc.WithEndpointURL(endpoint))
		} else {
			opts = append(opts, otlptracegrpc.WithEndpoint(endpoint))
		}
		if insecure || strings.HasPrefix(endpoint, "http://") {
			opts = append(opts, otlptracegrpc.WithInsecure())
		}
		return otlptracegrpc.New(ctx, opts...)
	}
	if exporterName != "" && exporterName != "otlp-http" {
		return nil, fmt.Errorf("unsupported OTLP trace exporter %q", exporterName)
	}

	opts := []otlptracehttp.Option{}
	if hasScheme {
		opts = append(opts, otlptracehttp.WithEndpointURL(endpoint))
	} else {
		opts = append(opts, otlptracehttp.WithEndpoint(endpoint))
	}
	if insecure {
		opts = append(opts, otlptracehttp.WithInsecure())
	}
	return otlptracehttp.New(ctx, opts...)
}

// StartExecutionSpan creates a root span for an execution workflow.
func (t *Tracer) StartExecutionSpan(ctx context.Context, executionID, workflowID, agentNodeID string) (context.Context, trace.Span) {
	ctx, span := t.tracer.Start(ctx, "execution",
		trace.WithSpanKind(trace.SpanKindInternal),
		trace.WithAttributes(
			attribute.String("agentfield.execution.id", executionID),
			attribute.String("agentfield.workflow.id", workflowID),
			attribute.String("agentfield.agent.node_id", agentNodeID),
		),
	)
	return ctx, span
}

// StartStepSpan creates a child span for an individual execution step (reasoner or skill invocation).
func (t *Tracer) StartStepSpan(ctx context.Context, stepType, stepID, agentNodeID string) (context.Context, trace.Span) {
	ctx, span := t.tracer.Start(ctx, fmt.Sprintf("step.%s", stepType),
		trace.WithSpanKind(trace.SpanKindInternal),
		trace.WithAttributes(
			attribute.String("agentfield.step.type", stepType),
			attribute.String("agentfield.step.id", stepID),
			attribute.String("agentfield.agent.node_id", agentNodeID),
		),
	)
	return ctx, span
}

// RecordError adds an error event to the span and sets its status to error.
func (t *Tracer) RecordError(span trace.Span, err error) {
	span.RecordError(err)
	span.SetStatus(2, err.Error()) // codes.Error = 2
}

// Enabled returns true if tracing is initialized.
func (t *Tracer) Enabled() bool {
	return t != nil && t.tracer != nil
}
