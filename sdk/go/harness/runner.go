package harness

import (
	"context"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"reflect"
	"slices"
	"strings"
	"time"
)

// Runner orchestrates harness invocations with schema validation, retries,
// and provider management.
type Runner struct {
	// DefaultOptions are merged with per-call options (per-call wins).
	DefaultOptions Options
	Logger         *log.Logger
}

// NewRunner creates a harness runner with default options.
func NewRunner(defaults Options) *Runner {
	return &Runner{
		DefaultOptions: defaults,
		Logger:         log.New(io.Discard, "[harness] ", log.LstdFlags),
	}
}

// Run dispatches a prompt to a coding agent and returns the result.
// If schema is non-nil, the runner instructs the agent to write structured
// JSON output and validates it, retrying on failure.
//
// The schema parameter should be a JSON Schema as map[string]any. If dest
// is non-nil, the validated output is unmarshalled into it.
func (r *Runner) Run(ctx context.Context, prompt string, schema map[string]any, dest any, overrides Options) (*Result, error) {
	opts := r.mergeOptions(overrides)

	if opts.Provider == "" {
		return nil, fmt.Errorf(
			"no harness provider specified: set Provider in runner defaults or pass it to Run()",
		)
	}

	provider, err := r.buildProvider(opts)
	if err != nil {
		return nil, err
	}

	// Determine output directory for schema files.
	outputDir := opts.Cwd
	if outputDir == "" {
		outputDir = "."
	}
	var tempOutputDir string
	if opts.ProjectDir != "" {
		tempOutputDir, err = os.MkdirTemp(opts.ProjectDir, ".agentfield-out-")
		if err != nil {
			return nil, fmt.Errorf("creating temp output dir: %w", err)
		}
		defer os.RemoveAll(tempOutputDir)
		outputDir = tempOutputDir
	}

	effectivePrompt := prompt
	if schema != nil {
		effectivePrompt = prompt + BuildPromptSuffix(schema, outputDir)
	}

	startTime := time.Now()

	raw, err := r.executeWithRetry(ctx, provider, effectivePrompt, opts)
	if err != nil {
		return nil, err
	}

	if schema != nil {
		result := r.handleSchemaWithRetry(ctx, raw, schema, dest, outputDir, startTime, provider, opts, effectivePrompt)
		CleanupTempFiles(outputDir)
		return result, nil
	}

	elapsed := int(time.Since(startTime).Milliseconds())
	return &Result{
		Result:       raw.Result,
		IsError:      raw.IsError,
		ErrorMessage: raw.ErrorMessage,
		FailureType:  raw.FailureType,
		NumTurns:     raw.Metrics.NumTurns,
		DurationMS:   elapsed,
		SessionID:    raw.Metrics.SessionID,
		Messages:     raw.Messages,
	}, nil
}

func (r *Runner) buildProvider(opts Options) (Provider, error) {
	binPath := opts.BinPath
	if binPath == "" {
		binPath = r.DefaultOptions.BinPath
	}
	return BuildProvider(opts.Provider, binPath)
}

// mergeOptions combines default and per-call options. Per-call values take
// precedence. Zero values in overrides are treated as "use default" — callers
// cannot explicitly set numeric fields to zero to override a non-zero default.
func (r *Runner) mergeOptions(overrides Options) Options {
	merged := r.DefaultOptions

	if overrides.Provider != "" {
		merged.Provider = overrides.Provider
	}
	if overrides.Model != "" {
		merged.Model = overrides.Model
	}
	if overrides.MaxTurns > 0 {
		merged.MaxTurns = overrides.MaxTurns
	}
	if overrides.PermissionMode != "" {
		merged.PermissionMode = overrides.PermissionMode
	}
	if overrides.SystemPrompt != "" {
		merged.SystemPrompt = overrides.SystemPrompt
	}
	if overrides.Env != nil {
		if merged.Env == nil {
			merged.Env = make(map[string]string)
		}
		for k, v := range overrides.Env {
			merged.Env[k] = v
		}
	}
	if overrides.Cwd != "" {
		merged.Cwd = overrides.Cwd
	}
	if overrides.ProjectDir != "" {
		merged.ProjectDir = overrides.ProjectDir
	}
	if overrides.Tools != nil {
		merged.Tools = overrides.Tools
	}
	if overrides.MaxBudgetUSD > 0 {
		merged.MaxBudgetUSD = overrides.MaxBudgetUSD
	}
	if overrides.ResumeSessionID != "" {
		merged.ResumeSessionID = overrides.ResumeSessionID
	}
	if overrides.BinPath != "" {
		merged.BinPath = overrides.BinPath
	}
	if overrides.Timeout > 0 {
		merged.Timeout = overrides.Timeout
	}
	if overrides.MaxRetries > 0 {
		merged.MaxRetries = overrides.MaxRetries
	}
	if overrides.InitialDelay > 0 {
		merged.InitialDelay = overrides.InitialDelay
	}
	if overrides.MaxDelay > 0 {
		merged.MaxDelay = overrides.MaxDelay
	}
	if overrides.BackoffFactor > 0 {
		merged.BackoffFactor = overrides.BackoffFactor
	}
	if overrides.SchemaMaxRetries > 0 {
		merged.SchemaMaxRetries = overrides.SchemaMaxRetries
	}

	return merged
}

// transientPatterns are substrings that indicate a retryable error.
var transientPatterns = []string{
	"rate limit", "rate_limit", "overloaded", "timeout", "timed out",
	"connection reset", "connection refused", "temporarily unavailable",
	"service unavailable", "503", "502", "504", "internal server error", "500",
}

func isTransient(errStr string) bool {
	lower := strings.ToLower(errStr)
	for _, p := range transientPatterns {
		if strings.Contains(lower, p) {
			return true
		}
	}
	return false
}

func (r *Runner) executeWithRetry(ctx context.Context, provider Provider, prompt string, opts Options) (*RawResult, error) {
	maxRetries := opts.maxRetries()
	initialDelay := opts.initialDelay()
	maxDelay := opts.maxDelay()
	backoff := opts.backoffFactor()

	var lastErr error

	for attempt := 0; attempt <= maxRetries; attempt++ {
		raw, err := provider.Execute(ctx, prompt, opts)
		if err != nil {
			lastErr = err
			if isTransient(err.Error()) && attempt < maxRetries {
				sleepWithJitter(ctx, initialDelay, maxDelay, backoff, attempt)
				continue
			}
			return nil, err
		}

		if !raw.IsError {
			return raw, nil
		}

		errMsg := raw.ErrorMessage
		if isTransient(errMsg) && attempt < maxRetries {
			sleepWithJitter(ctx, initialDelay, maxDelay, backoff, attempt)
			continue
		}
		return raw, nil
	}

	if lastErr != nil {
		return nil, lastErr
	}
	return &RawResult{IsError: true, ErrorMessage: "max retries exceeded"}, nil
}

// sleepWithJitter pauses for an exponentially increasing delay with ±25% jitter.
// Uses math/rand global source, which auto-seeds since Go 1.20.
func sleepWithJitter(ctx context.Context, initialDelay, maxDelay, backoff float64, attempt int) {
	delay := math.Min(initialDelay*math.Pow(backoff, float64(attempt)), maxDelay)
	jitter := delay * 0.25
	delay += (rand.Float64()*2 - 1) * jitter

	timer := time.NewTimer(time.Duration(delay * float64(time.Second)))
	defer timer.Stop()
	select {
	case <-timer.C:
	case <-ctx.Done():
	}
}

func (r *Runner) handleSchemaWithRetry(
	ctx context.Context,
	initialRaw *RawResult,
	schema map[string]any,
	dest any,
	outputDir string,
	startTime time.Time,
	provider Provider,
	opts Options,
	originalPrompt string,
) *Result {
	outputPath := OutputPath(outputDir)
	maxRetries := opts.schemaMaxRetries()

	allRaws := []*RawResult{initialRaw}

	// Try to parse the output file
	data, err := ParseAndValidate(outputPath, dest)
	if err != nil && initialRaw.Result != "" {
		r.Logger.Printf("Output file missing/invalid at %s - trying stdout fallback", outputPath)
		data, err = TryParseFromText(initialRaw.Result, dest)
		if err == nil {
			r.Logger.Println("Stdout fallback succeeded")
		}
	}

	if err == nil && data != nil {
		elapsed := int(time.Since(startTime).Milliseconds())
		turns, sid, msgs := accumulateMetrics(allRaws)
		return &Result{
			Result:     initialRaw.Result,
			Parsed:     dest,
			NumTurns:   turns,
			DurationMS: elapsed,
			SessionID:  sid,
			Messages:   msgs,
		}
	}

	// Check if the initial error is non-retryable
	retryableFailures := map[FailureType]bool{
		FailureCrash:    true,
		FailureNoOutput: true,
		FailureNone:     true,
	}
	if initialRaw.IsError && !fileExists(outputPath) && !retryableFailures[initialRaw.FailureType] {
		elapsed := int(time.Since(startTime).Milliseconds())
		turns, sid, msgs := accumulateMetrics(allRaws)
		providerError := initialRaw.ErrorMessage
		if providerError == "" {
			providerError = "Provider execution failed."
		}
		return &Result{
			Result:       initialRaw.Result,
			IsError:      true,
			ErrorMessage: fmt.Sprintf("%s Output file was not created at %s.", providerError, outputPath),
			FailureType:  initialRaw.FailureType,
			NumTurns:     turns,
			DurationMS:   elapsed,
			SessionID:    sid,
			Messages:     msgs,
		}
	}

	lastSessionID := initialRaw.Metrics.SessionID

	for retryNum := 0; retryNum < maxRetries; retryNum++ {
		if retryNum > 0 {
			delay := math.Min(0.5*math.Pow(2, float64(retryNum-1)), 5.0)
			timer := time.NewTimer(time.Duration(delay * float64(time.Second)))
			select {
			case <-timer.C:
			case <-ctx.Done():
				timer.Stop()
				elapsed := int(time.Since(startTime).Milliseconds())
				turns, sid, msgs := accumulateMetrics(allRaws)
				return &Result{
					IsError:      true,
					ErrorMessage: "context cancelled during schema retry",
					FailureType:  FailureTimeout,
					NumTurns:     turns,
					DurationMS:   elapsed,
					SessionID:    sid,
					Messages:     msgs,
				}
			}
		}

		isCrash := allRaws[len(allRaws)-1].FailureType == FailureCrash && !fileExists(outputPath)
		var retryPrompt string
		if isCrash {
			retryPrompt = originalPrompt
		} else {
			errorDetail := DiagnoseOutputFailure(outputPath, schema)
			retryPrompt = BuildFollowupPrompt(errorDetail, outputDir, schema)
		}

		r.Logger.Printf("Schema validation retry %d/%d: %s",
			retryNum+1, maxRetries,
			truncate(DiagnoseOutputFailure(outputPath, schema), 200))

		retryOpts := opts
		if lastSessionID != "" && !isCrash {
			retryOpts.ResumeSessionID = lastSessionID
		}

		retryRaw, retryErr := r.executeWithRetry(ctx, provider, retryPrompt, retryOpts)
		if retryErr != nil {
			r.Logger.Printf("Schema retry %d execute error: %v", retryNum+1, retryErr)
			continue
		}
		allRaws = append(allRaws, retryRaw)

		if retryRaw.Metrics.SessionID != "" {
			lastSessionID = retryRaw.Metrics.SessionID
		}

		if retryRaw.IsError {
			r.Logger.Printf("Schema retry %d provider error: %s", retryNum+1, retryRaw.ErrorMessage)
			continue
		}

		// Re-create dest for validation on retry
		data, err = ParseAndValidate(outputPath, dest)
		if err != nil && retryRaw.Result != "" {
			data, err = TryParseFromText(retryRaw.Result, dest)
			if err == nil {
				r.Logger.Printf("Schema retry %d succeeded via stdout fallback", retryNum+1)
			}
		}

		if err == nil && data != nil {
			elapsed := int(time.Since(startTime).Milliseconds())
			turns, sid, msgs := accumulateMetrics(allRaws)
			r.Logger.Printf("Schema validation succeeded on retry %d", retryNum+1)
			return &Result{
				Result:     retryRaw.Result,
				Parsed:     dest,
				NumTurns:   turns,
				DurationMS: elapsed,
				SessionID:  sid,
				Messages:   msgs,
			}
		}
	}

	elapsed := int(time.Since(startTime).Milliseconds())
	turns, sid, msgs := accumulateMetrics(allRaws)
	finalDiagnosis := DiagnoseOutputFailure(outputPath, schema)
	return &Result{
		Result:  allRaws[len(allRaws)-1].Result,
		IsError: true,
		ErrorMessage: fmt.Sprintf(
			"Schema validation failed after %d retry attempt(s). Last error: %s",
			maxRetries, finalDiagnosis,
		),
		FailureType: FailureSchema,
		NumTurns:    turns,
		DurationMS:  elapsed,
		SessionID:   sid,
		Messages:    msgs,
	}
}

func accumulateMetrics(raws []*RawResult) (totalTurns int, sessionID string, allMessages []map[string]any) {
	for _, raw := range raws {
		totalTurns += raw.Metrics.NumTurns
		if raw.Metrics.SessionID != "" {
			sessionID = raw.Metrics.SessionID
		}
		allMessages = append(allMessages, raw.Messages...)
	}
	return
}

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

// StructToJSONSchema converts a Go struct (or pointer to struct) to a basic
// JSON Schema map by inspecting its JSON tags. This is a convenience for
// callers who don't want to construct schemas manually.
//
// For production use, consider using a dedicated JSON Schema library.
func StructToJSONSchema(v any) (map[string]any, error) {
	t := reflect.TypeOf(v)
	if t == nil {
		return nil, fmt.Errorf("cannot build schema from nil value")
	}
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	if t.Kind() != reflect.Struct {
		return nil, fmt.Errorf("StructToJSONSchema expects a struct or pointer to struct, got %s", t.Kind())
	}

	properties := make(map[string]any, t.NumField())
	required := make([]string, 0, t.NumField())

	for i := 0; i < t.NumField(); i++ {
		field := t.Field(i)
		if field.PkgPath != "" {
			continue
		}

		name := field.Name
		isRequired := true
		tag := field.Tag.Get("json")
		if tag != "" {
			parts := strings.Split(tag, ",")
			if parts[0] == "-" {
				continue
			}
			if parts[0] != "" {
				name = parts[0]
			}
			if slices.Contains(parts[1:], "omitempty") {
				isRequired = false
			}
		}

		fieldType := field.Type
		for fieldType.Kind() == reflect.Pointer {
			fieldType = fieldType.Elem()
		}

		propType := "object"
		switch fieldType.Kind() {
		case reflect.String:
			propType = "string"
		case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
			reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
			propType = "integer"
		case reflect.Float32, reflect.Float64:
			propType = "number"
		case reflect.Bool:
			propType = "boolean"
		case reflect.Slice, reflect.Array:
			propType = "array"
		case reflect.Struct:
			propType = "object"
		}

		properties[name] = map[string]any{"type": propType}
		if isRequired {
			required = append(required, name)
		}
	}

	return map[string]any{
		"type":       "object",
		"properties": properties,
		"required":   required,
	}, nil
}
