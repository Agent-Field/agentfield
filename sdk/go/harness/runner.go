package harness

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
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

// schemaAware is implemented by providers that consume a JSON schema natively
// (codex's --output-schema / --output-last-message) instead of the Write-tool
// file protocol used by the claude/opencode providers. When a schema is
// present the runner writes the STRICT schema rewrite, hands the provider the
// deterministic schema/output paths, and uses a codex-native prompt suffix
// rather than BuildPromptSuffix — matching the provider-dispatching suffix in
// codex_harness_patch.py. Providers that do not implement this keep the
// original file-write protocol, so this is fully back-compatible.
type schemaAware interface {
	SetSchema(schemaPath, outputPath string)
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

	// schema_mode selects how the agent is asked to produce the output:
	//   "single"      — one Write of the whole object (default, cheapest)
	//   "incremental" — build the object one top-level field at a time, with
	//                   field-level recovery (robust for large/deep schemas)
	//   "auto"        — incremental only when the schema is large, else single
	useIncremental := resolveIncremental(schema, opts)

	// Native-schema providers (codex) consume the schema through CLI flags
	// instead of the Write-tool file protocol. Detect and prepare before the
	// prompt suffix is built so codex gets a codex-native instruction while
	// claude/opencode keep the Write-tool suffix.
	var nativeSchemaPath, nativeOutputPath string
	useNativeSchema := false
	if schema != nil {
		if sa, ok := provider.(schemaAware); ok {
			absDir, absErr := filepath.Abs(outputDir)
			if absErr != nil {
				absDir = outputDir
			}
			nativeSchemaPath = SchemaPath(absDir)
			nativeOutputPath = OutputPath(absDir)
			if strictJSON, mErr := json.MarshalIndent(codexStrictJSONSchema(schema), "", "  "); mErr == nil {
				if mkErr := os.MkdirAll(filepath.Dir(nativeSchemaPath), 0o700); mkErr == nil {
					if wErr := os.WriteFile(nativeSchemaPath, strictJSON, 0o600); wErr == nil {
						sa.SetSchema(nativeSchemaPath, nativeOutputPath)
						useNativeSchema = true
						// Clean up unconditionally: CleanupTempFiles no-ops when
						// outputDir is "." so the strict schema / native output
						// file would otherwise leak into the working directory.
						defer func() {
							_ = os.Remove(nativeSchemaPath)
							_ = os.Remove(nativeOutputPath)
						}()
					}
				}
			}
		}
	}

	effectivePrompt := prompt
	if schema != nil {
		switch {
		case useNativeSchema:
			effectivePrompt = prompt + BuildCodexNativeSuffix(nativeSchemaPath, nativeOutputPath)
		case useIncremental:
			effectivePrompt = prompt + BuildIncrementalPromptSuffix(schema, outputDir)
		default:
			effectivePrompt = prompt + BuildPromptSuffix(schema, outputDir)
		}
	}

	startTime := time.Now()

	raw, err := r.executeWithRetry(ctx, provider, effectivePrompt, opts)
	if err != nil {
		return nil, err
	}

	if schema != nil {
		result := r.handleSchemaWithRetry(ctx, raw, schema, dest, outputDir, startTime, provider, opts, effectivePrompt, useIncremental)
		CleanupTempFiles(outputDir)
		return result, nil
	}

	elapsed := int(time.Since(startTime).Milliseconds())
	return &Result{
		Result:       raw.Result,
		IsError:      raw.IsError,
		ErrorMessage: raw.ErrorMessage,
		FailureType:  raw.FailureType,
		CostUSD:      raw.Metrics.CostUSD,
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
	if overrides.SchemaMode != "" {
		merged.SchemaMode = overrides.SchemaMode
	}

	return merged
}

// resolveIncremental decides whether to use the incremental field-by-field
// schema build. Mirrors the Python HarnessRunner._resolve_incremental:
//
//	schema is nil            -> false
//	mode == "incremental"    -> true
//	mode == "auto"           -> true iff the compact JSON schema exceeds the
//	                            large-schema token threshold
//	otherwise ("single"/"")  -> false
//
// The "auto" branch uses the COMPACT JSON encoding (json.Marshal, matching
// Python's json.dumps with no indent), which differs from the indented
// encoding the prompt-suffix builders use to decide whether to spill the
// schema to a file — this matches Python exactly.
func resolveIncremental(jsonSchema map[string]any, opts Options) bool {
	if jsonSchema == nil {
		return false
	}
	mode := strings.ToLower(opts.SchemaMode)
	if mode == "" {
		mode = "single"
	}
	switch mode {
	case "incremental":
		return true
	case "auto":
		compact, err := json.Marshal(jsonSchema)
		if err != nil {
			return false
		}
		return estimateTokens(string(compact)) > largeSchemaTokenThreshold
	default:
		return false
	}
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
	useIncremental bool,
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
	// Enforce the schema: unmarshal-only parsing (ParseAndValidate /
	// TryParseFromText) accepts missing required fields, invalid enums and
	// extra fields silently, so validate here to drive the retry loop below.
	if err == nil {
		if verr := runSchemaValidation(data, schema, dest); verr != nil {
			r.Logger.Printf("Initial output failed schema validation: %v", verr)
			err = verr
			data = nil
		}
	}

	if err == nil && data != nil {
		elapsed := int(time.Since(startTime).Milliseconds())
		cost, turns, sid, msgs := accumulateMetrics(allRaws)
		return &Result{
			Result:     initialRaw.Result,
			Parsed:     dest,
			CostUSD:    cost,
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
		cost, turns, sid, msgs := accumulateMetrics(allRaws)
		providerError := initialRaw.ErrorMessage
		if providerError == "" {
			providerError = "Provider execution failed."
		}
		return &Result{
			Result:       initialRaw.Result,
			IsError:      true,
			ErrorMessage: fmt.Sprintf("%s Output file was not created at %s.", providerError, outputPath),
			FailureType:  initialRaw.FailureType,
			CostUSD:      cost,
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
				cost, turns, sid, msgs := accumulateMetrics(allRaws)
				return &Result{
					IsError:      true,
					ErrorMessage: "context cancelled during schema retry",
					FailureType:  FailureTimeout,
					CostUSD:      cost,
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
		} else if useIncremental {
			// Incremental mode: patch only the fields that are missing or
			// invalid, one at a time, instead of regenerating the whole
			// object. The partial output file persists on disk between
			// attempts, so the agent edits it in place. The original goal is
			// prepended so the agent keeps the task in view (parity with the
			// Python incremental retry path).
			fieldErrors := DiagnoseFieldFailures(outputPath, schema, dest)
			followup := BuildIncrementalFollowup(fieldErrors, outputDir, schema)
			if originalPrompt != "" {
				retryPrompt = originalPrompt + "\n\n" + followup
			} else {
				retryPrompt = followup
			}
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
		if err == nil {
			if verr := runSchemaValidation(data, schema, dest); verr != nil {
				r.Logger.Printf("Schema retry %d failed validation: %v", retryNum+1, verr)
				err = verr
				data = nil
			}
		}

		if err == nil && data != nil {
			elapsed := int(time.Since(startTime).Milliseconds())
			cost, turns, sid, msgs := accumulateMetrics(allRaws)
			r.Logger.Printf("Schema validation succeeded on retry %d", retryNum+1)
			return &Result{
				Result:     retryRaw.Result,
				Parsed:     dest,
				CostUSD:    cost,
				NumTurns:   turns,
				DurationMS: elapsed,
				SessionID:  sid,
				Messages:   msgs,
			}
		}
	}

	elapsed := int(time.Since(startTime).Milliseconds())
	cost, turns, sid, msgs := accumulateMetrics(allRaws)
	finalDiagnosis := DiagnoseOutputFailure(outputPath, schema)
	return &Result{
		Result:  allRaws[len(allRaws)-1].Result,
		IsError: true,
		ErrorMessage: fmt.Sprintf(
			"Schema validation failed after %d retry attempt(s). Last error: %s",
			maxRetries, finalDiagnosis,
		),
		FailureType: FailureSchema,
		CostUSD:     cost,
		NumTurns:    turns,
		DurationMS:  elapsed,
		SessionID:   sid,
		Messages:    msgs,
	}
}

// accumulateMetrics sums metrics across every provider execution that
// contributed to a result — including failed retry attempts, mirroring the
// Python _accumulate_metrics. CostUSD is summed only over executions that
// reported a cost; the returned pointer is nil when none did (distinguishing
// "unknown" from "$0.00").
func accumulateMetrics(raws []*RawResult) (totalCost *float64, totalTurns int, sessionID string, allMessages []map[string]any) {
	for _, raw := range raws {
		if raw.Metrics.CostUSD != nil {
			if totalCost == nil {
				zero := 0.0
				totalCost = &zero
			}
			*totalCost += *raw.Metrics.CostUSD
		}
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

// runSchemaValidation enforces the JSON Schema on parsed output when the normal
// harness contract holds — both a destination struct and a schema were
// provided. It returns nil (validation not applicable) when either is absent or
// the data is nil, preserving the previous unmarshal-only behavior for
// schema-only or dest-less callers.
func runSchemaValidation(data map[string]any, schema map[string]any, dest any) error {
	if data == nil || schema == nil || dest == nil {
		return nil
	}
	return validateAgainstSchema(data, schema)
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
