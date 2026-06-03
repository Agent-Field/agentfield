package harness

import (
	"context"
	"fmt"
	"strings"
	"time"
)

// GeminiProvider invokes the Gemini CLI as a subprocess.
// Gemini outputs plain text (no structured JSON), so results are single-turn.
type GeminiProvider struct {
	BinPath string
}

// NewGeminiProvider creates a Gemini provider. If binPath is empty,
// it defaults to "gemini".
func NewGeminiProvider(binPath string) *GeminiProvider {
	if binPath == "" {
		binPath = "gemini"
	}
	return &GeminiProvider{BinPath: binPath}
}

func (p *GeminiProvider) Execute(ctx context.Context, prompt string, options Options) (*RawResult, error) {
	cmd := []string{p.BinPath}

	if options.Cwd != "" {
		cmd = append(cmd, "-C", options.Cwd)
	} else if options.ProjectDir != "" {
		cmd = append(cmd, "-C", options.ProjectDir)
	}

	if options.PermissionMode == "auto" {
		cmd = append(cmd, "--sandbox")
	}

	if options.Model != "" {
		cmd = append(cmd, "-m", options.Model)
	}

	// Prompt passed via -p flag.
	cmd = append(cmd, "-p", prompt)

	env := make(map[string]string)
	for k, v := range options.Env {
		env[k] = v
	}

	cwd := options.Cwd
	if cwd == "" {
		cwd = options.ProjectDir
	}

	startAPI := time.Now()

	cliResult, err := RunCLI(ctx, cmd, env, cwd, options.timeout())
	apiMS := int(time.Since(startAPI).Milliseconds())

	if err != nil {
		if isExecNotFound(err) {
			return &RawResult{
				IsError: true,
				ErrorMessage: fmt.Sprintf(
					"Gemini binary not found at '%s'. Install: https://github.com/google-gemini/gemini-cli",
					p.BinPath,
				),
				FailureType: FailureCrash,
				Metrics:     Metrics{},
			}, nil
		}
		if strings.Contains(err.Error(), "timed out") {
			return &RawResult{
				IsError:      true,
				ErrorMessage: err.Error(),
				FailureType:  FailureTimeout,
				Metrics:      Metrics{DurationAPIMS: apiMS},
			}, nil
		}
		return nil, err
	}

	resultText := strings.TrimSpace(cliResult.Stdout)
	cleanStderr := StripANSI(strings.TrimSpace(cliResult.Stderr))

	raw := &RawResult{
		Result:   resultText,
		Messages: nil,
		Metrics: Metrics{
			DurationAPIMS: apiMS,
			NumTurns:      1,
		},
		ReturnCode: cliResult.ReturnCode,
	}

	if cliResult.ReturnCode < 0 {
		raw.FailureType = FailureCrash
		raw.IsError = true
		if cleanStderr != "" {
			raw.ErrorMessage = fmt.Sprintf("Process killed by signal %d. stderr: %.500s",
				-cliResult.ReturnCode, cleanStderr)
		} else {
			raw.ErrorMessage = fmt.Sprintf("Process killed by signal %d.", -cliResult.ReturnCode)
		}
	} else if cliResult.ReturnCode != 0 && resultText == "" {
		raw.FailureType = FailureCrash
		raw.IsError = true
		if cleanStderr != "" {
			raw.ErrorMessage = truncate(cleanStderr, 1000)
		} else {
			raw.ErrorMessage = fmt.Sprintf("Process exited with code %d and produced no output.", cliResult.ReturnCode)
		}
	}

	if resultText == "" {
		raw.Metrics.NumTurns = 0
	}

	return raw, nil
}
