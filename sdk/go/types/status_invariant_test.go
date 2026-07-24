package types

import (
	"strings"
	"testing"
	"unicode"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// allInputs provides a broad set of string inputs for property-based style testing.
var allInputs = []string{
	// Empty / whitespace
	"",
	" ",
	"\t",
	"\n",
	"   ",
	// Canonical statuses
	ExecutionStatusPending,
	ExecutionStatusQueued,
	ExecutionStatusWaiting,
	ExecutionStatusRunning,
	ExecutionStatusSucceeded,
	ExecutionStatusFailed,
	ExecutionStatusCancelled,
	ExecutionStatusTimeout,
	"unknown",
	// Aliases
	"success", "successful", "completed", "complete", "done", "ok",
	"error", "failure", "errored",
	"canceled", "cancel",
	"timed_out",
	"wait",
	"awaiting_approval", "awaiting_human", "approval_pending",
	"in_progress", "processing",
	// Uppercased variants
	"PENDING", "QUEUED", "RUNNING", "SUCCEEDED", "FAILED", "CANCELLED",
	"SUCCESS", "ERROR", "DONE",
	// Mixed case
	"Pending", "Running", "Failed",
	// Unicode inputs
	"状態", "عملية", "ставка",
	// Unusual inputs
	"@@@", "123", "   pending   ", "\trunning\n",
	"totally-unknown-status",
	"pending pending",
}

// TestInvariant_Status_NormalizeIsProjection verifies that normalize is
// idempotent: normalize(normalize(x)) == normalize(x) for ALL string inputs.
func TestInvariant_Status_NormalizeIsProjection(t *testing.T) {
	for _, input := range allInputs {
		t.Run("input="+safeTestName(input), func(t *testing.T) {
			once := NormalizeStatus(input)
			twice := NormalizeStatus(once)
			assert.Equal(t, once, twice,
				"NormalizeStatus must be idempotent: NormalizeStatus(%q) = %q, but NormalizeStatus(%q) = %q",
				input, once, once, twice)
		})
	}
}

// TestInvariant_Status_NormalizeProjection_AdditionalUnicode verifies
// idempotency on a wider range of unicode characters.
func TestInvariant_Status_NormalizeProjection_AdditionalUnicode(t *testing.T) {
	unicodeInputs := []string{
		"\u0000",       // null byte
		"\uFFFD",       // replacement character
		"café",         // accented characters
		"naïve",        // diacritic
		"日本語",          // CJK characters
		"Ω",            // Greek capital letter Omega
		"running\u200B", // zero-width space
	}
	for _, input := range unicodeInputs {
		once := NormalizeStatus(input)
		twice := NormalizeStatus(once)
		assert.Equal(t, once, twice,
			"NormalizeStatus must be idempotent for unicode input %q: once=%q, twice=%q",
			input, once, twice)
	}
}

// TestInvariant_Status_TerminalActivePartition verifies that every canonical
// status is in exactly one of {terminal, active, unknown} — no overlap, no gaps.
func TestInvariant_Status_TerminalActivePartition(t *testing.T) {
	for status := range canonicalStatuses {
		t.Run("status="+status, func(t *testing.T) {
			isTerminal := TerminalStatuses[status]
			isActive := ActiveStatuses[status]
			isUnknown := status == "unknown"

			// Count which categories this status belongs to
			categories := 0
			if isTerminal {
				categories++
			}
			if isActive {
				categories++
			}
			if isUnknown {
				categories++
			}

			// "unknown" should not be in terminal or active
			if isUnknown {
				assert.False(t, isTerminal, "'unknown' must not be in TerminalStatuses")
				assert.False(t, isActive, "'unknown' must not be in ActiveStatuses")
			} else {
				// Every non-unknown canonical status must be in exactly one of terminal or active
				assert.Equal(t, 1, categories,
					"status %q must be in exactly one of {terminal, active}: terminal=%v, active=%v",
					status, isTerminal, isActive)
			}
		})
	}
}

// TestInvariant_Status_AllCanonicalStatusesCovered verifies that the union of
// TerminalStatuses and ActiveStatuses covers all canonical statuses except "unknown".
func TestInvariant_Status_AllCanonicalStatusesCovered(t *testing.T) {
	for status := range canonicalStatuses {
		if status == "unknown" {
			continue
		}
		inTerminal := TerminalStatuses[status]
		inActive := ActiveStatuses[status]
		assert.True(t, inTerminal || inActive,
			"canonical status %q must appear in TerminalStatuses or ActiveStatuses", status)
	}
}

// TestInvariant_Status_AliasConsistency verifies that every alias maps to a
// canonical status that actually exists in canonicalStatuses.
func TestInvariant_Status_AliasConsistency(t *testing.T) {
	for alias, canonical := range statusAliases {
		t.Run("alias="+alias, func(t *testing.T) {
			_, exists := canonicalStatuses[canonical]
			assert.True(t, exists,
				"alias %q maps to %q which is not in canonicalStatuses", alias, canonical)

			// Normalize of an alias must equal the canonical value
			normalized := NormalizeStatus(alias)
			assert.Equal(t, canonical, normalized,
				"NormalizeStatus(%q) must return canonical %q, got %q", alias, canonical, normalized)
		})
	}
}

// TestInvariant_Status_AliasesDoNotOverlapCanonicals ensures that no alias
// is also a canonical status (which would create ambiguity).
func TestInvariant_Status_AliasesDoNotOverlapCanonicals(t *testing.T) {
	for alias := range statusAliases {
		_, isCanonical := canonicalStatuses[alias]
		assert.False(t, isCanonical,
			"alias %q must not also appear in canonicalStatuses", alias)
	}
}

// TestInvariant_Status_CaseInsensitivity verifies that for all canonical
// statuses and aliases, NormalizeStatus(strings.ToUpper(x)) == NormalizeStatus(x).
func TestInvariant_Status_CaseInsensitivity(t *testing.T) {
	testInputs := make([]string, 0, len(canonicalStatuses)+len(statusAliases))
	for s := range canonicalStatuses {
		testInputs = append(testInputs, s)
	}
	for a := range statusAliases {
		testInputs = append(testInputs, a)
	}

	for _, input := range testInputs {
		t.Run("input="+input, func(t *testing.T) {
			lower := NormalizeStatus(input)
			upper := NormalizeStatus(strings.ToUpper(input))
			mixed := NormalizeStatus(strings.ToTitle(input))

			assert.Equal(t, lower, upper,
				"NormalizeStatus must be case-insensitive: %q vs %q -> %q vs %q",
				input, strings.ToUpper(input), lower, upper)
			assert.Equal(t, lower, mixed,
				"NormalizeStatus must be case-insensitive for mixed case: %q -> %q vs %q",
				strings.ToTitle(input), lower, mixed)
		})
	}
}

// TestInvariant_Status_NormalizeOutputIsAlwaysCanonical verifies that the
// output of NormalizeStatus is always a member of canonicalStatuses.
func TestInvariant_Status_NormalizeOutputIsAlwaysCanonical(t *testing.T) {
	for _, input := range allInputs {
		output := NormalizeStatus(input)
		_, exists := canonicalStatuses[output]
		assert.True(t, exists,
			"NormalizeStatus(%q) = %q which is not a canonical status", input, output)
	}
}

// TestInvariant_Status_UnknownInputsReturnUnknown verifies that inputs that are
// neither canonical nor aliased normalize to "unknown".
func TestInvariant_Status_UnknownInputsReturnUnknown(t *testing.T) {
	unknownInputs := []string{
		"totally-unknown-status",
		"@@@",
		"123",
		"pending pending",
		"状態",
		"UNKNOWN_STATUS_XYZ",
	}
	for _, input := range unknownInputs {
		result := NormalizeStatus(input)
		assert.Equal(t, "unknown", result,
			"unrecognized input %q must normalize to 'unknown', got %q", input, result)
	}
}

// TestInvariant_Status_IsTerminalIsActiveMutuallyExclusive verifies that no
// status can be both terminal and active simultaneously.
func TestInvariant_Status_IsTerminalIsActiveMutuallyExclusive(t *testing.T) {
	for _, input := range allInputs {
		isTerminal := IsTerminalStatus(input)
		isActive := IsActiveStatus(input)
		assert.False(t, isTerminal && isActive,
			"status %q cannot be both terminal and active simultaneously", input)
	}
}

// TestInvariant_Status_CanonicalStatusesAreNormalized verifies that every key
// in canonicalStatuses is already in normalized form (lowercase, trimmed).
func TestInvariant_Status_CanonicalStatusesAreNormalized(t *testing.T) {
	for status := range canonicalStatuses {
		expected := strings.TrimSpace(strings.ToLower(status))
		require.Equal(t, expected, status,
			"canonical status key %q must be in normalized form (lowercase, trimmed)", status)
	}
}

// safeTestName converts a string to a safe test name (printable ASCII only).
func safeTestName(s string) string {
	if s == "" {
		return "(empty)"
	}
	var b strings.Builder
	for _, r := range s {
		if unicode.IsPrint(r) && r < 128 {
			b.WriteRune(r)
		} else {
			b.WriteString("_")
		}
	}
	result := b.String()
	if len(result) > 40 {
		result = result[:40] + "..."
	}
	if result == "" {
		result = "(non-ascii)"
	}
	return result
}
