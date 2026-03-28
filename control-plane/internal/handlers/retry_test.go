package handlers

import (
	"errors"
	"testing"
	"time"
)

func TestIsRetryableDBError(t *testing.T) {
	t.Parallel()

	t.Run("nil error", func(t *testing.T) {
		t.Parallel()
		if isRetryableDBError(nil) {
			t.Fatal("expected nil error to be non-retryable")
		}
	})

	for _, fragment := range retryableFragments {
		fragment := fragment
		t.Run("retryable fragment: "+fragment, func(t *testing.T) {
			t.Parallel()
			err := errors.New("prefix " + fragment + " suffix")
			if !isRetryableDBError(err) {
				t.Fatalf("expected retryable fragment %q to match", fragment)
			}
		})
	}

	t.Run("deadlock case insensitive", func(t *testing.T) {
		t.Parallel()
		err := errors.New("DeadLock Detected")
		if !isRetryableDBError(err) {
			t.Fatal("expected case-insensitive deadlock match")
		}
	})

	t.Run("non retryable", func(t *testing.T) {
		t.Parallel()
		err := errors.New("constraint violation")
		if isRetryableDBError(err) {
			t.Fatal("expected non-retryable error to return false")
		}
	})

	t.Run("execution not found is non retryable", func(t *testing.T) {
		t.Parallel()
		err := errors.New("execution exec-123 not found")
		if isRetryableDBError(err) {
			t.Fatal("expected execution-not-found error to be non-retryable")
		}
	})
}

func TestBackoffDelay_AttemptZeroOrNegativeUsesOne(t *testing.T) {
	t.Parallel()

	for _, attempt := range []int{0, -1, -3} {
		d := backoffDelay(attempt)
		min := 50 * time.Millisecond
		maxExclusive := 75 * time.Millisecond
		if d < min || d >= maxExclusive {
			t.Fatalf("backoffDelay(%d) = %v, want in [%v, %v)", attempt, d, min, maxExclusive)
		}
	}
}

func TestBackoffDelay_RangeByAttempt(t *testing.T) {
	t.Parallel()

	for _, attempt := range []int{1, 2, 4} {
		d := backoffDelay(attempt)
		min := time.Duration(50*attempt) * time.Millisecond
		maxExclusive := min + 25*time.Millisecond
		if d < min || d >= maxExclusive {
			t.Fatalf("backoffDelay(%d) = %v, want in [%v, %v)", attempt, d, min, maxExclusive)
		}
	}
}

func TestBackoffDelay_AttemptTwoAlwaysGreaterThanAttemptOne(t *testing.T) {
	t.Parallel()

	d1 := backoffDelay(1)
	d2 := backoffDelay(2)
	if d2 <= d1 {
		t.Fatalf("expected attempt 2 delay > attempt 1 delay, got d1=%v d2=%v", d1, d2)
	}
}
