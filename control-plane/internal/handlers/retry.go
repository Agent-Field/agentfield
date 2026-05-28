package handlers

import (
	"math/rand"
	"strings"
	"time"
)

var retryableFragments = []string{
	"database is locked",
	"SQLITE_BUSY",
	"database table is locked",
	"deadlock detected",
}

func isRetryableDBError(err error) bool {
	if err == nil {
		return false
	}
	msg := strings.ToLower(err.Error())
	for _, fragment := range retryableFragments {
		if strings.Contains(msg, strings.ToLower(fragment)) {
			return true
		}
	}
	return false
}

func backoffDelay(attempt int) time.Duration {
	if attempt <= 0 {
		attempt = 1
	}
	base := time.Duration(50*attempt) * time.Millisecond
	jitter := time.Duration(rand.Int63n(int64(25 * time.Millisecond)))
	return base + jitter
}
