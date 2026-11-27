package ui

import (
	"errors"
	"strconv"
	"strings"
)

// maxPage bounds every page query parameter. Page sizes here are at most 200,
// so (page-1)*size and page*size stay far from int overflow, which would
// otherwise wrap into a negative offset that storage serves as the first page.
const maxPage = 1_000_000

// parsePage reads a 1-based page query parameter. Missing, malformed and
// non-positive values mean the first page. Larger values are clamped to
// maxPage, including those too large for an int.
func parsePage(value string) int {
	page, err := strconv.Atoi(strings.TrimSpace(value))
	// On ErrRange, Atoi returns the int limit matching the value's sign, so the
	// clamp below handles it like any other out-of-range page. Atoi reports
	// ErrRange as soon as the digits overflow, without reading what follows.
	if err != nil && !errors.Is(err, strconv.ErrRange) {
		return 1
	}
	return min(max(page, 1), maxPage)
}
