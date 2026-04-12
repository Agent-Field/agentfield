package client

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestMinDuration(t *testing.T) {
	assert.Equal(t, 2*time.Second, minDuration(2*time.Second, 5*time.Second))
	assert.Equal(t, 5*time.Second, minDuration(8*time.Second, 5*time.Second))
}
