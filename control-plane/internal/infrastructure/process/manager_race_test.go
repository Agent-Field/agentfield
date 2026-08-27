package process

import (
	"sync"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/core/interfaces"
	"github.com/stretchr/testify/require"
)

// Boot restores that overran their bound keep running while later passes and
// update jobs start and stop other nodes; the PID map must survive that.
func TestProcessManagerIsSafeForConcurrentUse(t *testing.T) {
	pm := NewProcessManager()
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			pid, err := pm.Start(interfaces.ProcessConfig{Command: "sleep", Args: []string{"30"}})
			require.NoError(t, err)
			_, _ = pm.Status(pid)
			require.NoError(t, pm.Stop(pid))
		}()
	}
	wg.Wait()
}
