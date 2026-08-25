package packages

import "os"

const (
	HostingRailway = "railway"
	HostingDocker  = "docker"
	HostingLocal   = "local"
)

// HostingPlatform classifies the control-plane process using the same durable
// signals used by the version API and package-maintenance boot migration.
func HostingPlatform() string {
	if os.Getenv("RAILWAY_SERVICE_ID") != "" {
		return HostingRailway
	}
	if _, err := os.Stat("/.dockerenv"); err == nil || os.Getenv("AGENTFIELD_HOME") == "/data" {
		return HostingDocker
	}
	return HostingLocal
}

func HostedInContainer() bool {
	return HostingPlatform() != HostingLocal
}
