package ai

import "strings"

// cloudgpuModelPrefix marks a model as CloudGPU-routed. It is stripped before
// the request goes out; see stripCloudGPUPrefix.
const cloudgpuModelPrefix = "cloudgpu/"

// defaultCloudGPUModel is the model DefaultConfig falls back to when the
// configuration selects CloudGPU and AI_MODEL is unset. The full list comes
// from GET /v1/models on the CloudGPU base URL.
const defaultCloudGPUModel = "deepseek-v4-flash"

// stripCloudGPUPrefix removes the routing-only "cloudgpu/" prefix from a model
// name. CloudGPU serves bare model ids, so the marker must not reach the wire.
func stripCloudGPUPrefix(model string) string {
	if len(model) >= len(cloudgpuModelPrefix) &&
		strings.EqualFold(model[:len(cloudgpuModelPrefix)], cloudgpuModelPrefix) {
		return model[len(cloudgpuModelPrefix):]
	}
	return model
}
