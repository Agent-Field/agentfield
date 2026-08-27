module github.com/Agent-Field/agentfield/examples/triggers-demo-go

go 1.21

require github.com/Agent-Field/agentfield/sdk/go v0.0.0

require (
	github.com/santhosh-tekuri/jsonschema/v5 v5.3.1 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
)

// The demo deliberately builds against the in-tree SDK so changes to the
// triggers DX flow through immediately, without waiting on a module release.
replace github.com/Agent-Field/agentfield/sdk/go => ../../sdk/go
