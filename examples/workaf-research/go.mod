module github.com/Agent-Field/agentfield/examples/workaf-research

go 1.21

require github.com/Agent-Field/agentfield/sdk/go v0.0.0

require (
	github.com/gorilla/websocket v1.5.3 // indirect
	github.com/santhosh-tekuri/jsonschema/v5 v5.3.1 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
)

// Builds against the in-tree SDK.
replace github.com/Agent-Field/agentfield/sdk/go => ../../sdk/go
