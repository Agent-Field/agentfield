module github.com/Agent-Field/agentfield/integrations/databricks/node

go 1.25

require (
	github.com/Agent-Field/agentfield/sdk/go v0.0.0
	gopkg.in/yaml.v3 v3.0.1
)

replace github.com/Agent-Field/agentfield/sdk/go => ../../../sdk/go
