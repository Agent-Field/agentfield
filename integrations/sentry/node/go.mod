module github.com/Agent-Field/agentfield/integrations/sentry/node

go 1.23

require github.com/Agent-Field/agentfield/sdk/go v0.0.0

require gopkg.in/yaml.v3 v3.0.1 // indirect

replace github.com/Agent-Field/agentfield/sdk/go => ../../../sdk/go
