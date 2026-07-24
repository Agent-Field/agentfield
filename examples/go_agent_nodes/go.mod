module github.com/Agent-Field/agentfield/examples/go_agent_nodes

go 1.21

require (
	github.com/Agent-Field/agentfield/sdk/go v0.1.6
	github.com/aws/aws-lambda-go v1.47.0
	github.com/awslabs/aws-lambda-go-api-proxy v0.16.2
)

require gopkg.in/yaml.v3 v3.0.1 // indirect

replace github.com/Agent-Field/agentfield/sdk/go => ../../sdk/go
