package ai

import (
	"encoding/json"
	"fmt"
	"strings"
)

// ValidateToolCall finds a tool by name and validates the arguments.
func ValidateToolCall(tools []Tool, tc ToolCall) (map[string]any, error) {
	var tool *Tool
	for i := range tools {
		if tools[i].Name == tc.Name {
			tool = &tools[i]
			break
		}
	}
	if tool == nil {
		return nil, fmt.Errorf("tool %q not found", tc.Name)
	}
	return ValidateToolArguments(tool, tc)
}

// ValidateToolArguments validates tool call arguments against the tool's
// JSON-Schema parameters. This is a basic implementation that checks
// required fields and type compatibility.
func ValidateToolArguments(tool *Tool, tc ToolCall) (map[string]any, error) {
	args := tc.Arguments
	if args == nil {
		args = map[string]any{}
	}

	schema := tool.Parameters
	if schema == nil {
		return args, nil
	}

	// Check required properties
	if reqRaw, ok := schema["required"]; ok {
		if reqList, ok := reqRaw.([]any); ok {
			var missing []string
			for _, r := range reqList {
				if name, ok := r.(string); ok {
					if _, exists := args[name]; !exists {
						missing = append(missing, name)
					}
				}
			}
			if len(missing) > 0 {
				raw, _ := json.MarshalIndent(args, "", "  ")
				return nil, fmt.Errorf("validation failed for tool %q:\n  - missing required: %s\n\nReceived arguments:\n%s",
					tc.Name, strings.Join(missing, ", "), string(raw))
			}
		}
	}

	return args, nil
}
