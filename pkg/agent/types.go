package agent

import (
	"context"

	"github.com/badlogic/pi-go/pkg/ai"
)

// StreamFn is the function signature for making LLM streaming calls.
// Mirrors the TypeScript StreamFn type.
type StreamFn func(model *ai.Model, ctx ai.Context, opts *ai.SimpleStreamOptions) *ai.AssistantMessageEventStream

// AgentLoopConfig configures a single run of the agent loop.
type AgentLoopConfig struct {
	ai.SimpleStreamOptions

	Model *ai.Model

	// ConvertToLLM transforms AgentMessages to LLM-compatible Messages before each call.
	ConvertToLLM func(messages []AgentMessage) ([]ai.Message, error)

	// TransformContext optionally transforms the agent-level context before ConvertToLLM.
	TransformContext func(ctx context.Context, messages []AgentMessage) ([]AgentMessage, error)

	// GetApiKey dynamically resolves an API key for expiring tokens.
	GetApiKey func(provider string) (string, error)

	// GetSteeringMessages returns steering messages to inject mid-run.
	GetSteeringMessages func() ([]AgentMessage, error)

	// GetFollowUpMessages returns follow-up messages after the agent would stop.
	GetFollowUpMessages func() ([]AgentMessage, error)
}

// AgentMessage is a union: it can be a standard LLM Message or a custom app message.
// The Custom field can hold arbitrary application-specific data.
type AgentMessage struct {
	ai.Message
	Custom any `json:"custom,omitempty"`
}

// NewAgentMessageFromMessage wraps a standard Message.
func NewAgentMessageFromMessage(m ai.Message) AgentMessage {
	return AgentMessage{Message: m}
}

// IsLLMMessage returns true if this is a standard LLM message (not custom).
func (m AgentMessage) IsLLMMessage() bool {
	return m.Custom == nil && m.Role() != ""
}

// AgentState contains the full state of an agent.
type AgentState struct {
	SystemPrompt    string
	Model           *ai.Model
	ThinkingLevel   ai.ThinkingLevel
	Tools           []AgentTool
	Messages        []AgentMessage
	IsStreaming      bool
	StreamMessage   *AgentMessage
	PendingToolCalls map[string]struct{}
	Error           string
}

// AgentToolResult is the result of executing a tool.
type AgentToolResult struct {
	Content []ai.Content `json:"content"`
	Details any          `json:"details,omitempty"`
}

// AgentToolUpdateCallback is called with partial results during tool execution.
type AgentToolUpdateCallback func(partialResult AgentToolResult)

// AgentTool extends ai.Tool with a label and execute function.
type AgentTool struct {
	ai.Tool
	Label   string `json:"label"`
	Execute func(ctx context.Context, toolCallID string, params map[string]any, onUpdate AgentToolUpdateCallback) (AgentToolResult, error)
}

// AgentContext bundles the system prompt, messages, and tools for the agent loop.
type AgentContext struct {
	SystemPrompt string
	Messages     []AgentMessage
	Tools        []AgentTool
}

// ---------------------------------------------------------------------------
// Agent events — emitted for UI/observability
// ---------------------------------------------------------------------------

// AgentEventType discriminates agent lifecycle events.
type AgentEventType string

const (
	AgentEventStart          AgentEventType = "agent_start"
	AgentEventEnd            AgentEventType = "agent_end"
	TurnEventStart           AgentEventType = "turn_start"
	TurnEventEnd             AgentEventType = "turn_end"
	MessageEventStart        AgentEventType = "message_start"
	MessageEventUpdate       AgentEventType = "message_update"
	MessageEventEnd          AgentEventType = "message_end"
	ToolExecutionEventStart  AgentEventType = "tool_execution_start"
	ToolExecutionEventUpdate AgentEventType = "tool_execution_update"
	ToolExecutionEventEnd    AgentEventType = "tool_execution_end"
)

// AgentEvent is emitted during the agent loop for lifecycle observability.
type AgentEvent struct {
	Type AgentEventType

	// agent_end
	Messages []AgentMessage

	// message_start, message_update, message_end, turn_end
	Message *AgentMessage

	// message_update
	AssistantMessageEvent *ai.AssistantMessageEvent

	// turn_end
	ToolResults []ai.ToolResultMessage

	// tool_execution_*
	ToolCallID    string
	ToolName      string
	Args          any
	PartialResult any
	Result        any
	IsError       bool
}

// AgentEventStream is an EventStream for agent events with a final result
// of the list of all new messages produced.
type AgentEventStream = ai.EventStream[AgentEvent, []AgentMessage]

// NewAgentEventStream creates a new agent event stream.
func NewAgentEventStream() *AgentEventStream {
	return ai.NewEventStream[AgentEvent, []AgentMessage](
		func(e AgentEvent) bool { return e.Type == AgentEventEnd },
		func(e AgentEvent) []AgentMessage { return e.Messages },
	)
}
