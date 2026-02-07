package ai

import (
	"encoding/json"
	"time"
)

// Api identifies the wire-protocol used to talk to a provider.
type Api string

const (
	ApiOpenAICompletions     Api = "openai-completions"
	ApiOpenAIResponses       Api = "openai-responses"
	ApiAzureOpenAIResponses  Api = "azure-openai-responses"
	ApiOpenAICodexResponses  Api = "openai-codex-responses"
	ApiAnthropicMessages     Api = "anthropic-messages"
	ApiBedrockConverseStream Api = "bedrock-converse-stream"
	ApiGoogleGenerativeAI    Api = "google-generative-ai"
	ApiGoogleGeminiCLI       Api = "google-gemini-cli"
	ApiGoogleVertex          Api = "google-vertex"
)

// Provider identifies the upstream service (may host multiple APIs).
type Provider = string

const (
	ProviderAmazonBedrock       Provider = "amazon-bedrock"
	ProviderAnthropic           Provider = "anthropic"
	ProviderGoogle              Provider = "google"
	ProviderGoogleGeminiCLI     Provider = "google-gemini-cli"
	ProviderGoogleAntigravity   Provider = "google-antigravity"
	ProviderGoogleVertex        Provider = "google-vertex"
	ProviderOpenAI              Provider = "openai"
	ProviderAzureOpenAIResp     Provider = "azure-openai-responses"
	ProviderOpenAICodex         Provider = "openai-codex"
	ProviderGitHubCopilot       Provider = "github-copilot"
	ProviderXAI                 Provider = "xai"
	ProviderGroq                Provider = "groq"
	ProviderCerebras            Provider = "cerebras"
	ProviderOpenRouter          Provider = "openrouter"
	ProviderVercelAIGateway     Provider = "vercel-ai-gateway"
	ProviderZAI                 Provider = "zai"
	ProviderMistral             Provider = "mistral"
	ProviderMinimax             Provider = "minimax"
	ProviderMinimaxCN           Provider = "minimax-cn"
	ProviderHuggingface         Provider = "huggingface"
	ProviderOpenCode            Provider = "opencode"
	ProviderKimiCoding          Provider = "kimi-coding"
)

// ThinkingLevel controls reasoning effort for models that support it.
type ThinkingLevel string

const (
	ThinkingOff     ThinkingLevel = "off"
	ThinkingMinimal ThinkingLevel = "minimal"
	ThinkingLow     ThinkingLevel = "low"
	ThinkingMedium  ThinkingLevel = "medium"
	ThinkingHigh    ThinkingLevel = "high"
	ThinkingXHigh   ThinkingLevel = "xhigh"
)

// ThinkingBudgets maps thinking levels to token budgets.
type ThinkingBudgets struct {
	Minimal *int `json:"minimal,omitempty"`
	Low     *int `json:"low,omitempty"`
	Medium  *int `json:"medium,omitempty"`
	High    *int `json:"high,omitempty"`
}

// CacheRetention controls prompt cache behaviour.
type CacheRetention string

const (
	CacheNone  CacheRetention = "none"
	CacheShort CacheRetention = "short"
	CacheLong  CacheRetention = "long"
)

// StreamOptions are the common options shared by all providers.
type StreamOptions struct {
	Temperature     *float64          `json:"temperature,omitempty"`
	MaxTokens       *int              `json:"maxTokens,omitempty"`
	ApiKey          string            `json:"apiKey,omitempty"`
	CacheRetention  CacheRetention    `json:"cacheRetention,omitempty"`
	SessionID       string            `json:"sessionId,omitempty"`
	Headers         map[string]string `json:"headers,omitempty"`
	MaxRetryDelayMs *int              `json:"maxRetryDelayMs,omitempty"`
}

// SimpleStreamOptions extends StreamOptions with reasoning controls.
type SimpleStreamOptions struct {
	StreamOptions
	Reasoning       ThinkingLevel    `json:"reasoning,omitempty"`
	ThinkingBudgets *ThinkingBudgets `json:"thinkingBudgets,omitempty"`
}

// ---------------------------------------------------------------------------
// Content types
// ---------------------------------------------------------------------------

// ContentType discriminates the union members inside a message.
type ContentType string

const (
	ContentText     ContentType = "text"
	ContentThinking ContentType = "thinking"
	ContentImage    ContentType = "image"
	ContentToolCall ContentType = "toolCall"
)

// TextContent is a text block in a message.
type TextContent struct {
	Type          ContentType `json:"type"` // always "text"
	Text          string      `json:"text"`
	TextSignature string      `json:"textSignature,omitempty"`
}

// ThinkingContent is a reasoning block in a message.
type ThinkingContent struct {
	Type              ContentType `json:"type"` // always "thinking"
	Thinking          string      `json:"thinking"`
	ThinkingSignature string      `json:"thinkingSignature,omitempty"`
}

// ImageContent is a base64-encoded image in a message.
type ImageContent struct {
	Type     ContentType `json:"type"` // always "image"
	Data     string      `json:"data"`
	MimeType string      `json:"mimeType"`
}

// ToolCall is a tool invocation requested by the assistant.
type ToolCall struct {
	Type             ContentType            `json:"type"` // always "toolCall"
	ID               string                 `json:"id"`
	Name             string                 `json:"name"`
	Arguments        map[string]any `json:"arguments"`
	ThoughtSignature string                 `json:"thoughtSignature,omitempty"`
}

// Content is a union type for content blocks.
// Exactly one of the pointer fields is non-nil.
type Content struct {
	Text     *TextContent     `json:"-"`
	Thinking *ThinkingContent `json:"-"`
	Image    *ImageContent    `json:"-"`
	ToolCall *ToolCall        `json:"-"`
}

// ContentType returns the discriminator for this content block.
func (c Content) ContentType() ContentType {
	switch {
	case c.Text != nil:
		return ContentText
	case c.Thinking != nil:
		return ContentThinking
	case c.Image != nil:
		return ContentImage
	case c.ToolCall != nil:
		return ContentToolCall
	default:
		return ""
	}
}

// MarshalJSON encodes the non-nil variant.
func (c Content) MarshalJSON() ([]byte, error) {
	switch {
	case c.Text != nil:
		return json.Marshal(c.Text)
	case c.Thinking != nil:
		return json.Marshal(c.Thinking)
	case c.Image != nil:
		return json.Marshal(c.Image)
	case c.ToolCall != nil:
		return json.Marshal(c.ToolCall)
	default:
		return []byte("null"), nil
	}
}

// UnmarshalJSON decodes a content block by inspecting the "type" field.
func (c *Content) UnmarshalJSON(data []byte) error {
	var raw struct {
		Type ContentType `json:"type"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	switch raw.Type {
	case ContentText:
		c.Text = &TextContent{}
		return json.Unmarshal(data, c.Text)
	case ContentThinking:
		c.Thinking = &ThinkingContent{}
		return json.Unmarshal(data, c.Thinking)
	case ContentImage:
		c.Image = &ImageContent{}
		return json.Unmarshal(data, c.Image)
	case ContentToolCall:
		c.ToolCall = &ToolCall{}
		return json.Unmarshal(data, c.ToolCall)
	default:
		return nil
	}
}

// Helper constructors for Content.

func NewTextContent(text string) Content {
	return Content{Text: &TextContent{Type: ContentText, Text: text}}
}

func NewThinkingContent(thinking string) Content {
	return Content{Thinking: &ThinkingContent{Type: ContentThinking, Thinking: thinking}}
}

func NewImageContent(data, mimeType string) Content {
	return Content{Image: &ImageContent{Type: ContentImage, Data: data, MimeType: mimeType}}
}

func NewToolCallContent(id, name string, args map[string]any) Content {
	return Content{ToolCall: &ToolCall{Type: ContentToolCall, ID: id, Name: name, Arguments: args}}
}

// ---------------------------------------------------------------------------
// Usage & cost
// ---------------------------------------------------------------------------

// Cost tracks monetary cost by category.
type Cost struct {
	Input      float64 `json:"input"`
	Output     float64 `json:"output"`
	CacheRead  float64 `json:"cacheRead"`
	CacheWrite float64 `json:"cacheWrite"`
	Total      float64 `json:"total"`
}

// Usage records token counts and cost for a single response.
type Usage struct {
	Input       int  `json:"input"`
	Output      int  `json:"output"`
	CacheRead   int  `json:"cacheRead"`
	CacheWrite  int  `json:"cacheWrite"`
	TotalTokens int  `json:"totalTokens"`
	Cost        Cost `json:"cost"`
}

// StopReason indicates why the model stopped generating.
type StopReason string

const (
	StopReasonStop    StopReason = "stop"
	StopReasonLength  StopReason = "length"
	StopReasonToolUse StopReason = "toolUse"
	StopReasonError   StopReason = "error"
	StopReasonAborted StopReason = "aborted"
)

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

// MessageRole discriminates the three message types.
type MessageRole string

const (
	RoleUser       MessageRole = "user"
	RoleAssistant  MessageRole = "assistant"
	RoleToolResult MessageRole = "toolResult"
)

// UserMessage is a message from the user.
type UserMessage struct {
	Role      MessageRole `json:"role"` // always "user"
	Content   []Content   `json:"content"`
	Timestamp int64       `json:"timestamp"` // Unix ms
}

// AssistantMessage is a response from the model.
type AssistantMessage struct {
	Role         MessageRole `json:"role"` // always "assistant"
	Content      []Content   `json:"content"`
	Api          Api         `json:"api"`
	Provider     Provider    `json:"provider"`
	Model        string      `json:"model"`
	Usage        Usage       `json:"usage"`
	StopReason   StopReason  `json:"stopReason"`
	ErrorMessage string      `json:"errorMessage,omitempty"`
	Timestamp    int64       `json:"timestamp"` // Unix ms
}

// ToolResultMessage is the result of a tool execution.
type ToolResultMessage struct {
	Role       MessageRole `json:"role"` // always "toolResult"
	ToolCallID string      `json:"toolCallId"`
	ToolName   string      `json:"toolName"`
	Content    []Content   `json:"content"`
	Details    any `json:"details,omitempty"`
	IsError    bool        `json:"isError"`
	Timestamp  int64       `json:"timestamp"` // Unix ms
}

// Message is a union type; exactly one pointer field is non-nil.
type Message struct {
	User      *UserMessage       `json:"-"`
	Assistant *AssistantMessage   `json:"-"`
	ToolResult *ToolResultMessage `json:"-"`
}

// Role returns the role of whichever variant is set.
func (m Message) Role() MessageRole {
	switch {
	case m.User != nil:
		return RoleUser
	case m.Assistant != nil:
		return RoleAssistant
	case m.ToolResult != nil:
		return RoleToolResult
	default:
		return ""
	}
}

func (m Message) MarshalJSON() ([]byte, error) {
	switch {
	case m.User != nil:
		return json.Marshal(m.User)
	case m.Assistant != nil:
		return json.Marshal(m.Assistant)
	case m.ToolResult != nil:
		return json.Marshal(m.ToolResult)
	default:
		return []byte("null"), nil
	}
}

func (m *Message) UnmarshalJSON(data []byte) error {
	var raw struct {
		Role MessageRole `json:"role"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	switch raw.Role {
	case RoleUser:
		m.User = &UserMessage{}
		return json.Unmarshal(data, m.User)
	case RoleAssistant:
		m.Assistant = &AssistantMessage{}
		return json.Unmarshal(data, m.Assistant)
	case RoleToolResult:
		m.ToolResult = &ToolResultMessage{}
		return json.Unmarshal(data, m.ToolResult)
	default:
		return nil
	}
}

// Convenience constructors for Message.

func NewUserMessage(text string) Message {
	return Message{User: &UserMessage{
		Role:      RoleUser,
		Content:   []Content{NewTextContent(text)},
		Timestamp: time.Now().UnixMilli(),
	}}
}

func NewUserMessageWithContent(content []Content) Message {
	return Message{User: &UserMessage{
		Role:      RoleUser,
		Content:   content,
		Timestamp: time.Now().UnixMilli(),
	}}
}

// ---------------------------------------------------------------------------
// Tool definition
// ---------------------------------------------------------------------------

// ToolSchema is a JSON-Schema-like object describing tool parameters.
// In Go we use map[string]any since the actual schema is provider-specific.
type ToolSchema = map[string]any

// Tool describes a function the model can call.
type Tool struct {
	Name        string     `json:"name"`
	Description string     `json:"description"`
	Parameters  ToolSchema `json:"parameters"`
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

// Context bundles a system prompt, messages, and tools for a single LLM call.
type Context struct {
	SystemPrompt string    `json:"systemPrompt,omitempty"`
	Messages     []Message `json:"messages"`
	Tools        []Tool    `json:"tools,omitempty"`
}

// ---------------------------------------------------------------------------
// Model definition
// ---------------------------------------------------------------------------

// ModelCost contains per-million-token pricing.
type ModelCost struct {
	Input      float64 `json:"input"`
	Output     float64 `json:"output"`
	CacheRead  float64 `json:"cacheRead"`
	CacheWrite float64 `json:"cacheWrite"`
}

// Model describes a specific LLM endpoint.
type Model struct {
	ID            string            `json:"id"`
	Name          string            `json:"name"`
	Api           Api               `json:"api"`
	Provider      Provider          `json:"provider"`
	BaseURL       string            `json:"baseUrl"`
	Reasoning     bool              `json:"reasoning"`
	Input         []string          `json:"input"` // "text", "image"
	Cost          ModelCost         `json:"cost"`
	ContextWindow int               `json:"contextWindow"`
	MaxTokens     int               `json:"maxTokens"`
	Headers       map[string]string `json:"headers,omitempty"`
}

// ---------------------------------------------------------------------------
// AssistantMessageEvent — stream events emitted by providers
// ---------------------------------------------------------------------------

// AssistantMessageEventType discriminates stream events.
type AssistantMessageEventType string

const (
	EventStart         AssistantMessageEventType = "start"
	EventTextStart     AssistantMessageEventType = "text_start"
	EventTextDelta     AssistantMessageEventType = "text_delta"
	EventTextEnd       AssistantMessageEventType = "text_end"
	EventThinkingStart AssistantMessageEventType = "thinking_start"
	EventThinkingDelta AssistantMessageEventType = "thinking_delta"
	EventThinkingEnd   AssistantMessageEventType = "thinking_end"
	EventToolCallStart AssistantMessageEventType = "toolcall_start"
	EventToolCallDelta AssistantMessageEventType = "toolcall_delta"
	EventToolCallEnd   AssistantMessageEventType = "toolcall_end"
	EventDone          AssistantMessageEventType = "done"
	EventError         AssistantMessageEventType = "error"
)

// AssistantMessageEvent is a single event from a streaming LLM response.
type AssistantMessageEvent struct {
	Type         AssistantMessageEventType `json:"type"`
	ContentIndex int                       `json:"contentIndex,omitempty"`
	Delta        string                    `json:"delta,omitempty"`
	Content      string                    `json:"content,omitempty"` // used in text_end / thinking_end
	Partial      *AssistantMessage         `json:"partial,omitempty"`
	Message      *AssistantMessage         `json:"message,omitempty"` // used in done
	Error        *AssistantMessage         `json:"error,omitempty"`   // used in error
	ToolCallData *ToolCall                 `json:"toolCall,omitempty"`
	Reason       StopReason                `json:"reason,omitempty"`
}
