package agent

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/badlogic/pi-go/pkg/ai"
)

// DefaultConvertToLLM keeps only LLM-compatible messages.
func DefaultConvertToLLM(messages []AgentMessage) ([]ai.Message, error) {
	var out []ai.Message
	for _, m := range messages {
		r := m.Role()
		if r == ai.RoleUser || r == ai.RoleAssistant || r == ai.RoleToolResult {
			out = append(out, m.Message)
		}
	}
	return out, nil
}

// AgentOptions configures an Agent.
type AgentOptions struct {
	InitialState     *AgentState
	ConvertToLLM     func([]AgentMessage) ([]ai.Message, error)
	TransformContext func(ctx context.Context, messages []AgentMessage) ([]AgentMessage, error)
	SteeringMode     string // "all" or "one-at-a-time"
	FollowUpMode     string // "all" or "one-at-a-time"
	StreamFn         StreamFn
	SessionID        string
	GetApiKey        func(provider string) (string, error)
	ThinkingBudgets  *ai.ThinkingBudgets
	MaxRetryDelayMs  *int
}

// Agent manages a conversation loop with an LLM.
type Agent struct {
	mu sync.Mutex

	state AgentState

	listeners      map[int]func(AgentEvent)
	nextListenerID int

	abortCancel context.CancelFunc
	abortCtx    context.Context

	convertToLLM     func([]AgentMessage) ([]ai.Message, error)
	transformContext  func(ctx context.Context, messages []AgentMessage) ([]AgentMessage, error)
	steeringQueue    []AgentMessage
	followUpQueue    []AgentMessage
	steeringMode     string
	followUpMode     string
	StreamFn         StreamFn
	sessionID        string
	GetApiKey        func(provider string) (string, error)
	thinkingBudgets  *ai.ThinkingBudgets
	maxRetryDelayMs  *int

	running chan struct{} // closed when current run completes
}

// NewAgent creates a new Agent with the given options.
func NewAgent(opts AgentOptions) *Agent {
	a := &Agent{
		state: AgentState{
			ThinkingLevel:    ai.ThinkingOff,
			PendingToolCalls: map[string]struct{}{},
		},
		listeners:       map[int]func(AgentEvent){},
		convertToLLM:    DefaultConvertToLLM,
		steeringMode:    "one-at-a-time",
		followUpMode:    "one-at-a-time",
	}

	if opts.InitialState != nil {
		a.state = *opts.InitialState
		if a.state.PendingToolCalls == nil {
			a.state.PendingToolCalls = map[string]struct{}{}
		}
	}
	if opts.ConvertToLLM != nil {
		a.convertToLLM = opts.ConvertToLLM
	}
	if opts.TransformContext != nil {
		a.transformContext = opts.TransformContext
	}
	if opts.SteeringMode != "" {
		a.steeringMode = opts.SteeringMode
	}
	if opts.FollowUpMode != "" {
		a.followUpMode = opts.FollowUpMode
	}
	if opts.StreamFn != nil {
		a.StreamFn = opts.StreamFn
	}
	a.sessionID = opts.SessionID
	a.GetApiKey = opts.GetApiKey
	a.thinkingBudgets = opts.ThinkingBudgets
	a.maxRetryDelayMs = opts.MaxRetryDelayMs

	return a
}

// State returns a snapshot of the current agent state.
func (a *Agent) State() AgentState {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.state
}

// Subscribe registers a listener. Returns an unsubscribe function.
func (a *Agent) Subscribe(fn func(AgentEvent)) func() {
	a.mu.Lock()
	defer a.mu.Unlock()
	id := a.nextListenerID
	a.nextListenerID++
	a.listeners[id] = fn
	return func() {
		a.mu.Lock()
		defer a.mu.Unlock()
		delete(a.listeners, id)
	}
}

// SetSystemPrompt sets the system prompt.
func (a *Agent) SetSystemPrompt(v string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.state.SystemPrompt = v
}

// SetModel sets the model.
func (a *Agent) SetModel(m *ai.Model) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.state.Model = m
}

// SetThinkingLevel sets the thinking level.
func (a *Agent) SetThinkingLevel(l ai.ThinkingLevel) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.state.ThinkingLevel = l
}

// SetTools sets the agent tools.
func (a *Agent) SetTools(t []AgentTool) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.state.Tools = t
}

// ReplaceMessages replaces all messages.
func (a *Agent) ReplaceMessages(ms []AgentMessage) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.state.Messages = append([]AgentMessage{}, ms...)
}

// AppendMessage adds a message.
func (a *Agent) AppendMessage(m AgentMessage) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.state.Messages = append(a.state.Messages, m)
}

// Steer queues a steering message to interrupt the agent mid-run.
func (a *Agent) Steer(m AgentMessage) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.steeringQueue = append(a.steeringQueue, m)
}

// FollowUp queues a follow-up message.
func (a *Agent) FollowUp(m AgentMessage) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.followUpQueue = append(a.followUpQueue, m)
}

// ClearSteeringQueue clears the steering queue.
func (a *Agent) ClearSteeringQueue() {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.steeringQueue = nil
}

// ClearFollowUpQueue clears the follow-up queue.
func (a *Agent) ClearFollowUpQueue() {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.followUpQueue = nil
}

// ClearAllQueues clears both queues.
func (a *Agent) ClearAllQueues() {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.steeringQueue = nil
	a.followUpQueue = nil
}

// HasQueuedMessages returns true if there are steering or follow-up messages.
func (a *Agent) HasQueuedMessages() bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	return len(a.steeringQueue) > 0 || len(a.followUpQueue) > 0
}

// ClearMessages clears all messages.
func (a *Agent) ClearMessages() {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.state.Messages = nil
}

// Abort cancels the current run.
func (a *Agent) Abort() {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.abortCancel != nil {
		a.abortCancel()
	}
}

// WaitForIdle blocks until the agent is no longer running.
func (a *Agent) WaitForIdle() {
	a.mu.Lock()
	ch := a.running
	a.mu.Unlock()
	if ch != nil {
		<-ch
	}
}

// Reset clears the agent state.
func (a *Agent) Reset() {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.state.Messages = nil
	a.state.IsStreaming = false
	a.state.StreamMessage = nil
	a.state.PendingToolCalls = map[string]struct{}{}
	a.state.Error = ""
	a.steeringQueue = nil
	a.followUpQueue = nil
}

// Prompt sends a text prompt to the agent.
func (a *Agent) Prompt(text string, images ...ai.ImageContent) error {
	content := []ai.Content{ai.NewTextContent(text)}
	for _, img := range images {
		content = append(content, ai.Content{Image: &img})
	}
	msgs := []AgentMessage{
		NewAgentMessageFromMessage(ai.Message{User: &ai.UserMessage{
			Role:      ai.RoleUser,
			Content:   content,
			Timestamp: time.Now().UnixMilli(),
		}}),
	}
	return a.runLoop(msgs, false)
}

// PromptMessages sends agent messages as a prompt.
func (a *Agent) PromptMessages(msgs []AgentMessage) error {
	return a.runLoop(msgs, false)
}

// Continue resumes from the current context.
func (a *Agent) Continue() error {
	a.mu.Lock()
	if a.state.IsStreaming {
		a.mu.Unlock()
		return fmt.Errorf("agent is already processing")
	}
	if len(a.state.Messages) == 0 {
		a.mu.Unlock()
		return fmt.Errorf("no messages to continue from")
	}
	last := a.state.Messages[len(a.state.Messages)-1]
	a.mu.Unlock()

	if last.Role() == ai.RoleAssistant {
		// Try steering queue first.
		steering := a.dequeueSteeringMessages()
		if len(steering) > 0 {
			return a.runLoop(steering, true)
		}
		followUp := a.dequeueFollowUpMessages()
		if len(followUp) > 0 {
			return a.runLoop(followUp, false)
		}
		return fmt.Errorf("cannot continue from message role: assistant")
	}

	return a.runLoop(nil, false)
}

func (a *Agent) dequeueSteeringMessages() []AgentMessage {
	a.mu.Lock()
	defer a.mu.Unlock()

	if a.steeringMode == "one-at-a-time" {
		if len(a.steeringQueue) > 0 {
			first := a.steeringQueue[0]
			a.steeringQueue = a.steeringQueue[1:]
			return []AgentMessage{first}
		}
		return nil
	}

	out := a.steeringQueue
	a.steeringQueue = nil
	return out
}

func (a *Agent) dequeueFollowUpMessages() []AgentMessage {
	a.mu.Lock()
	defer a.mu.Unlock()

	if a.followUpMode == "one-at-a-time" {
		if len(a.followUpQueue) > 0 {
			first := a.followUpQueue[0]
			a.followUpQueue = a.followUpQueue[1:]
			return []AgentMessage{first}
		}
		return nil
	}

	out := a.followUpQueue
	a.followUpQueue = nil
	return out
}

func (a *Agent) runLoop(messages []AgentMessage, skipInitialSteeringPoll bool) error {
	a.mu.Lock()
	if a.state.IsStreaming {
		a.mu.Unlock()
		return fmt.Errorf("agent is already processing a prompt")
	}
	model := a.state.Model
	if model == nil {
		a.mu.Unlock()
		return fmt.Errorf("no model configured")
	}

	a.running = make(chan struct{})
	a.abortCtx, a.abortCancel = context.WithCancel(context.Background())
	a.state.IsStreaming = true
	a.state.StreamMessage = nil
	a.state.Error = ""

	reasoning := a.state.ThinkingLevel
	if reasoning == ai.ThinkingOff {
		reasoning = ""
	}

	agentCtx := AgentContext{
		SystemPrompt: a.state.SystemPrompt,
		Messages:     append([]AgentMessage{}, a.state.Messages...),
		Tools:        a.state.Tools,
	}

	skipSteering := skipInitialSteeringPoll

	config := AgentLoopConfig{
		SimpleStreamOptions: ai.SimpleStreamOptions{
			StreamOptions: ai.StreamOptions{
				ApiKey:          a.state.SystemPrompt, // Will be overridden by GetApiKey
				MaxRetryDelayMs: a.maxRetryDelayMs,
			},
			Reasoning:       reasoning,
			ThinkingBudgets: a.thinkingBudgets,
		},
		Model:        model,
		ConvertToLLM: a.convertToLLM,
		TransformContext: a.transformContext,
		GetApiKey:    a.GetApiKey,
		GetSteeringMessages: func() ([]AgentMessage, error) {
			if skipSteering {
				skipSteering = false
				return nil, nil
			}
			return a.dequeueSteeringMessages(), nil
		},
		GetFollowUpMessages: func() ([]AgentMessage, error) {
			return a.dequeueFollowUpMessages(), nil
		},
	}
	// Fix: don't use system prompt as API key
	config.SimpleStreamOptions.StreamOptions.ApiKey = ""
	if a.sessionID != "" {
		config.SimpleStreamOptions.StreamOptions.SessionID = a.sessionID
	}

	ctx := a.abortCtx
	a.mu.Unlock()

	var stream *AgentEventStream
	if messages != nil {
		stream = AgentLoop(ctx, messages, agentCtx, config, a.StreamFn)
	} else {
		var err error
		stream, err = AgentLoopContinue(ctx, agentCtx, config, a.StreamFn)
		if err != nil {
			a.mu.Lock()
			a.state.IsStreaming = false
			close(a.running)
			a.mu.Unlock()
			return err
		}
	}

	// Process events in background.
	go func() {
		defer func() {
			a.mu.Lock()
			a.state.IsStreaming = false
			a.state.StreamMessage = nil
			a.state.PendingToolCalls = map[string]struct{}{}
			a.abortCancel = nil
			ch := a.running
			a.running = nil
			a.mu.Unlock()
			close(ch)
		}()

		for event := range stream.Events() {
			a.mu.Lock()
			switch event.Type {
			case MessageEventStart:
				a.state.StreamMessage = event.Message
			case MessageEventUpdate:
				a.state.StreamMessage = event.Message
			case MessageEventEnd:
				a.state.StreamMessage = nil
				a.state.Messages = append(a.state.Messages, *event.Message)
			case ToolExecutionEventStart:
				a.state.PendingToolCalls[event.ToolCallID] = struct{}{}
			case ToolExecutionEventEnd:
				delete(a.state.PendingToolCalls, event.ToolCallID)
			case TurnEventEnd:
				if event.Message != nil && event.Message.Assistant != nil {
					if event.Message.Assistant.ErrorMessage != "" {
						a.state.Error = event.Message.Assistant.ErrorMessage
					}
				}
			case AgentEventEnd:
				a.state.IsStreaming = false
				a.state.StreamMessage = nil
			}
			a.mu.Unlock()

			// Emit to listeners.
			a.mu.Lock()
			listeners := make([]func(AgentEvent), 0, len(a.listeners))
			for _, fn := range a.listeners {
				listeners = append(listeners, fn)
			}
			a.mu.Unlock()
			for _, fn := range listeners {
				fn(event)
			}
		}
	}()

	return nil
}
