package agent

import (
	"context"
	"fmt"
	"time"

	"github.com/badlogic/pi-go/pkg/ai"
)

// AgentLoop starts the agent loop with new prompt messages.
// The prompts are added to the context and events are emitted.
func AgentLoop(
	ctx context.Context,
	prompts []AgentMessage,
	agentCtx AgentContext,
	config AgentLoopConfig,
	streamFn StreamFn,
) *AgentEventStream {
	stream := NewAgentEventStream()

	go func() {
		newMessages := make([]AgentMessage, len(prompts))
		copy(newMessages, prompts)

		currentCtx := AgentContext{
			SystemPrompt: agentCtx.SystemPrompt,
			Messages:     append(append([]AgentMessage{}, agentCtx.Messages...), prompts...),
			Tools:        agentCtx.Tools,
		}

		stream.Push(AgentEvent{Type: AgentEventStart})
		stream.Push(AgentEvent{Type: TurnEventStart})

		for _, p := range prompts {
			pm := p
			stream.Push(AgentEvent{Type: MessageEventStart, Message: &pm})
			stream.Push(AgentEvent{Type: MessageEventEnd, Message: &pm})
		}

		runLoop(ctx, &currentCtx, &newMessages, config, stream, streamFn)
	}()

	return stream
}

// AgentLoopContinue continues the agent loop from existing context
// (used for retries — context already has user message or tool results).
func AgentLoopContinue(
	ctx context.Context,
	agentCtx AgentContext,
	config AgentLoopConfig,
	streamFn StreamFn,
) (*AgentEventStream, error) {
	if len(agentCtx.Messages) == 0 {
		return nil, fmt.Errorf("cannot continue: no messages in context")
	}
	last := agentCtx.Messages[len(agentCtx.Messages)-1]
	if last.Role() == ai.RoleAssistant {
		return nil, fmt.Errorf("cannot continue from message role: assistant")
	}

	stream := NewAgentEventStream()

	go func() {
		newMessages := []AgentMessage{}
		currentCtx := AgentContext{
			SystemPrompt: agentCtx.SystemPrompt,
			Messages:     append([]AgentMessage{}, agentCtx.Messages...),
			Tools:        agentCtx.Tools,
		}

		stream.Push(AgentEvent{Type: AgentEventStart})
		stream.Push(AgentEvent{Type: TurnEventStart})

		runLoop(ctx, &currentCtx, &newMessages, config, stream, streamFn)
	}()

	return stream, nil
}

// runLoop is the shared main loop for AgentLoop and AgentLoopContinue.
func runLoop(
	ctx context.Context,
	currentCtx *AgentContext,
	newMessages *[]AgentMessage,
	config AgentLoopConfig,
	stream *AgentEventStream,
	streamFn StreamFn,
) {
	firstTurn := true

	// Check for steering messages at start.
	var pendingMessages []AgentMessage
	if config.GetSteeringMessages != nil {
		if msgs, err := config.GetSteeringMessages(); err == nil {
			pendingMessages = msgs
		}
	}

	// Outer loop: continues when queued follow-up messages arrive.
	for {
		hasMoreToolCalls := true
		var steeringAfterTools []AgentMessage

		// Inner loop: process tool calls and steering messages.
		for hasMoreToolCalls || len(pendingMessages) > 0 {
			if !firstTurn {
				stream.Push(AgentEvent{Type: TurnEventStart})
			} else {
				firstTurn = false
			}

			// Process pending messages.
			if len(pendingMessages) > 0 {
				for _, msg := range pendingMessages {
					m := msg
					stream.Push(AgentEvent{Type: MessageEventStart, Message: &m})
					stream.Push(AgentEvent{Type: MessageEventEnd, Message: &m})
					currentCtx.Messages = append(currentCtx.Messages, m)
					*newMessages = append(*newMessages, m)
				}
				pendingMessages = nil
			}

			// Stream assistant response.
			message, err := streamAssistantResponse(ctx, currentCtx, config, stream, streamFn)
			if err != nil {
				// Create error message and end.
				errMsg := makeErrorAssistantMessage(config.Model, err.Error())
				am := NewAgentMessageFromMessage(ai.Message{Assistant: errMsg})
				*newMessages = append(*newMessages, am)
				stream.Push(AgentEvent{Type: TurnEventEnd, Message: &am, ToolResults: nil})
				stream.Push(AgentEvent{Type: AgentEventEnd, Messages: *newMessages})
				stream.End(*newMessages)
				return
			}

			am := NewAgentMessageFromMessage(ai.Message{Assistant: message})
			*newMessages = append(*newMessages, am)

			if message.StopReason == ai.StopReasonError || message.StopReason == ai.StopReasonAborted {
				stream.Push(AgentEvent{Type: TurnEventEnd, Message: &am, ToolResults: nil})
				stream.Push(AgentEvent{Type: AgentEventEnd, Messages: *newMessages})
				stream.End(*newMessages)
				return
			}

			// Check for tool calls.
			var toolCalls []ai.ToolCall
			for _, c := range message.Content {
				if c.ToolCall != nil {
					toolCalls = append(toolCalls, *c.ToolCall)
				}
			}
			hasMoreToolCalls = len(toolCalls) > 0

			var toolResults []ai.ToolResultMessage
			if hasMoreToolCalls {
				results, steering := executeToolCalls(ctx, currentCtx.Tools, message, stream, config.GetSteeringMessages)
				toolResults = results
				steeringAfterTools = steering

				for _, r := range toolResults {
					trMsg := NewAgentMessageFromMessage(ai.Message{ToolResult: &r})
					currentCtx.Messages = append(currentCtx.Messages, trMsg)
					*newMessages = append(*newMessages, trMsg)
				}
			}

			stream.Push(AgentEvent{Type: TurnEventEnd, Message: &am, ToolResults: toolResults})

			// Get steering messages after turn completes.
			if len(steeringAfterTools) > 0 {
				pendingMessages = steeringAfterTools
				steeringAfterTools = nil
			} else if config.GetSteeringMessages != nil {
				if msgs, err := config.GetSteeringMessages(); err == nil {
					pendingMessages = msgs
				}
			}
		}

		// Agent would stop here. Check for follow-up messages.
		if config.GetFollowUpMessages != nil {
			if followUp, err := config.GetFollowUpMessages(); err == nil && len(followUp) > 0 {
				pendingMessages = followUp
				continue
			}
		}

		break
	}

	stream.Push(AgentEvent{Type: AgentEventEnd, Messages: *newMessages})
	stream.End(*newMessages)
}

// streamAssistantResponse streams a single LLM response, transforming
// AgentMessages to LLM Messages at the boundary.
func streamAssistantResponse(
	ctx context.Context,
	agentCtx *AgentContext,
	config AgentLoopConfig,
	stream *AgentEventStream,
	streamFn StreamFn,
) (*ai.AssistantMessage, error) {
	messages := agentCtx.Messages

	// Apply context transform if configured.
	if config.TransformContext != nil {
		var err error
		messages, err = config.TransformContext(ctx, messages)
		if err != nil {
			return nil, fmt.Errorf("transformContext: %w", err)
		}
	}

	// Convert to LLM messages.
	llmMessages, err := config.ConvertToLLM(messages)
	if err != nil {
		return nil, fmt.Errorf("convertToLLM: %w", err)
	}

	// Build LLM context.
	llmCtx := ai.Context{
		SystemPrompt: agentCtx.SystemPrompt,
		Messages:     llmMessages,
	}

	// Convert AgentTools to ai.Tools.
	if len(agentCtx.Tools) > 0 {
		tools := make([]ai.Tool, len(agentCtx.Tools))
		for i, t := range agentCtx.Tools {
			tools[i] = t.Tool
		}
		llmCtx.Tools = tools
	}

	sf := streamFn
	if sf == nil {
		return nil, fmt.Errorf("no stream function provided")
	}

	// Resolve API key.
	opts := config.SimpleStreamOptions
	if config.GetApiKey != nil {
		key, err := config.GetApiKey(config.Model.Provider)
		if err == nil && key != "" {
			opts.ApiKey = key
		}
	}

	response := sf(config.Model, llmCtx, &opts)

	var partialMessage *ai.AssistantMessage
	addedPartial := false

	for event := range response.Events() {
		switch event.Type {
		case ai.EventStart:
			partialMessage = event.Partial
			agentCtx.Messages = append(agentCtx.Messages, NewAgentMessageFromMessage(ai.Message{Assistant: partialMessage}))
			addedPartial = true
			am := NewAgentMessageFromMessage(ai.Message{Assistant: cloneAssistant(partialMessage)})
			stream.Push(AgentEvent{Type: MessageEventStart, Message: &am})

		case ai.EventTextStart, ai.EventTextDelta, ai.EventTextEnd,
			ai.EventThinkingStart, ai.EventThinkingDelta, ai.EventThinkingEnd,
			ai.EventToolCallStart, ai.EventToolCallDelta, ai.EventToolCallEnd:
			if partialMessage != nil {
				partialMessage = event.Partial
				if addedPartial {
					agentCtx.Messages[len(agentCtx.Messages)-1] = NewAgentMessageFromMessage(ai.Message{Assistant: partialMessage})
				}
				am := NewAgentMessageFromMessage(ai.Message{Assistant: cloneAssistant(partialMessage)})
				stream.Push(AgentEvent{Type: MessageEventUpdate, AssistantMessageEvent: &event, Message: &am})
			}

		case ai.EventDone, ai.EventError:
			finalMessage := response.Result()
			if addedPartial {
				agentCtx.Messages[len(agentCtx.Messages)-1] = NewAgentMessageFromMessage(ai.Message{Assistant: finalMessage})
			} else {
				agentCtx.Messages = append(agentCtx.Messages, NewAgentMessageFromMessage(ai.Message{Assistant: finalMessage}))
			}
			if !addedPartial {
				am := NewAgentMessageFromMessage(ai.Message{Assistant: cloneAssistant(finalMessage)})
				stream.Push(AgentEvent{Type: MessageEventStart, Message: &am})
			}
			fam := NewAgentMessageFromMessage(ai.Message{Assistant: finalMessage})
			stream.Push(AgentEvent{Type: MessageEventEnd, Message: &fam})
			return finalMessage, nil
		}
	}

	return response.Result(), nil
}

// executeToolCalls runs tool calls sequentially, checking for steering after each.
func executeToolCalls(
	ctx context.Context,
	tools []AgentTool,
	assistantMsg *ai.AssistantMessage,
	stream *AgentEventStream,
	getSteeringMessages func() ([]AgentMessage, error),
) ([]ai.ToolResultMessage, []AgentMessage) {
	var toolCalls []ai.ToolCall
	for _, c := range assistantMsg.Content {
		if c.ToolCall != nil {
			toolCalls = append(toolCalls, *c.ToolCall)
		}
	}

	var results []ai.ToolResultMessage
	var steeringMessages []AgentMessage

	for i, tc := range toolCalls {
		tool := findTool(tools, tc.Name)

		stream.Push(AgentEvent{
			Type:       ToolExecutionEventStart,
			ToolCallID: tc.ID,
			ToolName:   tc.Name,
			Args:       tc.Arguments,
		})

		var result AgentToolResult
		var isError bool

		if tool == nil {
			result = AgentToolResult{
				Content: []ai.Content{ai.NewTextContent(fmt.Sprintf("Tool %s not found", tc.Name))},
			}
			isError = true
		} else {
			// Validate arguments.
			args, err := ai.ValidateToolArguments(&tool.Tool, tc)
			if err != nil {
				result = AgentToolResult{
					Content: []ai.Content{ai.NewTextContent(err.Error())},
				}
				isError = true
			} else {
				onUpdate := func(partial AgentToolResult) {
					stream.Push(AgentEvent{
						Type:          ToolExecutionEventUpdate,
						ToolCallID:    tc.ID,
						ToolName:      tc.Name,
						Args:          tc.Arguments,
						PartialResult: partial,
					})
				}

				execResult, err := tool.Execute(ctx, tc.ID, args, onUpdate)
				if err != nil {
					result = AgentToolResult{
						Content: []ai.Content{ai.NewTextContent(err.Error())},
					}
					isError = true
				} else {
					result = execResult
				}
			}
		}

		stream.Push(AgentEvent{
			Type:       ToolExecutionEventEnd,
			ToolCallID: tc.ID,
			ToolName:   tc.Name,
			Result:     result,
			IsError:    isError,
		})

		trMsg := ai.ToolResultMessage{
			Role:       ai.RoleToolResult,
			ToolCallID: tc.ID,
			ToolName:   tc.Name,
			Content:    result.Content,
			Details:    result.Details,
			IsError:    isError,
			Timestamp:  time.Now().UnixMilli(),
		}
		results = append(results, trMsg)

		am := NewAgentMessageFromMessage(ai.Message{ToolResult: &trMsg})
		stream.Push(AgentEvent{Type: MessageEventStart, Message: &am})
		stream.Push(AgentEvent{Type: MessageEventEnd, Message: &am})

		// Check for steering messages — skip remaining tools if user interrupted.
		if getSteeringMessages != nil {
			if steering, err := getSteeringMessages(); err == nil && len(steering) > 0 {
				steeringMessages = steering
				for _, skipped := range toolCalls[i+1:] {
					results = append(results, skipToolCall(skipped, stream))
				}
				break
			}
		}
	}

	return results, steeringMessages
}

func skipToolCall(tc ai.ToolCall, stream *AgentEventStream) ai.ToolResultMessage {
	result := AgentToolResult{
		Content: []ai.Content{ai.NewTextContent("Skipped due to queued user message.")},
	}

	stream.Push(AgentEvent{
		Type:       ToolExecutionEventStart,
		ToolCallID: tc.ID,
		ToolName:   tc.Name,
		Args:       tc.Arguments,
	})
	stream.Push(AgentEvent{
		Type:       ToolExecutionEventEnd,
		ToolCallID: tc.ID,
		ToolName:   tc.Name,
		Result:     result,
		IsError:    true,
	})

	trMsg := ai.ToolResultMessage{
		Role:       ai.RoleToolResult,
		ToolCallID: tc.ID,
		ToolName:   tc.Name,
		Content:    result.Content,
		IsError:    true,
		Timestamp:  time.Now().UnixMilli(),
	}

	am := NewAgentMessageFromMessage(ai.Message{ToolResult: &trMsg})
	stream.Push(AgentEvent{Type: MessageEventStart, Message: &am})
	stream.Push(AgentEvent{Type: MessageEventEnd, Message: &am})

	return trMsg
}

func findTool(tools []AgentTool, name string) *AgentTool {
	for i := range tools {
		if tools[i].Name == name {
			return &tools[i]
		}
	}
	return nil
}

func makeErrorAssistantMessage(model *ai.Model, errMsg string) *ai.AssistantMessage {
	return &ai.AssistantMessage{
		Role:    ai.RoleAssistant,
		Content: []ai.Content{ai.NewTextContent("")},
		Api:     model.Api,
		Provider: model.Provider,
		Model:    model.ID,
		Usage: ai.Usage{
			Cost: ai.Cost{},
		},
		StopReason:   ai.StopReasonError,
		ErrorMessage: errMsg,
		Timestamp:    time.Now().UnixMilli(),
	}
}

func cloneAssistant(m *ai.AssistantMessage) *ai.AssistantMessage {
	if m == nil {
		return nil
	}
	clone := *m
	clone.Content = make([]ai.Content, len(m.Content))
	copy(clone.Content, m.Content)
	return &clone
}
