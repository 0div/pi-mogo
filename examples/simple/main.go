package main

import (
	"context"
	"fmt"
	"time"

	"github.com/badlogic/pi-go/pkg/agent"
	"github.com/badlogic/pi-go/pkg/ai"
)

func main() {
	// 1. Register a dummy provider (replace with a real one)
	ai.RegisterApiProvider(&ai.ApiProvider{
		Api: ai.ApiAnthropicMessages,
		StreamSimple: func(model *ai.Model, ctx ai.Context, opts *ai.SimpleStreamOptions) *ai.AssistantMessageEventStream {
			stream := ai.NewAssistantMessageEventStream()
			go func() {
				msg := &ai.AssistantMessage{
					Role:       ai.RoleAssistant,
					Content:    []ai.Content{ai.NewTextContent("Hello! I'm a dummy response.")},
					Api:        model.Api,
					Provider:   model.Provider,
					Model:      model.ID,
					Usage:      ai.Usage{Input: 10, Output: 5, TotalTokens: 15},
					StopReason: ai.StopReasonStop,
					Timestamp:  time.Now().UnixMilli(),
				}
				stream.Push(ai.AssistantMessageEvent{Type: ai.EventStart, Partial: msg})
				stream.Push(ai.AssistantMessageEvent{Type: ai.EventDone, Reason: ai.StopReasonStop, Message: msg})
			}()
			return stream
		},
		Stream: func(model *ai.Model, ctx ai.Context, opts *ai.StreamOptions) *ai.AssistantMessageEventStream {
			return nil // not used in this example
		},
	}, "")

	// 2. Register a model
	model := &ai.Model{
		ID:            "claude-sonnet-4-5-20250929",
		Name:          "Claude Sonnet 4.5",
		Api:           ai.ApiAnthropicMessages,
		Provider:      ai.ProviderAnthropic,
		BaseURL:       "https://api.anthropic.com",
		Reasoning:     true,
		Input:         []string{"text", "image"},
		Cost:          ai.ModelCost{Input: 3.0, Output: 15.0},
		ContextWindow: 200000,
		MaxTokens:     8192,
	}
	ai.RegisterModel(model)

	// 3. Create an agent
	a := agent.NewAgent(agent.AgentOptions{
		StreamFn: func(m *ai.Model, ctx ai.Context, opts *ai.SimpleStreamOptions) *ai.AssistantMessageEventStream {
			s, _ := ai.StreamSimple(m, ctx, opts)
			return s
		},
	})
	a.SetModel(model)
	a.SetSystemPrompt("You are a helpful assistant.")

	// 4. Subscribe to events
	unsub := a.Subscribe(func(e agent.AgentEvent) {
		switch e.Type {
		case agent.MessageEventEnd:
			if e.Message != nil && e.Message.Assistant != nil {
				for _, c := range e.Message.Assistant.Content {
					if c.Text != nil {
						fmt.Printf("Assistant: %s\n", c.Text.Text)
					}
				}
			}
		}
	})
	defer unsub()

	// 5. Send a prompt
	if err := a.Prompt("Hello!"); err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	a.WaitForIdle()

	// 6. You can also use the loop directly
	fmt.Println("\n--- Direct loop usage ---")
	agentCtx := agent.AgentContext{
		SystemPrompt: "You are helpful.",
		Messages: []agent.AgentMessage{
			agent.NewAgentMessageFromMessage(ai.NewUserMessage("What is 2+2?")),
		},
	}
	config := agent.AgentLoopConfig{
		Model:        model,
		ConvertToLLM: agent.DefaultConvertToLLM,
	}
	streamFn := func(m *ai.Model, ctx ai.Context, opts *ai.SimpleStreamOptions) *ai.AssistantMessageEventStream {
		s, _ := ai.StreamSimple(m, ctx, opts)
		return s
	}

	stream := agent.AgentLoop(
		context.Background(),
		[]agent.AgentMessage{agent.NewAgentMessageFromMessage(ai.NewUserMessage("What is 2+2?"))},
		agentCtx,
		config,
		streamFn,
	)

	for event := range stream.Events() {
		if event.Type == agent.MessageEventEnd && event.Message != nil && event.Message.Assistant != nil {
			for _, c := range event.Message.Assistant.Content {
				if c.Text != nil {
					fmt.Printf("Loop response: %s\n", c.Text.Text)
				}
			}
		}
	}
}
