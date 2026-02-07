package agent

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/badlogic/pi-go/pkg/ai"
)

// ProxyStreamOptions configures a proxy stream call.
type ProxyStreamOptions struct {
	ai.SimpleStreamOptions
	AuthToken string
	ProxyURL  string
}

// ProxyAssistantMessageEvent is the wire format sent by the proxy server
// (partial field stripped to reduce bandwidth).
type ProxyAssistantMessageEvent struct {
	Type             string    `json:"type"`
	ContentIndex     int       `json:"contentIndex,omitempty"`
	Delta            string    `json:"delta,omitempty"`
	ID               string    `json:"id,omitempty"`
	ToolName         string    `json:"toolName,omitempty"`
	ContentSignature string    `json:"contentSignature,omitempty"`
	Reason           string    `json:"reason,omitempty"`
	ErrorMessage     string    `json:"errorMessage,omitempty"`
	Usage            *ai.Usage `json:"usage,omitempty"`
}

// StreamProxy is a StreamFn that routes LLM calls through a proxy server.
func StreamProxy(model *ai.Model, ctx ai.Context, opts *ProxyStreamOptions) *ai.AssistantMessageEventStream {
	stream := ai.NewAssistantMessageEventStream()

	go func() {
		partial := &ai.AssistantMessage{
			Role:       ai.RoleAssistant,
			StopReason: ai.StopReasonStop,
			Content:    []ai.Content{},
			Api:        model.Api,
			Provider:   model.Provider,
			Model:      model.ID,
			Usage:      ai.Usage{},
			Timestamp:  time.Now().UnixMilli(),
		}

		body := map[string]any{
			"model":   model,
			"context": ctx,
			"options": map[string]any{
				"temperature": opts.Temperature,
				"maxTokens":   opts.MaxTokens,
				"reasoning":   opts.Reasoning,
			},
		}
		bodyJSON, err := json.Marshal(body)
		if err != nil {
			emitProxyError(stream, partial, fmt.Sprintf("marshal error: %v", err))
			return
		}

		req, err := http.NewRequest("POST", opts.ProxyURL+"/api/stream", strings.NewReader(string(bodyJSON)))
		if err != nil {
			emitProxyError(stream, partial, fmt.Sprintf("request error: %v", err))
			return
		}
		req.Header.Set("Authorization", "Bearer "+opts.AuthToken)
		req.Header.Set("Content-Type", "application/json")

		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			emitProxyError(stream, partial, fmt.Sprintf("request failed: %v", err))
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			bodyBytes, _ := io.ReadAll(resp.Body)
			var errData struct {
				Error string `json:"error"`
			}
			errMsg := fmt.Sprintf("Proxy error: %d %s", resp.StatusCode, resp.Status)
			if json.Unmarshal(bodyBytes, &errData) == nil && errData.Error != "" {
				errMsg = fmt.Sprintf("Proxy error: %s", errData.Error)
			}
			emitProxyError(stream, partial, errMsg)
			return
		}

		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			line := scanner.Text()
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			data := strings.TrimSpace(line[6:])
			if data == "" {
				continue
			}

			var proxyEvent ProxyAssistantMessageEvent
			if err := json.Unmarshal([]byte(data), &proxyEvent); err != nil {
				continue
			}

			event := processProxyEvent(&proxyEvent, partial)
			if event != nil {
				stream.Push(*event)
			}
		}

		stream.End(partial)
	}()

	return stream
}

func processProxyEvent(pe *ProxyAssistantMessageEvent, partial *ai.AssistantMessage) *ai.AssistantMessageEvent {
	ensureContentIndex(partial, pe.ContentIndex)

	switch pe.Type {
	case "start":
		return &ai.AssistantMessageEvent{Type: ai.EventStart, Partial: partial}

	case "text_start":
		partial.Content[pe.ContentIndex] = ai.NewTextContent("")
		return &ai.AssistantMessageEvent{Type: ai.EventTextStart, ContentIndex: pe.ContentIndex, Partial: partial}

	case "text_delta":
		c := partial.Content[pe.ContentIndex]
		if c.Text != nil {
			c.Text.Text += pe.Delta
			partial.Content[pe.ContentIndex] = c
			return &ai.AssistantMessageEvent{Type: ai.EventTextDelta, ContentIndex: pe.ContentIndex, Delta: pe.Delta, Partial: partial}
		}
		return nil

	case "text_end":
		c := partial.Content[pe.ContentIndex]
		if c.Text != nil {
			c.Text.TextSignature = pe.ContentSignature
			partial.Content[pe.ContentIndex] = c
			return &ai.AssistantMessageEvent{Type: ai.EventTextEnd, ContentIndex: pe.ContentIndex, Content: c.Text.Text, Partial: partial}
		}
		return nil

	case "thinking_start":
		partial.Content[pe.ContentIndex] = ai.NewThinkingContent("")
		return &ai.AssistantMessageEvent{Type: ai.EventThinkingStart, ContentIndex: pe.ContentIndex, Partial: partial}

	case "thinking_delta":
		c := partial.Content[pe.ContentIndex]
		if c.Thinking != nil {
			c.Thinking.Thinking += pe.Delta
			partial.Content[pe.ContentIndex] = c
			return &ai.AssistantMessageEvent{Type: ai.EventThinkingDelta, ContentIndex: pe.ContentIndex, Delta: pe.Delta, Partial: partial}
		}
		return nil

	case "thinking_end":
		c := partial.Content[pe.ContentIndex]
		if c.Thinking != nil {
			c.Thinking.ThinkingSignature = pe.ContentSignature
			partial.Content[pe.ContentIndex] = c
			return &ai.AssistantMessageEvent{Type: ai.EventThinkingEnd, ContentIndex: pe.ContentIndex, Content: c.Thinking.Thinking, Partial: partial}
		}
		return nil

	case "toolcall_start":
		partial.Content[pe.ContentIndex] = ai.NewToolCallContent(pe.ID, pe.ToolName, map[string]any{})
		return &ai.AssistantMessageEvent{Type: ai.EventToolCallStart, ContentIndex: pe.ContentIndex, Partial: partial}

	case "toolcall_delta":
		c := partial.Content[pe.ContentIndex]
		if c.ToolCall != nil {
			// Parse partial JSON for arguments.
			c.ToolCall.Arguments = ai.ParseStreamingJSON(pe.Delta)
			partial.Content[pe.ContentIndex] = c
			return &ai.AssistantMessageEvent{Type: ai.EventToolCallDelta, ContentIndex: pe.ContentIndex, Delta: pe.Delta, Partial: partial}
		}
		return nil

	case "toolcall_end":
		c := partial.Content[pe.ContentIndex]
		if c.ToolCall != nil {
			return &ai.AssistantMessageEvent{Type: ai.EventToolCallEnd, ContentIndex: pe.ContentIndex, ToolCallData: c.ToolCall, Partial: partial}
		}
		return nil

	case "done":
		partial.StopReason = ai.StopReason(pe.Reason)
		if pe.Usage != nil {
			partial.Usage = *pe.Usage
		}
		return &ai.AssistantMessageEvent{Type: ai.EventDone, Reason: ai.StopReason(pe.Reason), Message: partial}

	case "error":
		partial.StopReason = ai.StopReason(pe.Reason)
		partial.ErrorMessage = pe.ErrorMessage
		if pe.Usage != nil {
			partial.Usage = *pe.Usage
		}
		return &ai.AssistantMessageEvent{Type: ai.EventError, Reason: ai.StopReason(pe.Reason), Error: partial}
	}

	return nil
}

func ensureContentIndex(msg *ai.AssistantMessage, idx int) {
	for len(msg.Content) <= idx {
		msg.Content = append(msg.Content, ai.Content{})
	}
}

func emitProxyError(stream *ai.AssistantMessageEventStream, partial *ai.AssistantMessage, errMsg string) {
	partial.StopReason = ai.StopReasonError
	partial.ErrorMessage = errMsg
	stream.Push(ai.AssistantMessageEvent{
		Type:   ai.EventError,
		Reason: ai.StopReasonError,
		Error:  partial,
	})
	stream.End(partial)
}
