package ai

import "fmt"

// Stream starts a streaming LLM call using the provider-level API.
func Stream(model *Model, ctx Context, opts *StreamOptions) (*AssistantMessageEventStream, error) {
	p := GetApiProvider(model.Api)
	if p == nil {
		return nil, fmt.Errorf("no API provider registered for api: %s", model.Api)
	}
	return p.Stream(model, ctx, opts), nil
}

// Complete performs a streaming call and blocks until the final message.
func Complete(model *Model, ctx Context, opts *StreamOptions) (*AssistantMessage, error) {
	s, err := Stream(model, ctx, opts)
	if err != nil {
		return nil, err
	}
	return s.Result(), nil
}

// StreamSimple starts a streaming call with reasoning options.
func StreamSimple(model *Model, ctx Context, opts *SimpleStreamOptions) (*AssistantMessageEventStream, error) {
	p := GetApiProvider(model.Api)
	if p == nil {
		return nil, fmt.Errorf("no API provider registered for api: %s", model.Api)
	}
	return p.StreamSimple(model, ctx, opts), nil
}

// CompleteSimple performs a simple streaming call and blocks until the final message.
func CompleteSimple(model *Model, ctx Context, opts *SimpleStreamOptions) (*AssistantMessage, error) {
	s, err := StreamSimple(model, ctx, opts)
	if err != nil {
		return nil, err
	}
	return s.Result(), nil
}
