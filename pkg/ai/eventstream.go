package ai

import "sync"

// EventStream is a push-based, channel-backed async event stream.
// Consumers range over Events(); producers call Push/End.
// R is the final result type extracted from the terminal event.
type EventStream[T any, R any] struct {
	ch            chan T
	once          sync.Once
	resultCh      chan R
	isComplete    func(T) bool
	extractResult func(T) R
}

// NewEventStream creates an event stream.
// isComplete returns true for terminal events (done/error).
// extractResult pulls the final value from a terminal event.
func NewEventStream[T any, R any](
	isComplete func(T) bool,
	extractResult func(T) R,
) *EventStream[T, R] {
	return &EventStream[T, R]{
		ch:            make(chan T, 64),
		resultCh:      make(chan R, 1),
		isComplete:    isComplete,
		extractResult: extractResult,
	}
}

// Push sends an event to consumers. If the event is terminal the result is
// resolved and the channel is closed.
func (s *EventStream[T, R]) Push(event T) {
	if s.isComplete(event) {
		s.resultCh <- s.extractResult(event)
	}
	s.ch <- event
	if s.isComplete(event) {
		s.once.Do(func() { close(s.ch) })
	}
}

// End closes the stream with an explicit result (used when no terminal event).
func (s *EventStream[T, R]) End(result R) {
	select {
	case s.resultCh <- result:
	default:
	}
	s.once.Do(func() { close(s.ch) })
}

// Events returns a channel that yields events until the stream ends.
func (s *EventStream[T, R]) Events() <-chan T {
	return s.ch
}

// Result blocks until the final result is available.
func (s *EventStream[T, R]) Result() R {
	return <-s.resultCh
}

// ---------------------------------------------------------------------------
// AssistantMessageEventStream — the concrete type used by providers
// ---------------------------------------------------------------------------

// AssistantMessageEventStream is an EventStream specialised for
// AssistantMessageEvent → AssistantMessage.
type AssistantMessageEventStream = EventStream[AssistantMessageEvent, *AssistantMessage]

// NewAssistantMessageEventStream creates a stream for assistant message events.
func NewAssistantMessageEventStream() *AssistantMessageEventStream {
	return NewEventStream[AssistantMessageEvent, *AssistantMessage](
		func(e AssistantMessageEvent) bool {
			return e.Type == EventDone || e.Type == EventError
		},
		func(e AssistantMessageEvent) *AssistantMessage {
			if e.Type == EventDone {
				return e.Message
			}
			if e.Type == EventError {
				return e.Error
			}
			return nil
		},
	)
}
