package ai

import "regexp"

// overflowPatterns detects context-overflow errors from various providers.
var overflowPatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)prompt is too long`),
	regexp.MustCompile(`(?i)input is too long for requested model`),
	regexp.MustCompile(`(?i)exceeds the context window`),
	regexp.MustCompile(`(?i)input token count.*exceeds the maximum`),
	regexp.MustCompile(`(?i)maximum prompt length is \d+`),
	regexp.MustCompile(`(?i)reduce the length of the messages`),
	regexp.MustCompile(`(?i)maximum context length is \d+ tokens`),
	regexp.MustCompile(`(?i)exceeds the limit of \d+`),
	regexp.MustCompile(`(?i)exceeds the available context size`),
	regexp.MustCompile(`(?i)greater than the context length`),
	regexp.MustCompile(`(?i)context window exceeds limit`),
	regexp.MustCompile(`(?i)exceeded model token limit`),
	regexp.MustCompile(`(?i)context[_ ]length[_ ]exceeded`),
	regexp.MustCompile(`(?i)too many tokens`),
	regexp.MustCompile(`(?i)token limit exceeded`),
}

// noBodyPattern matches Cerebras/Mistral-style 400/413 status codes with no body.
var noBodyPattern = regexp.MustCompile(`(?i)^4(00|13)\s*(status code)?\s*\(no body\)`)

// IsContextOverflow returns true when an assistant message indicates the
// input exceeded the model's context window.
//
// contextWindow is optional; if > 0 it enables silent-overflow detection
// (e.g. z.ai accepts overflow requests but returns inflated usage).
func IsContextOverflow(msg *AssistantMessage, contextWindow int) bool {
	if msg.StopReason == StopReasonError && msg.ErrorMessage != "" {
		for _, p := range overflowPatterns {
			if p.MatchString(msg.ErrorMessage) {
				return true
			}
		}
		if noBodyPattern.MatchString(msg.ErrorMessage) {
			return true
		}
	}

	// Silent overflow detection.
	if contextWindow > 0 && msg.StopReason == StopReasonStop {
		inputTokens := msg.Usage.Input + msg.Usage.CacheRead
		if inputTokens > contextWindow {
			return true
		}
	}

	return false
}

// GetOverflowPatterns returns the compiled patterns (for testing).
func GetOverflowPatterns() []*regexp.Regexp {
	out := make([]*regexp.Regexp, len(overflowPatterns))
	copy(out, overflowPatterns)
	return out
}
