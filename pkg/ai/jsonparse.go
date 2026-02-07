package ai

import (
	"encoding/json"
	"strings"
)

// ParseStreamingJSON attempts to parse potentially incomplete JSON.
// It tries standard parsing first, then falls back to best-effort
// recovery for incomplete JSON (e.g. missing closing braces).
// Returns an empty map on failure.
func ParseStreamingJSON(partial string) map[string]any {
	partial = strings.TrimSpace(partial)
	if partial == "" {
		return map[string]any{}
	}

	// Fast path: try complete JSON.
	var result map[string]any
	if err := json.Unmarshal([]byte(partial), &result); err == nil {
		return result
	}

	// Best-effort: close open braces/brackets.
	result = tryRepairAndParse(partial)
	if result != nil {
		return result
	}

	return map[string]any{}
}

// tryRepairAndParse attempts to repair incomplete JSON by appending
// missing closing braces/brackets and removing trailing partial tokens.
func tryRepairAndParse(s string) map[string]any {
	// Remove trailing comma if present.
	trimmed := strings.TrimRight(s, " \t\n\r")
	trimmed = strings.TrimSuffix(trimmed, ",")

	// Count open braces/brackets.
	openBraces := 0
	openBrackets := 0
	inString := false
	escaped := false

	for _, c := range trimmed {
		if escaped {
			escaped = false
			continue
		}
		if c == '\\' && inString {
			escaped = true
			continue
		}
		if c == '"' {
			inString = !inString
			continue
		}
		if inString {
			continue
		}
		switch c {
		case '{':
			openBraces++
		case '}':
			openBraces--
		case '[':
			openBrackets++
		case ']':
			openBrackets--
		}
	}

	// If we're inside a string, try to close it.
	if inString {
		trimmed += `"`
	}

	// Close any open brackets then braces.
	for openBrackets > 0 {
		trimmed += "]"
		openBrackets--
	}
	for openBraces > 0 {
		trimmed += "}"
		openBraces--
	}

	var result map[string]any
	if err := json.Unmarshal([]byte(trimmed), &result); err == nil {
		return result
	}
	return nil
}
