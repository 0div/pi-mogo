package ai

import "os"

// providerEnvKeys maps provider names to environment variable names.
var providerEnvKeys = map[Provider][]string{
	ProviderOpenAI:            {"OPENAI_API_KEY"},
	ProviderAnthropic:         {"ANTHROPIC_API_KEY"},
	ProviderGoogle:            {"GOOGLE_API_KEY", "GEMINI_API_KEY"},
	ProviderGoogleVertex:      {"GOOGLE_API_KEY"},
	ProviderXAI:               {"XAI_API_KEY"},
	ProviderGroq:              {"GROQ_API_KEY"},
	ProviderCerebras:          {"CEREBRAS_API_KEY"},
	ProviderOpenRouter:        {"OPENROUTER_API_KEY"},
	ProviderMistral:           {"MISTRAL_API_KEY"},
	ProviderMinimax:           {"MINIMAX_API_KEY"},
	ProviderMinimaxCN:         {"MINIMAX_API_KEY"},
	ProviderHuggingface:       {"HUGGINGFACE_API_KEY", "HF_TOKEN"},
	ProviderAmazonBedrock:     {"AWS_BEARER_TOKEN_BEDROCK"},
	ProviderVercelAIGateway:   {"VERCEL_API_KEY"},
	ProviderZAI:               {"ZAI_API_KEY"},
	ProviderKimiCoding:        {"KIMI_API_KEY"},
}

// GetEnvApiKey returns the API key for a provider from environment variables.
// Returns empty string if no key is found.
func GetEnvApiKey(provider Provider) string {
	keys, ok := providerEnvKeys[provider]
	if !ok {
		return ""
	}
	for _, k := range keys {
		if v := os.Getenv(k); v != "" {
			return v
		}
	}
	return ""
}
