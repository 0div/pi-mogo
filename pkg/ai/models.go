package ai

import "sync"

var (
	modelRegistry   = map[Provider]map[string]*Model{}
	modelRegistryMu sync.RWMutex
)

// RegisterModel adds a model to the registry.
func RegisterModel(m *Model) {
	modelRegistryMu.Lock()
	defer modelRegistryMu.Unlock()
	if modelRegistry[m.Provider] == nil {
		modelRegistry[m.Provider] = map[string]*Model{}
	}
	modelRegistry[m.Provider][m.ID] = m
}

// GetModel returns a model by provider and id, or nil.
func GetModel(provider Provider, modelID string) *Model {
	modelRegistryMu.RLock()
	defer modelRegistryMu.RUnlock()
	if pm := modelRegistry[provider]; pm != nil {
		return pm[modelID]
	}
	return nil
}

// GetProviders returns all registered provider names.
func GetProviders() []Provider {
	modelRegistryMu.RLock()
	defer modelRegistryMu.RUnlock()
	out := make([]Provider, 0, len(modelRegistry))
	for p := range modelRegistry {
		out = append(out, p)
	}
	return out
}

// GetModels returns all models for a provider.
func GetModels(provider Provider) []*Model {
	modelRegistryMu.RLock()
	defer modelRegistryMu.RUnlock()
	pm := modelRegistry[provider]
	if pm == nil {
		return nil
	}
	out := make([]*Model, 0, len(pm))
	for _, m := range pm {
		out = append(out, m)
	}
	return out
}

// CalculateCost computes costs on a Usage given a Model's pricing.
func CalculateCost(model *Model, usage *Usage) Cost {
	usage.Cost.Input = (model.Cost.Input / 1_000_000) * float64(usage.Input)
	usage.Cost.Output = (model.Cost.Output / 1_000_000) * float64(usage.Output)
	usage.Cost.CacheRead = (model.Cost.CacheRead / 1_000_000) * float64(usage.CacheRead)
	usage.Cost.CacheWrite = (model.Cost.CacheWrite / 1_000_000) * float64(usage.CacheWrite)
	usage.Cost.Total = usage.Cost.Input + usage.Cost.Output + usage.Cost.CacheRead + usage.Cost.CacheWrite
	return usage.Cost
}

// SupportsXHigh returns true if the model supports xhigh thinking level.
func SupportsXHigh(model *Model) bool {
	if contains(model.ID, "gpt-5.2") || contains(model.ID, "gpt-5.3") {
		return true
	}
	if model.Api == ApiAnthropicMessages {
		return contains(model.ID, "opus-4-6") || contains(model.ID, "opus-4.6")
	}
	return false
}

// ModelsAreEqual compares two models by ID and Provider.
func ModelsAreEqual(a, b *Model) bool {
	if a == nil || b == nil {
		return false
	}
	return a.ID == b.ID && a.Provider == b.Provider
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && searchString(s, substr)
}

func searchString(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
