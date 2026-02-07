package ai

import "sync"

// StreamFunction is the signature for a provider's streaming function.
type StreamFunction func(model *Model, ctx Context, opts *StreamOptions) *AssistantMessageEventStream

// StreamSimpleFunction is the higher-level variant that accepts reasoning options.
type StreamSimpleFunction func(model *Model, ctx Context, opts *SimpleStreamOptions) *AssistantMessageEventStream

// ApiProvider bundles a provider's stream functions for a specific API.
type ApiProvider struct {
	Api          Api
	Stream       StreamFunction
	StreamSimple StreamSimpleFunction
}

type registeredProvider struct {
	provider *ApiProvider
	sourceID string
}

var (
	providerRegistry   = map[Api]*registeredProvider{}
	providerRegistryMu sync.RWMutex
)

// RegisterApiProvider registers a provider for the given API.
// An optional sourceID can be supplied so that a batch of providers can be
// unregistered together via UnregisterApiProviders.
func RegisterApiProvider(p *ApiProvider, sourceID string) {
	providerRegistryMu.Lock()
	defer providerRegistryMu.Unlock()
	providerRegistry[p.Api] = &registeredProvider{provider: p, sourceID: sourceID}
}

// GetApiProvider returns the registered provider for an API, or nil.
func GetApiProvider(api Api) *ApiProvider {
	providerRegistryMu.RLock()
	defer providerRegistryMu.RUnlock()
	if r := providerRegistry[api]; r != nil {
		return r.provider
	}
	return nil
}

// GetApiProviders returns all registered providers.
func GetApiProviders() []*ApiProvider {
	providerRegistryMu.RLock()
	defer providerRegistryMu.RUnlock()
	out := make([]*ApiProvider, 0, len(providerRegistry))
	for _, r := range providerRegistry {
		out = append(out, r.provider)
	}
	return out
}

// UnregisterApiProviders removes all providers with the given sourceID.
func UnregisterApiProviders(sourceID string) {
	providerRegistryMu.Lock()
	defer providerRegistryMu.Unlock()
	for api, r := range providerRegistry {
		if r.sourceID == sourceID {
			delete(providerRegistry, api)
		}
	}
}

// ClearApiProviders removes all registered providers.
func ClearApiProviders() {
	providerRegistryMu.Lock()
	defer providerRegistryMu.Unlock()
	providerRegistry = map[Api]*registeredProvider{}
}
