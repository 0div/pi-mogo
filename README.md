# Pi Mogorepo

A Go implementation of some parts of the [Pi Monorepo](https://github.com/badlogic/pi-mono) 

Tools for building AI agents and managing LLM deployments.

## Packages

| Package      | Description                                                              | Original                                                                                                                                                    |
|--------------|--------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `pkg/ai`     | Unified multi-provider LLM API (OpenAI, Anthropic, Google, etc.)         | [@mariozechner/pi-ai](https://github.com/badlogic/pi-mono/tree/main/packages/ai)                                                                           |
| `pkg/agent`  | Agent runtime with tool calling and state management                     | [@mariozechner/pi-agent-core](https://github.com/badlogic/pi-mono/tree/main/packages/agent)                                                                |

## Usage

See [`examples/simple/main.go`](examples/simple/main.go) for a working example.

## License

MIT
