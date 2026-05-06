# LLM Providers

assistant replay uses the `LLMProvider` port. Providers are plugins; the
assistant replay application service does not know whether a model is local or cloud-hosted.

## Built-In Providers

- `llama-cpp`: recommended thesis-reproducibility path. It targets a running
  llama.cpp OpenAI-compatible chat server, usually
  `http://127.0.0.1:8080/v1`. The default reproducible profile is
  Qwen2.5-7B-Instruct GGUF Q4_K_M served by llama.cpp. Structured assistant
  replay uses OpenAI-compatible JSON-object mode because current llama.cpp
  servers accept `response_format = {"type": "json_object"}`.
- `openai-compatible`: generic OpenAI-compatible endpoint for local or cloud
  servers.
- `openai`: OpenAI-compatible hosted endpoint using `OPENAI_API_KEY`.
- `anthropic`: Anthropic Messages API using `ANTHROPIC_API_KEY`.
- `google`: Gemini OpenAI-compatible endpoint using `GOOGLE_API_KEY`.
- `xai`: xAI OpenAI-compatible endpoint using `XAI_API_KEY`.
- `fake`: deterministic CI/smoke provider. Do not use it for thesis-full runs
  unless explicitly running a smoke profile.

Provider config stores only environment variable names, never secrets:

```toml
[assistant.provider]
name = "llama-cpp"
model = "Qwen2.5-7B-Instruct-GGUF-Q4_K_M"
base_url = "http://127.0.0.1:8080/v1"
timeout_s = 180.0
temperature = 0.0
top_p = 1.0
max_tokens = 700
seed = 1337
```

Cloud example:

```toml
[assistant.provider]
name = "openai"
model = "gpt-4.1-mini"
api_key_env = "OPENAI_API_KEY"
temperature = 0.0
max_tokens = 700
```

Use:

```powershell
itse assistant providers
itse assistant preflight --config config/reproduction.toml
```

The historical vLLM Qwen profile remains useful for comparing older thesis
artifacts, but the default local reproduction path is llama.cpp on port 8080.
