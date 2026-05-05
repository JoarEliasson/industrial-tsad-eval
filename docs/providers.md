# LLM Providers

RQ3 assistant replay uses the `LLMProvider` port. Providers are plugins; the
RQ3 application service does not know whether a model is local or cloud-hosted.

## Built-In Providers

- `llama-cpp`: recommended thesis-reproducibility path. It targets a running
  llama.cpp OpenAI-compatible chat server, usually
  `http://127.0.0.1:8080/v1`.
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
[rq3.provider]
name = "llama-cpp"
model = "local-llama"
base_url = "http://127.0.0.1:8080/v1"
temperature = 0.0
top_p = 1.0
max_tokens = 700
seed = 1337
```

Cloud example:

```toml
[rq3.provider]
name = "openai"
model = "gpt-4.1-mini"
api_key_env = "OPENAI_API_KEY"
temperature = 0.0
max_tokens = 700
```

Use:

```powershell
itse rq3 providers
itse rq3 preflight --config config/reproduction.toml
```
