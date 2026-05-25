# models

Cloud inference infrastructure for Kiki OS — open models deployed on [Modal.com](https://modal.com) with an OpenAI-compatible API.

Used when the local device cannot run inference (insufficient RAM, battery constraints, or explicit user opt-in for larger models).

---

## Design

Kiki is **local-first by default** (`allow_remote = false` in `agentd.toml`). When the user or agent opts into remote inference, requests are routed through the Kiki AI Gateway (Cloudflare AI Gateway + this service) rather than directly to third-party providers.

```
agentd (on device)
    │
    │  (remote inference requested)
    ▼
Kiki AI Gateway (Cloudflare)
    │
    ▼
kiki-models (Modal.com)
    │
    ├── vLLM endpoint  (large models)
    └── adapter        (model-specific preprocessing)
```

All endpoints expose an **OpenAI-compatible API** — `agentd`'s provider abstraction works identically for local and remote inference.

---

## Structure

```
modal/
  serve.py             → vLLM OpenAI-compatible server (one Modal app per model)
tests/
  test_serve.py        → registry contract tests (stdlib only, no Modal/GPU)
pyproject.toml         → Python project definition
```

`serve.py` runs vLLM's own OpenAI server as a subprocess behind Modal's
`web_server`, so `/v1/chat/completions` (streaming + structured tool calls),
`/v1/completions` and `/v1/models` come straight from vLLM — byte-identical to
the local llama.cpp / ollama surface `kiki-provider`'s OpenAI backend already
speaks.

---

## Supported models

Defined in the `MODELS` registry in `modal/serve.py` — each pins an HF repo, a
GPU, and vLLM's matching tool-call parser (so structured tool calls round-trip):

| Key | Model | Tool parser |
|---|---|---|
| `llama-3.1-8b` (default) | meta-llama/Llama-3.1-8B-Instruct | `llama3_json` |
| `qwen2.5-7b` | Qwen/Qwen2.5-7B-Instruct | `hermes` |
| `granite-3.1-8b` | ibm-granite/granite-3.1-8b-instruct | `granite` |

Add a model by adding a `ModelSpec` to the registry. Smaller models (≤ ~3B) run
on-device; this service is for larger ones or explicit user opt-in.

---

## Deploy

Requires a Modal account. Create the `kiki-models-secrets` Modal secret with:

- `KIKI_MODELS_API_KEY` — the bearer the gateway / agentd authenticates with
- `HF_TOKEN` — to pull gated repos (e.g. Llama)

```sh
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Deploy one model (one Modal app per model key):
MODEL_KEY=llama-3.1-8b modal deploy modal/serve.py
MODEL_KEY=qwen2.5-7b    modal deploy modal/serve.py

# Contract tests (no Modal/GPU needed):
python3 tests/test_serve.py     # or: pytest
```

Point a device at it by setting its OpenAI-compatible provider `base_url` to
`https://<workspace>--kiki-models-<model>-serve.modal.run/v1` and `api_key` to
`KIKI_MODELS_API_KEY` (typically injected by the AI Gateway, not the device).

---

## Related repos

| Repo | Description |
|---|---|
| [agent](https://github.com/Kiki-OS/agent) | `agentd` — `kiki-provider` routes inference here |
| [cloud](https://github.com/Kiki-OS/cloud) | AI Gateway — sits between device and this service |

---

## License

MIT License. See [LICENSE](LICENSE).
