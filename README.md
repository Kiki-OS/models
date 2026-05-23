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
  deploy.py            → Modal app definitions and endpoint deployment
  inference.py         → vLLM inference endpoint
adapters/
  <model-name>/        → per-model preprocessing and schema adapters
pyproject.toml         → Python project definition
```

---

## Supported model families

Models are selected based on the hardware class declared in the device manifest. Smaller models (≤ 7B) run on-device; larger ones or user-configured alternatives can route here.

---

## Running locally

Requires Python 3.11+ and a Modal account.

```sh
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Deploy to Modal
modal deploy modal/deploy.py

# Run tests
pytest
```

---

## Related repos

| Repo | Description |
|---|---|
| [agent](https://github.com/Kiki-OS/agent) | `agentd` — `kiki-provider` routes inference here |
| [cloud](https://github.com/Kiki-OS/cloud) | AI Gateway — sits between device and this service |

---

## License

MIT License. See [LICENSE](LICENSE).
