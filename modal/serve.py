"""
Modal.com inference deployment for Kiki OS.

Serves open models behind an **OpenAI-compatible API** (vLLM's built-in
`/v1/chat/completions`, `/v1/completions`, `/v1/models`) so `kiki-agent`'s
OpenAI provider (`kiki-provider/src/openai.rs`) routes to cloud inference
unchanged when on-device compute is insufficient.

Rather than re-implement the OpenAI routes, we run vLLM's own OpenAI server as a
subprocess behind Modal's web_server: it gives streaming SSE and structured
tool-calls (the agent harness needs both) for free, exactly matching the local
llama.cpp / ollama surface.

Deploy one model:
    MODEL_KEY=llama-3.1-8b modal deploy modal/serve.py

The served base URL is `https://<workspace>--kiki-models-<model>-serve.modal.run/v1`;
point agentd at it with the bearer key in the `kiki-models-secrets` Modal secret.
"""

import os
import subprocess

import modal

# ── Model registry ────────────────────────────────────────────────────────────
# Each entry pins the HF repo, the GPU it needs, and vLLM's tool-call parser for
# that model family (so structured tool calls round-trip). Keep tool_parser in
# sync with the model's chat template — a wrong parser silently breaks tool use.


class ModelSpec:
    def __init__(
        self,
        hf_repo: str,
        gpu: str,
        tool_parser: str,
        max_model_len: int = 8192,
        extra_args: list[str] | None = None,
    ):
        self.hf_repo = hf_repo
        self.gpu = gpu
        self.tool_parser = tool_parser
        self.max_model_len = max_model_len
        self.extra_args = extra_args or []


MODELS: dict[str, ModelSpec] = {
    "llama-3.1-8b": ModelSpec(
        hf_repo="meta-llama/Llama-3.1-8B-Instruct",
        gpu="A10G",
        tool_parser="llama3_json",
        max_model_len=16384,
    ),
    "qwen2.5-7b": ModelSpec(
        hf_repo="Qwen/Qwen2.5-7B-Instruct",
        gpu="A10G",
        tool_parser="hermes",
        max_model_len=16384,
    ),
    "granite-3.1-8b": ModelSpec(
        hf_repo="ibm-granite/granite-3.1-8b-instruct",
        gpu="A10G",
        tool_parser="granite",
        max_model_len=16384,
    ),
}

# Which model this deployment serves. Set at deploy time:
#   MODEL_KEY=qwen2.5-7b modal deploy modal/serve.py
MODEL_KEY = os.environ.get("MODEL_KEY", "llama-3.1-8b")
if MODEL_KEY not in MODELS:
    raise ValueError(f"unknown MODEL_KEY {MODEL_KEY!r}; one of {sorted(MODELS)}")
SPEC = MODELS[MODEL_KEY]

VLLM_PORT = 8000

app = modal.App(f"kiki-models-{MODEL_KEY}")

# vLLM image. flashinfer speeds up attention; harmless if unused.
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "vllm==0.6.6",
    "huggingface_hub[hf_transfer]==0.27.0",
)

# Cache weights across cold starts so only the first boot downloads them.
hf_cache = modal.Volume.from_name("kiki-models-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("kiki-models-vllm-cache", create_if_missing=True)

# `kiki-models-secrets` must carry:
#   - HF_TOKEN           (to pull gated repos like Llama)
#   - KIKI_MODELS_API_KEY (the bearer the gateway/agentd authenticates with)
secrets = modal.Secret.from_name("kiki-models-secrets")


@app.function(
    image=image,
    gpu=SPEC.gpu,
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
    secrets=[secrets],
    # Scale to zero when idle; keep a warm container briefly between requests.
    scaledown_window=300,
    timeout=30 * 60,
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=VLLM_PORT, startup_timeout=20 * 60)
def serve():
    """Boot vLLM's OpenAI-compatible server as a subprocess.

    vLLM enforces the bearer itself via `--api-key`, so every route is
    authenticated without us proxying. `--enable-auto-tool-choice` +
    `--tool-call-parser` make `/v1/chat/completions` emit structured tool calls
    the agent harness can parse.
    """
    api_key = os.environ["KIKI_MODELS_API_KEY"]
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    cmd = [
        "vllm",
        "serve",
        SPEC.hf_repo,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--api-key",
        api_key,
        "--served-model-name",
        MODEL_KEY,
        "--max-model-len",
        str(SPEC.max_model_len),
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        SPEC.tool_parser,
        *SPEC.extra_args,
    ]
    subprocess.Popen(" ".join(cmd), shell=True)
