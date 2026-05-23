"""
Modal.com inference deployment for Kiki OS.

Serves open models (Llama, Mistral, etc.) behind an OpenAI-compatible API
so kiki-provider/local.rs can route to cloud inference when on-device
compute is insufficient.

Deploy: modal deploy modal/serve.py
"""

import modal
from fastapi import FastAPI

app = modal.App("kiki-models")

# GPU image with vLLM for fast inference
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm>=0.4", "fastapi", "uvicorn")
)

web_app = FastAPI(title="kiki-models")

@app.function(
    image=image,
    gpu="A10G",
    container_idle_timeout=300,
    secrets=[modal.Secret.from_name("kiki-models-secrets")],
)
@modal.asgi_app()
def inference():
    # TODO: load model via vLLM, expose OpenAI-compat /v1/chat/completions
    return web_app
