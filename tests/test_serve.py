"""Contract tests for the vLLM deployment registry.

These run with **stdlib only** — `modal` is stubbed, so no Modal account, GPU, or
model weights are needed. They validate the part we can check locally: that the
model registry is well-formed and the deploy-time model selection fails closed on
an unknown key. (Serving itself can only be validated against a real deploy.)

Run: python3 tests/test_serve.py   (or: pytest)
"""

import importlib.util
import os
import pathlib
import sys
from unittest import mock

SERVE_PATH = pathlib.Path(__file__).resolve().parent.parent / "modal" / "serve.py"


def _load(model_key: str | None = None):
    """Import modal/serve.py with `modal` stubbed and MODEL_KEY set."""
    if model_key is None:
        os.environ.pop("MODEL_KEY", None)
    else:
        os.environ["MODEL_KEY"] = model_key
    sys.modules["modal"] = mock.MagicMock()
    spec = importlib.util.spec_from_file_location("kiki_serve", SERVE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_registry_is_well_formed():
    m = _load()
    assert m.MODELS, "registry must not be empty"
    for key, s in m.MODELS.items():
        assert s.hf_repo and "/" in s.hf_repo, f"{key}: hf_repo must be 'org/name'"
        assert s.gpu, f"{key}: a GPU must be specified"
        assert s.tool_parser, f"{key}: a tool-call parser is required for agent tool use"
        assert s.max_model_len >= 4096, f"{key}: context too small for the agent harness"


def test_default_model_is_valid():
    m = _load()
    assert m.MODEL_KEY in m.MODELS


def test_unknown_model_key_fails_closed():
    raised = False
    try:
        _load("does-not-exist")
    except ValueError as e:
        raised = True
        assert "unknown MODEL_KEY" in str(e)
    assert raised, "an unknown MODEL_KEY must raise at deploy time"


def test_each_model_selectable():
    for key in _load().MODELS:
        assert _load(key).MODEL_KEY == key


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"ok   {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
