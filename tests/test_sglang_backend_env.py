import os
import subprocess

import pytest

pytest.importorskip("aiohttp")

from areal.engine.sglang_remote import SGLangBackend
from areal.api.cli_args import SGLangConfig


def test_sglang_backend_launch_server_disables_proxy_env(monkeypatch):
    backend = SGLangBackend()
    server_args = {"host": "::", "port": 12345, "model_path": "/tmp/does_not_matter"}

    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:8080")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:8080")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)

    captured = {}

    monkeypatch.setattr(SGLangConfig, "build_cmd_from_args", lambda _args: ["true"])

    class DummyProc:
        def poll(self):
            return None

    def fake_popen(cmd, env, stdout, stderr):
        captured["env"] = env
        return DummyProc()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    backend.launch_server(server_args)

    env = captured["env"]
    assert "HTTP_PROXY" not in env
    assert "HTTPS_PROXY" not in env
    assert env.get("NO_PROXY") == "*"
    assert env.get("no_proxy") == "*"
