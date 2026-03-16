import pytest

pytest.importorskip("aiohttp")
pytest.importorskip("ray")
pytest.importorskip("torch")

from areal.api.cli_args import InferenceEngineConfig
from areal.infra.remote_inf_engine import RemoteInfEngine


class DummyBackend:
    def get_health_check_request(self):
        class Req:
            endpoint = "/health"
            method = "GET"
            payload = None

        return Req()


def test_wait_for_server_formats_ipv6_base_url(monkeypatch):
    cfg = InferenceEngineConfig(setup_timeout=1.0)
    engine = RemoteInfEngine(cfg, DummyBackend())

    seen = {}

    def fake_check_health(base_url):
        seen["base_url"] = base_url
        return True

    monkeypatch.setattr(engine, "check_health", fake_check_health)
    engine._wait_for_server("2001:db8::1:8000")
    assert seen["base_url"] == "http://[2001:db8::1]:8000"
