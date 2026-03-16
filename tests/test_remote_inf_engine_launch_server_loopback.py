import pytest

pytest.importorskip("aiohttp")

from areal.api.cli_args import InferenceEngineConfig
from areal.infra.remote_inf_engine import RemoteInfEngine
from areal.utils.network import get_loopback_ip


class DummyBackend:
    def launch_server(self, server_args):
        class DummyProc:
            returncode = None

            def poll(self):
                return None

        return DummyProc()

    def get_health_check_request(self):
        class Req:
            endpoint = "/health"
            method = "GET"
            payload = None

        return Req()


def test_launch_server_defaults_to_loopback_host(monkeypatch):
    cfg = InferenceEngineConfig(setup_timeout=0.01)
    engine = RemoteInfEngine(cfg, DummyBackend())

    monkeypatch.setattr(engine, "_wait_for_server", lambda _addr, process=None: None)

    info = engine.launch_server(server_args={})
    assert info.host == get_loopback_ip()
