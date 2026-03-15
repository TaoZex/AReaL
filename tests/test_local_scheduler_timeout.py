import pytest

aiohttp = pytest.importorskip("aiohttp")

from areal.api.scheduler_api import Worker
from areal.infra.scheduler.exceptions import WorkerTimeoutError
from areal.infra.scheduler.local import LocalScheduler, WorkerInfo


class DummyProcess:
    def __init__(self):
        self.returncode = None

    def poll(self):
        return None


def test_configure_worker_timeout(monkeypatch, tmp_path):
    (tmp_path / "nr").mkdir(parents=True, exist_ok=True)
    scheduler = LocalScheduler(
        experiment_name="exp",
        trial_name="trial",
        fileroot=str(tmp_path),
        nfs_record_root=str(tmp_path / "nr"),
        startup_timeout=0.1,
        health_check_interval=0.1,
    )

    worker = Worker(
        id="actor/0",
        ip="::1",
        worker_ports=["12345"],
        engine_ports=[],
    )
    worker_info = WorkerInfo(
        worker=worker,
        process=DummyProcess(),
        role="actor",
        gpu_devices=[],
        created_at=0.0,
        log_file=str(tmp_path / "actor.log"),
        env_vars={},
    )

    monkeypatch.setattr(scheduler, "_is_worker_ready", lambda _info: False)

    import areal.infra.scheduler.local as local_module

    t = [0.0]

    def fake_time():
        t[0] += 0.2
        return t[0]

    monkeypatch.setattr(local_module.time, "time", fake_time)
    monkeypatch.setattr(local_module.time, "sleep", lambda _seconds: None)

    with pytest.raises(WorkerTimeoutError):
        scheduler._configure_worker(worker_info, 0)
