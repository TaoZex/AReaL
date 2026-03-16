import os
import subprocess
import sys

import pytest

pytest.importorskip("aiohttp")

from areal.infra.utils import proc


def test_kill_process_tree_survives_missing_psutil_module_entry():
    p = subprocess.Popen(
        ["bash", "-lc", "sleep 60"],
        preexec_fn=os.setsid,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        sys.modules.pop("psutil", None)
        proc.kill_process_tree(p.pid, timeout=1, graceful=True)
        p.wait(timeout=5)
    finally:
        try:
            p.kill()
        except Exception:
            pass
