# SPDX-License-Identifier: Apache-2.0

"""Utilities for torch_memory_saver (TMS) configuration and setup.

This module handles the environment variable setup required for TMS to work
properly with LD_PRELOAD hooks.
"""

import os
from contextlib import nullcontext

try:
    from torch_memory_saver import torch_memory_saver
except ImportError:

    class MockTorchMemorySaver:
        def disable(self):
            return nullcontext()

        def pause(self):
            pass

        def resume(self):
            pass

    torch_memory_saver = MockTorchMemorySaver()


def get_tms_env_vars() -> dict[str, str]:
    """Get environment variables for torch_memory_saver (TMS).

    The returned ``LD_PRELOAD`` is *exactly* the TMS shim path with no
    extra entries. Callers that build worker command lines must take care
    not to concatenate this with another ``LD_PRELOAD`` value (e.g. the
    one ``stdbuf`` injects), otherwise ``ld.so`` may treat the entire
    ``:``-separated string as a single shared-object filename and fail
    with ``cannot open shared object file``. See
    ``areal/infra/utils/proc.py::build_streaming_log_cmd`` for the
    matching guard against ``stdbuf`` wrapping when ``LD_PRELOAD`` is
    already supplied.
    """
    import torch_memory_saver as tms_pkg

    # Locate the LD_PRELOAD shared library
    dynlib_path = os.path.join(
        os.path.dirname(os.path.dirname(tms_pkg.__file__)),
        "torch_memory_saver_hook_mode_preload.abi3.so",
    )

    if not os.path.exists(dynlib_path):
        raise RuntimeError(f"LD_PRELOAD so file {dynlib_path} does not exist.")

    env_vars = {
        "LD_PRELOAD": dynlib_path,
        "TMS_INIT_ENABLE": "1",
        "TMS_INIT_ENABLE_CPU_BACKUP": "1",
    }
    return env_vars


def scrub_ld_preload_for_tms() -> None:
    """Reduce ``os.environ['LD_PRELOAD']`` to just the TMS shim, if present.

    Some launchers (notably ``stdbuf -oL``) concatenate their own hooks
    onto ``LD_PRELOAD`` using ``:`` as the separator before exec. By the
    time Python is up, the dynamic linker has already loaded everything
    it intended to load, so the env var is no longer needed for *that*
    purpose; but ``torch_memory_saver``'s own helpers may still read
    ``os.environ['LD_PRELOAD']`` to locate its preload shim, and on
    systems whose ``ld.so`` does not split on ``:`` this fails with
    ``cannot open shared object file``.

    Call this very early in the worker entry point — before TMS is first
    imported — to keep only the TMS shim in ``LD_PRELOAD`` (or unset the
    variable entirely if TMS is not configured for this worker). Safe to
    call unconditionally: it is a no-op when TMS is not enabled.
    """
    raw = os.environ.get("LD_PRELOAD", "")
    if not raw:
        return
    # ``LD_PRELOAD`` is officially space-separated, but stdbuf and a few
    # other tools use ``:`` and many ld.so's accept both. Split on either
    # so we recover the individual entries.
    entries = [p for p in raw.replace(":", " ").split() if p]
    tms_entries = [
        p for p in entries if p.endswith("torch_memory_saver_hook_mode_preload.abi3.so")
    ]
    if tms_entries:
        # Keep exactly one TMS entry; drop everything else (stdbuf hook etc).
        os.environ["LD_PRELOAD"] = tms_entries[0]
    else:
        # Nothing TMS-related survived; leave the variable alone so we do
        # not silently break unrelated preload setups in non-TMS workers.
        return


def is_tms_enabled() -> bool:
    return os.environ.get("TMS_INIT_ENABLE", "0") == "1"
