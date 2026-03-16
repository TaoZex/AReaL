import pytest

pytest.importorskip("aiohttp")

from areal.engine.sglang_remote import SGLangBackend
from areal.engine.vllm_remote import VLLMBackend
from areal.utils.network import format_host_for_url


class DummyAllocModeGen:
    pp_size = 1
    tp_size = 1
    world_size = 2


class DummyAllocMode:
    gen = DummyAllocModeGen()


class DummyMeta:
    type = "xccl"
    alloc_mode = DummyAllocMode()
    nccl_master_address = "fdbd:dc05:13::28"
    nccl_master_port = 10367
    nccl_group_name = "update_weight_group"
    use_lora = False


def test_sglang_init_weights_group_payload_types_and_ipv6():
    req = SGLangBackend().build_init_weights_group_request(
        addr="127.0.0.1:1", server_idx=0, meta=DummyMeta()
    )
    assert req.payload["master_address"] == format_host_for_url("fdbd:dc05:13::28")
    assert req.payload["master_port"] == 10367


def test_vllm_init_weights_group_payload_types_and_ipv6():
    req = VLLMBackend().build_init_weights_group_request(
        addr="127.0.0.1:1", server_idx=0, meta=DummyMeta()
    )
    assert req.payload["master_address"] == format_host_for_url("fdbd:dc05:13::28")
    assert req.payload["master_port"] == 10367
