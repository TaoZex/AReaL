import pytest

pytest.importorskip("aiohttp")

from areal.utils.network import format_hostport


def test_format_hostport_ipv6_callback_addr():
    host = "fdbd:dc05:13::28"
    addr = format_hostport(host, 21975)
    assert addr == "[fdbd:dc05:13::28]:21975"
    url = f"http://{addr}/callback/init_weights_group"
    assert url == "http://[fdbd:dc05:13::28]:21975/callback/init_weights_group"
