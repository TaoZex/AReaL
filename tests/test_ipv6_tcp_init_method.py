import pytest

pytest.importorskip("aiohttp")

from areal.utils.network import format_host_for_url


def test_format_host_for_url_ipv6_tcp_rendezvous():
    host = "fdbd:dc05:13::28"
    init_method = f"tcp://{format_host_for_url(host)}:10367"
    assert init_method == "tcp://[fdbd:dc05:13::28]:10367"
