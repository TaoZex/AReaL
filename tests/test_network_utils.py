import pytest

aiohttp = pytest.importorskip("aiohttp")

from areal.utils.network import format_hostport, split_hostport


def test_format_hostport_ipv6():
    assert format_hostport("2001:db8::1", 8000) == "[2001:db8::1]:8000"
    assert format_hostport("[2001:db8::1]", 8000) == "[2001:db8::1]:8000"


def test_split_hostport_ipv6():
    host, port = split_hostport("[2001:db8::1]:8000")
    assert host == "2001:db8::1"
    assert port == 8000

    host, port = split_hostport("2001:db8::1:8000")
    assert host == "2001:db8::1"
    assert port == 8000
