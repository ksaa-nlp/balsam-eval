"""The egress guard: a model call may only reach a public internet address."""

import socket

import pytest
from aiohttp import web
from aiohttp.abc import AbstractResolver

from decisioneval.egress import EgressRefused, GuardedResolver, check_url, is_public_address
from decisioneval.transport import ModelClient, RetryPolicy

NOT_PUBLIC = [
    "10.0.0.5",
    "172.16.0.1",
    "192.168.1.1",
    "127.0.0.1",
    "169.254.169.254",
    "100.64.0.1",
    "0.0.0.0",
    "::1",
    "::",
    "fd00::1",
    "fe80::1",
    "::ffff:10.0.0.1",
    "::ffff:127.0.0.1",
    "64:ff9b::a9fe:a9fe",  # NAT64 of the metadata address
    "64:ff9b:1::a00:5",
    "::169.254.169.254",  # IPv4-compatible
    "2002:a9fe:a9fe::1",  # 6to4
    "::ffff:0:169.254.169.254",  # SIIT IPv4-translated
    "2001:0:4136:e378:8000:63bf:3fff:fdd2",  # Teredo
]


@pytest.mark.parametrize("address", NOT_PUBLIC)
def test_non_global_addresses_are_not_public(address):
    assert not is_public_address(address)


@pytest.mark.parametrize("address", ["8.8.8.8", "2001:4860:4860::8888", "::ffff:8.8.8.8"])
def test_global_addresses_are_public(address):
    assert is_public_address(address)


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com/x",
        "file:///etc/passwd",
        "https://user:pass@example.com/x",
        "http://10.0.0.5:8080/v1",
        "http://[::ffff:10.0.0.1]/v1",
        "http://[64:ff9b::a9fe:a9fe]/computeMetadata/v1/",
        "http://[::169.254.169.254]/",
        "http://169.254.169.254/computeMetadata/v1/",
        "http://metadata.google.internal/computeMetadata/v1/",
        "http://METADATA.GOOGLE.INTERNAL./x",
        "http://localhost:8080/x",
        "https:///nohost",
    ],
)
def test_unsafe_urls_are_refused_before_any_connection(url):
    with pytest.raises(EgressRefused):
        check_url(url)


def test_a_public_hostname_url_passes_the_precheck():
    check_url("https://api.openai.com/v1/decisions")


class StubResolver(AbstractResolver):
    def __init__(self, *addresses):
        self.addresses = addresses

    async def resolve(self, host, port=0, family=socket.AF_INET):
        return [
            {"hostname": host, "host": address, "port": port, "family": family, "proto": 0, "flags": 0}
            for address in self.addresses
        ]

    async def close(self):
        return None


async def test_a_host_resolving_to_a_private_address_is_refused():
    with pytest.raises(EgressRefused, match="10.1.2.3"):
        await GuardedResolver(StubResolver("10.1.2.3")).resolve("model.example.com", 443)


async def test_one_private_address_among_public_ones_refuses_the_host():
    with pytest.raises(EgressRefused):
        await GuardedResolver(StubResolver("8.8.8.8", "169.254.169.254")).resolve("model.example.com", 443)


async def test_a_host_resolving_only_to_public_addresses_is_kept():
    resolved = await GuardedResolver(StubResolver("8.8.8.8")).resolve("model.example.com", 443)
    assert [entry["host"] for entry in resolved] == ["8.8.8.8"]


async def test_the_metadata_hostname_is_refused_without_resolving():
    with pytest.raises(EgressRefused):
        await GuardedResolver(StubResolver("8.8.8.8")).resolve("metadata.google.internal", 80)


async def test_the_guarded_client_never_connects_to_a_rebound_private_address():
    hits = []

    async def handle(request):
        hits.append(request.path)
        return web.json_response({})

    app = web.Application()
    app.router.add_post("/x", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        # The hostname passes the precheck, then resolves to loopback at connect time.
        client = ModelClient(retry=RetryPolicy(attempts=1), resolver=GuardedResolver(StubResolver("127.0.0.1")))
        async with client as model:
            with pytest.raises(EgressRefused):
                await model.send(f"http://model.example.com:{port}/x", {}, {})
    finally:
        await runner.cleanup()
    assert hits == []


async def test_the_guarded_client_refuses_a_private_literal_before_connecting():
    async with ModelClient(retry=RetryPolicy(attempts=1)) as model:
        with pytest.raises(EgressRefused):
            await model.send("http://10.0.0.5:8080/v1", {}, {})
