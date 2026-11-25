"""Model calls reach public internet addresses only, checked at connect time.

The resolver vets every address a hostname resolves to and aiohttp connects to
the vetted address, so DNS rebinding has no window. IP literals skip resolution
in aiohttp, so `check_url` refuses private ones before any connection.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlsplit

from aiohttp.abc import AbstractResolver, ResolveResult
from aiohttp.resolver import DefaultResolver

from decisioneval.spec.transport import TransportError

BLOCKED_HOSTS = frozenset({"localhost", "metadata", "metadata.google.internal"})


class EgressRefused(TransportError):
    """The URL or an address it resolves to is not on the public internet."""


# IPv6 forms that carry an IPv4 address a gateway may route to; a model host has no need of them.
TRANSITION_NETWORKS = tuple(
    ipaddress.ip_network(network)
    for network in ("::/96", "::ffff:0:0:0/96", "64:ff9b::/96", "64:ff9b:1::/48", "2002::/16", "2001::/32")
)


def is_public_address(address: str) -> bool:
    ip = ipaddress.ip_address(address.split("%", 1)[0])
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            return ip.ipv4_mapped.is_global
        if any(ip in network for network in TRANSITION_NETWORKS):
            return False
    return ip.is_global


def _blocked_host(host: str) -> bool:
    return host.rstrip(".").lower() in BLOCKED_HOSTS


def check_url(url: str) -> None:
    """Refuse a URL a model call must never reach. Raises `EgressRefused`."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise EgressRefused(f"model url must be http(s), not {parts.scheme or 'missing'}")
    if parts.username is not None or parts.password is not None:
        raise EgressRefused("model url must not carry credentials")
    host = parts.hostname
    if not host:
        raise EgressRefused("model url has no host")
    if _blocked_host(host):
        raise EgressRefused(f"model url host {host} is not public")
    try:
        public = is_public_address(host)
    except ValueError:
        return  # a hostname: the resolver vets what it resolves to
    if not public:
        raise EgressRefused(f"model url address {host} is not public")


class GuardedResolver(AbstractResolver):
    """Resolves through `inner` and refuses the host if any address is not public."""

    def __init__(self, inner: AbstractResolver | None = None):
        self._inner = inner or DefaultResolver()

    async def resolve(
        self, host: str, port: int = 0, family: socket.AddressFamily = socket.AF_INET
    ) -> list[ResolveResult]:
        if _blocked_host(host):
            raise EgressRefused(f"model host {host} is not public")
        resolved = await self._inner.resolve(host, port, family)
        for entry in resolved:
            if not is_public_address(entry["host"]):
                raise EgressRefused(f"model host {host} resolves to {entry['host']}, which is not public")
        return resolved

    async def close(self) -> None:
        await self._inner.close()
