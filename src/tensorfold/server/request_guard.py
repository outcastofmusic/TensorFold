"""Host and Origin checks before authentication, bodies or routing (upstream 1.1.1, request_guard.zig: 9fe5ecff).

A server bound to loopback answers only a loopback Host, so a page that rebinds its own DNS name to 127.0.0.1 is
refused. A browser's Origin must name this server (http, the Host's own name and port) or an origin listed in
TENSORFOLD_ALLOWED_ORIGINS (comma-separated, such as https://ui.example:8443). A request without Origin, which every
non-browser client sends, passes."""

from __future__ import annotations

import ipaddress
import os
import re

HOST_MESSAGE = "Host header is not allowed on this loopback bind; reach the server as localhost or 127.0.0.1"
ORIGIN_MESSAGE = ("Origin header does not match this server; list a trusted origin in TENSORFOLD_ALLOWED_ORIGINS, "
                  "or send the request without one")
_LABEL = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$")


def _authority(raw: str, ports: bool) -> tuple[str, int | None] | None:
    """(host, port) of a Host value or an origin's site part, or None when it is malformed."""

    if not raw or len(raw) > 300 or "%" in raw:
        return None
    port = None
    if raw.startswith("["):
        end = raw.find("]")
        if end < 0:
            return None
        host, rest = raw[1:end], raw[end + 1:]
        try:
            if ipaddress.ip_address(host).version != 6:
                return None
        except ValueError:
            return None
        if rest:
            if not rest.startswith(":"):
                return None
            port = rest[1:]
    else:
        host, _, port_text = raw.partition(":")
        port = port_text if ":" in raw else None
        if not host or len(host) > 253 or not all(_LABEL.match(label) for label in host.split(".")):
            return None
    if port is None:
        return host, None
    if not ports or not port.isdigit() or int(port) > 65535:
        return None
    return host, int(port)


def _origin(raw: str) -> tuple[bool, str, int] | None:
    scheme, sep, site = raw.partition("://")
    https = scheme.lower() == "https"
    if not sep or not (https or scheme.lower() == "http"):
        return None
    got = _authority(site, True)
    if got is None:
        return None
    return https, got[0], got[1] if got[1] is not None else (443 if https else 80)


def _same_host(a: str, b: str) -> bool:
    if a.lower() == b.lower():
        return True
    try:
        return ipaddress.ip_address(a) == ipaddress.ip_address(b)
    except ValueError:
        return False


def _loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def allowed_origins() -> list[str]:
    return [o.strip() for o in os.environ.get("TENSORFOLD_ALLOWED_ORIGINS", "").split(",") if o.strip()]


def blocked(hosts: list[str], origins: list[str], loopback_bind: bool, allowed: list[str] = ()) -> str | None:
    """"host" or "origin" when the request must be refused, else None; ``hosts`` and ``origins`` are every value of
    each header the request sent."""

    if len(hosts) != 1:
        return "host"
    host = _authority(hosts[0], True)
    if host is None or (loopback_bind and not _loopback(host[0])):
        return "host"
    if not origins:
        return None
    origin = _origin(origins[0]) if len(origins) == 1 else None
    if origin is None:
        return "origin"
    https, name, port = origin
    if not https and port == (host[1] or 80) and _same_host(name, host[0]):
        return None
    for entry in allowed:
        trusted = _origin(entry)
        if trusted is not None and trusted[0] == https and trusted[2] == port and _same_host(trusted[1], name):
            return None
    return "origin"


def bind_is_loopback(address: str) -> bool:
    """Whether the server listens on loopback only (127.0.0.1, ::1 or localhost)."""

    return _loopback(address)
