"""
The shared Ollama transport refuses HTTP redirects (ai/ollama_http.py).

Why these tests use real sockets rather than the usual urlopen mock. Every
other transport test in this suite patches urllib.request.urlopen, which is the
right tool for retry counts and error translation but cannot say anything about
redirects: a mock never redirects, so a test built on one would assert that the
code does not follow something that was never offered. The claim being made
here - "the redirect destination is never contacted" - is a claim about network
behaviour, so it is verified against two real HTTP servers on loopback: one
that replies with a 3xx, and one that records every request it receives. The
assertion is that the recorder stays empty.

Both servers bind 127.0.0.1 with port 0, so the kernel picks a free port and
nothing leaves the machine. They are torn down in a finally block.

What is being protected: ai/local_endpoint.py validates the host that was
configured. Without this refusal, a server on that validated host could reply
301/302/307/308 and urllib would re-send the request - carrying the retrieval
query, which is built from repository source - to any address it named.
"""
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

import pytest

from sentinelai.ai import ollama_http
from sentinelai.ai.ollama_http import read_with_retry

BODY = b'{"model":"nomic-embed-text","input":["def login(pw): return pw == SECRET"]}'


def _serve(handler_cls) -> HTTPServer:
    server = HTTPServer(("127.0.0.1", 0), handler_cls)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _recording_server(received: list) -> HTTPServer:
    """A server that records every request it gets and answers 200."""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            received.append((self.command, self.path, self.rfile.read(length) if length else b""))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", "11")
            self.end_headers()
            self.wfile.write(b'{"ok":true}')

        do_GET = do_POST

        def log_message(self, *args):  # keep pytest output clean
            pass

    return _serve(Handler)


def _redirecting_server(code: int, location: str, received: list) -> HTTPServer:
    """A server that answers every request with `code` and a Location header."""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            received.append((self.command, self.path, self.rfile.read(length) if length else b""))
            self.send_response(code)
            self.send_header("Location", location)
            self.send_header("Content-Length", "0")
            self.end_headers()

        do_GET = do_POST

        def log_message(self, *args):
            pass

    return _serve(Handler)


def _post(url: str) -> urllib.request.Request:
    return urllib.request.Request(
        url, data=BODY, headers={"Content-Type": "application/json"}, method="POST"
    )


# ------------------------------------------------- redirects are refused

@pytest.mark.parametrize("code", [301, 302, 307, 308])
def test_redirect_is_refused_and_the_destination_is_never_contacted(code):
    destination_got, redirector_got = [], []
    destination = _recording_server(destination_got)
    redirector = _redirecting_server(
        code, f"http://127.0.0.1:{destination.server_port}/api/embed", redirector_got
    )
    try:
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            read_with_retry(
                _post(f"http://127.0.0.1:{redirector.server_port}/api/embed"),
                timeout=5.0,
                host=f"http://127.0.0.1:{redirector.server_port}",
            )
        assert excinfo.value.code == code, "the 3xx must surface as the failure, not be followed"
        assert destination_got == [], "the redirect destination was contacted"
        assert len(redirector_got) == 1, "the request was repeated"
    finally:
        for server in (redirector, destination):
            server.shutdown()
            server.server_close()


@pytest.mark.parametrize("code", [301, 302, 307, 308])
def test_the_request_body_is_not_resent_after_a_redirect(code):
    destination_got, redirector_got = [], []
    destination = _recording_server(destination_got)
    redirector = _redirecting_server(
        code, f"http://127.0.0.1:{destination.server_port}/api/embed", redirector_got
    )
    try:
        with pytest.raises(urllib.error.HTTPError):
            read_with_retry(
                _post(f"http://127.0.0.1:{redirector.server_port}/api/embed"),
                timeout=5.0,
                host="http://127.0.0.1",
            )
        # The body reached the validated host exactly once and nowhere else.
        assert [entry[2] for entry in redirector_got] == [BODY]
        assert all(entry[2] == b"" for entry in destination_got)
        assert destination_got == []
    finally:
        for server in (redirector, destination):
            server.shutdown()
            server.server_close()


def test_a_remote_location_is_not_resolved_or_contacted():
    """A remote Location must surface as the 3xx, not as a DNS or connection error.

    If the redirect were followed, the failure would be a URLError from resolving
    ollama.example.com. An HTTPError carrying 302 proves no attempt was made.
    """
    redirector_got = []
    redirector = _redirecting_server(302, "http://ollama.example.com/api/embed", redirector_got)
    try:
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            read_with_retry(
                _post(f"http://127.0.0.1:{redirector.server_port}/api/embed"),
                timeout=5.0,
                host="http://127.0.0.1",
            )
        assert excinfo.value.code == 302
        assert len(redirector_got) == 1
    finally:
        redirector.shutdown()
        redirector.server_close()


def test_a_redirect_is_not_retried():
    """3xx is absent from _RETRYABLE_HTTP_STATUS, so it fails on the first attempt."""
    redirector_got = []
    redirector = _redirecting_server(307, "http://127.0.0.1:1/x", redirector_got)
    try:
        with patch("time.sleep") as sleep:
            with pytest.raises(urllib.error.HTTPError):
                read_with_retry(
                    _post(f"http://127.0.0.1:{redirector.server_port}/api/embed"),
                    timeout=5.0,
                    host="http://127.0.0.1",
                    max_attempts=3,
                )
        assert len(redirector_got) == 1
        sleep.assert_not_called()
    finally:
        redirector.shutdown()
        redirector.server_close()


# ------------------------------------------- ordinary requests still work

def test_a_normal_local_request_still_succeeds():
    """The refusal must not disturb the path this transport exists to serve."""
    received = []
    server = _recording_server(received)
    try:
        body = read_with_retry(
            _post(f"http://127.0.0.1:{server.server_port}/api/embed"),
            timeout=5.0,
            host=f"http://127.0.0.1:{server.server_port}",
        )
        assert body == b'{"ok":true}'
        assert len(received) == 1
        assert received[0][2] == BODY
    finally:
        server.shutdown()
        server.server_close()


def test_connection_failure_still_raises_urlerror_after_retrying():
    """Existing retry and error handling are unchanged by the redirect refusal."""
    with patch("time.sleep") as sleep:
        with pytest.raises(urllib.error.URLError):
            # Port 1 on loopback: nothing listens, so this is a connection error.
            read_with_retry(_post("http://127.0.0.1:1/api/embed"), timeout=2.0, host="http://127.0.0.1", max_attempts=2)
    assert sleep.call_count == 1, "the single retry backoff should still have been taken"


def test_max_attempts_validation_is_unchanged():
    with pytest.raises(TypeError):
        read_with_retry(_post("http://127.0.0.1:1/x"), timeout=1.0, host="h", max_attempts=True)
    with pytest.raises(ValueError):
        read_with_retry(_post("http://127.0.0.1:1/x"), timeout=1.0, host="h", max_attempts=0)


# --------------------------------------------------- the opener is wired

# ------------------------------------------- proxies cannot reroute it

def test_http_proxy_cannot_intercept_a_loopback_request(monkeypatch):
    """A proxy must not receive a request addressed to a validated local host.

    Without ProxyHandler({}) urllib builds its proxy map from http_proxy, and
    with no_proxy unset a loopback request is handed to the proxy instead -
    carrying the prompt body. Verified against two real servers: the assertion
    is that the proxy's recorder stays empty and the real one does not.
    """
    proxy_got, real_got = [], []
    proxy = _recording_server(proxy_got)
    real = _recording_server(real_got)
    try:
        monkeypatch.setenv("http_proxy", f"http://127.0.0.1:{proxy.server_port}")
        monkeypatch.setenv("HTTP_PROXY", f"http://127.0.0.1:{proxy.server_port}")
        monkeypatch.delenv("no_proxy", raising=False)
        monkeypatch.delenv("NO_PROXY", raising=False)

        body = read_with_retry(
            _post(f"http://127.0.0.1:{real.server_port}/api/embed"),
            timeout=5.0,
            host=f"http://127.0.0.1:{real.server_port}",
        )
        assert body == b'{"ok":true}'
        assert proxy_got == [], "the proxy intercepted a loopback request"
        assert len(real_got) == 1, "the request did not reach the real local server"
        assert real_got[0][2] == BODY
    finally:
        for server in (proxy, real):
            server.shutdown()
            server.server_close()


def test_the_opener_carries_no_proxy_handler_at_all():
    """Absence is the guarantee, not presence.

    build_opener() only keeps a handler that contributes a *_open method, and
    ProxyHandler({}) registers none because its proxy map is empty (CPython
    skips proxy_open explicitly). So an opener built with ProxyHandler({})
    ends up with no proxy handler whatsoever - which is precisely the point:
    the default ProxyHandler, the one that reads http_proxy from the
    environment, never got installed. A ProxyHandler appearing here would mean
    environment proxies are back in play.
    """
    handlers = ollama_http._NO_PROXY_NO_REDIRECT_OPENER.handlers
    assert not [h for h in handlers if isinstance(h, urllib.request.ProxyHandler)]


def test_the_installed_opener_refuses_redirects():
    assert any(
        isinstance(handler, ollama_http._RefuseRedirects)
        for handler in ollama_http._NO_REDIRECT_OPENER.handlers
    )


def test_redirect_request_returns_none():
    """Returning None is what makes urllib surface the 3xx instead of following it."""
    handler = ollama_http._RefuseRedirects()
    assert handler.redirect_request(None, None, 302, "Found", {}, "http://evil.example.com") is None
