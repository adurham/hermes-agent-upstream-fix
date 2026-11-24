"""The CONNECT phase has its own short budget; it must never eat the generation budget.

A stalled TCP connect / TLS handshake used to be allowed the whole
``auxiliary.<task>.timeout``: openai-python turns a scalar ``timeout=`` into
``httpx.Timeout(total)``, which applies that budget to the connect phase too. A dead edge therefore
burned ~20 minutes (primary + same-provider retry) before the fallback chain was even tried
(ps5-pm, 2026-11-23: ``httpx.ConnectTimeout: The handshake operation timed out``).

These tests pin both halves of the contract against real sockets, not guesses:

* a stalled handshake fails inside the connect cap and the chain reaches its fallback in that same
  budget;
* a slow-but-alive call still gets the full generation budget — the connect cap must not regress into
  a total cap;
* the internal guards still see the numeric budget, never the wire object.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import openai
import pytest

from agent import auxiliary_client as ac

BUDGET = 120.0
# The acceptance bound: a dead edge must be given up on, and the fallback reached, well inside this.
DEAD_EDGE_DEADLINE = 15.0


# ── a socket that accepts TCP and never completes a TLS handshake ────────────────────────────────
class _StallServer:
    """Accept connections and never write a byte: exactly what a dead/blackholed edge looks like."""

    def __init__(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(16)
        self._held: list[socket.socket] = []
        self._closed = False
        self.port = self._sock.getsockname()[1]
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self) -> None:
        while not self._closed:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            self._held.append(conn)

    @property
    def base_url(self) -> str:
        return f"https://127.0.0.1:{self.port}/v1"

    def close(self) -> None:
        self._closed = True
        for conn in self._held:
            conn.close()
        self._sock.close()


@pytest.fixture()
def stall_server():
    server = _StallServer()
    try:
        yield server
    finally:
        server.close()


def _real_client(base_url: str) -> openai.OpenAI:
    return openai.OpenAI(api_key="sk-test", base_url=base_url, max_retries=0)


def _no_fallbacks():
    """No configured/discovered fallback may be reached: these tests are about the primary only."""
    return (
        patch.object(ac, "_try_configured_fallback_chain", return_value=(None, None, "")),
        patch.object(ac, "_try_main_agent_model_fallback", return_value=(None, None, "")),
        patch.object(ac, "_try_main_fallback_chain", return_value=(None, None, "")),
        patch.object(ac, "_try_payment_fallback", return_value=(None, None, "")),
    )


def _cause_chain(exc: BaseException):
    seen, current = set(), exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__


# ── the wire timeout itself ─────────────────────────────────────────────────────────────────────
def test_a_scalar_budget_becomes_a_split_timeout():
    wire = ac._wire_timeout(BUDGET)
    assert isinstance(wire, httpx.Timeout)
    assert wire.connect == ac._AUX_CONNECT_TIMEOUT_SECONDS
    assert (wire.read, wire.write, wire.pool) == (BUDGET, BUDGET, BUDGET)


def test_a_tiny_budget_keeps_the_connect_cap_below_it():
    """``connect`` must never exceed the caller's own budget (a 3 s call cannot wait 10 s to connect)."""
    wire = ac._wire_timeout(3.0)
    assert wire.connect == 3.0
    assert wire.read == 3.0


@pytest.mark.parametrize("value", [None, 0, -1, "not-a-number"])
def test_a_non_budget_passes_through_untouched(value):
    """No per-request override: the client's own timeout keeps governing, exactly as before."""
    assert ac._wire_timeout(value) is value


def test_the_wire_timeout_is_idempotent():
    wire = httpx.Timeout(BUDGET, connect=1.0)
    assert ac._wire_timeout(wire) is wire  # a rebuilt/re-sent request keeps its split


def test_internal_arithmetic_reads_the_budget_not_the_object():
    wire = ac._wire_timeout(300.0)
    assert ac._timeout_seconds(wire) == 300.0
    assert ac._aux_stream_total_ceiling(wire) == 4 * 300.0  # arithmetic, not a TypeError
    assert ac._anthropic_no_progress_timeout(wire) == ac._AUX_STREAM_NO_PROGRESS_TIMEOUT_SECONDS
    assert ac._anthropic_no_progress_timeout(ac._wire_timeout(30.0)) == 30.0  # clamped by the budget


@pytest.mark.parametrize("value", [None, 0, -1, "30", ""])
def test_a_non_budget_is_never_forwarded_as_a_timeout(value):
    """The guards keep their old strictness: a string/None/0 still means "use the client default"."""
    assert ac._is_positive_timeout(value) is False


def test_a_real_budget_is_forwarded():
    assert ac._is_positive_timeout(30.0) is True
    assert ac._is_positive_timeout(ac._wire_timeout(30.0)) is True


def test_the_codex_stream_guard_still_enforces_a_total_ceiling():
    """A wire ``httpx.Timeout`` must still arm the guard: without the coercion it went unbounded."""

    class _SlowAliveStream:
        def __iter__(self):
            for _ in range(50):
                time.sleep(0.03)
                yield SimpleNamespace(type="response.in_progress")

        def close(self) -> None:
            pass

    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **_: _SlowAliveStream()),
                             close=lambda: None)
    adapter = ac._CodexCompletionsAdapter(client, "gpt-5.5")

    started = time.monotonic()
    with pytest.raises(TimeoutError):
        adapter.create(messages=[{"role": "user", "content": "x"}],
                       timeout=httpx.Timeout(0.05, connect=0.05))
    assert time.monotonic() - started < 2.0


# ── a dead edge: fails fast, and the chain moves on in the same budget ───────────────────────────
def test_a_stalled_tls_handshake_fails_inside_the_connect_cap(stall_server, caplog):
    client = _real_client(stall_server.base_url)
    p1, p2, p3, p4 = _no_fallbacks()
    with patch.object(ac, "_resolve_task_provider_model",
                      return_value=("custom", "m", stall_server.base_url, "sk-test", None)), \
         patch.object(ac, "_get_cached_client", return_value=(client, "m")), \
         patch.object(ac, "_get_auxiliary_task_config", return_value={}), \
         p1, p2, p3, p4:
        started = time.monotonic()
        with pytest.raises(Exception) as excinfo:
            ac.call_llm(task="consult", messages=[{"role": "user", "content": "hi"}], timeout=BUDGET)
        elapsed = time.monotonic() - started

    assert elapsed < DEAD_EDGE_DEADLINE, f"a stalled handshake burned {elapsed:.1f}s of a {BUDGET}s budget"
    # The failure really is the connect phase, not a read timeout.
    assert any(isinstance(cause, httpx.ConnectTimeout) for cause in _cause_chain(excinfo.value))

    # ... and the log names the connect cap instead of telling the user to raise a timeout knob.
    assert "could not connect" in caplog.text
    assert "raise auxiliary.timeout" not in caplog.text


def test_the_fallback_chain_is_reached_inside_the_connect_cap(stall_server):
    primary = _real_client(stall_server.base_url)
    fallback = MagicMock()
    fallback.base_url = "https://fallback.invalid/v1"
    fallback_response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="from-fallback"))])
    fallback.chat.completions.create.return_value = fallback_response

    with patch.object(ac, "_resolve_task_provider_model",
                      return_value=("custom", "m", stall_server.base_url, "sk-test", None)), \
         patch.object(ac, "_get_cached_client", return_value=(primary, "m")), \
         patch.object(ac, "_get_auxiliary_task_config", return_value={}), \
         patch.object(ac, "_try_configured_fallback_chain",
                      return_value=(fallback, "fb-model", "fallback_chain[0](stub)")), \
         patch.object(ac, "_validate_llm_response", side_effect=lambda resp, *_a, **_kw: resp):
        started = time.monotonic()
        resp = ac.call_llm(task="consult", messages=[{"role": "user", "content": "hi"}], timeout=BUDGET)
        elapsed = time.monotonic() - started

    assert resp is fallback_response, "the chain never reached its fallback"
    assert elapsed < DEAD_EDGE_DEADLINE, f"the fallback was reached only after {elapsed:.1f}s"

    # The fallback request carries the same split: a rebuilt request must not lose the connect cap,
    # and must keep the whole generation budget for a slow-but-alive reasoning model.
    sent = fallback.chat.completions.create.call_args.kwargs["timeout"]
    assert isinstance(sent, httpx.Timeout)
    assert sent.connect == ac._AUX_CONNECT_TIMEOUT_SECONDS
    assert sent.read == BUDGET


# ── a slow-but-alive call keeps the whole generation budget ─────────────────────────────────────
class _SlowAliveHandler(BaseHTTPRequestHandler):
    delay = 11.5  # longer than the connect cap: only the read budget may cover this

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("content-length") or 0))
        time.sleep(self.delay)
        body = json.dumps({
            "id": "chatcmpl-test", "object": "chat.completion", "created": 0, "model": "slow-model",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "slow but alive"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:
        pass


def test_a_slow_but_alive_call_gets_the_whole_generation_budget():
    """The connect cap must never become a total cap: a response arriving after it still lands."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _SlowAliveHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}/v1"
    client = _real_client(base_url)
    p1, p2, p3, p4 = _no_fallbacks()
    try:
        with patch.object(ac, "_resolve_task_provider_model",
                          return_value=("custom", "slow-model", base_url, "sk-test", None)), \
             patch.object(ac, "_get_cached_client", return_value=(client, "slow-model")), \
             patch.object(ac, "_get_auxiliary_task_config", return_value={}), \
             patch.object(ac, "_validate_llm_response", side_effect=lambda resp, *_a, **_kw: resp), \
             p1, p2, p3, p4:
            started = time.monotonic()
            resp = ac.call_llm(task="consult", messages=[{"role": "user", "content": "hi"}], timeout=60)
            elapsed = time.monotonic() - started
    finally:
        server.shutdown()

    assert elapsed > ac._AUX_CONNECT_TIMEOUT_SECONDS, "the test server answered too fast to prove anything"
    assert resp.choices[0].message.content == "slow but alive"


# ── config semantics are untouched ─────────────────────────────────────────────────────────────
def test_the_task_timeout_config_still_governs_the_budget():
    wire = ac._wire_timeout(ac._effective_aux_timeout("consult", None))
    assert wire.read == ac._get_task_timeout("consult")
    assert wire.connect == ac._AUX_CONNECT_TIMEOUT_SECONDS
