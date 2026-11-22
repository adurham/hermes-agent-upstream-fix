"""A 400 for the model's own malformed tool-call JSON must not walk the fallback chain (#12770)."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.error_classifier import FailoverReason, classify_api_error
from agent.turn_api_error import settle_unrecovered_error


class _Err(Exception):
    status_code = 400
    response = None

    def __init__(self, message, body=None):
        super().__init__(message)
        self.body = body or {"error": {"message": message, "type": "invalid_request_error"}}


@pytest.mark.parametrize("wording", ["invalid tool call arguments", "Invalid function_call arguments"])
def test_malformed_tool_args_400_is_terminal_without_fallback(wording):
    """Large session included: the bare-ish message must not read as context overflow either."""
    verdict = classify_api_error(_Err(f"Error code: 400 - {wording}"), provider="ollama",
                                 approx_tokens=90_000, context_length=128_000, num_messages=120)
    assert verdict.reason is FailoverReason.format_error
    assert (verdict.retryable, verdict.should_compress, verdict.should_fallback) == (False, False, False)

    # Unrelated request-shape 400s keep their fallback (another provider may accept the request).
    assert classify_api_error(_Err("Unsupported parameter: 'max_tokens'")).should_fallback is True


class _Agent:
    """Only the fallback seam is real; every other helper the terminal path touches is a no-op."""
    log_prefix = ""
    verbose = False
    provider = "ollama"
    _fallback_chain = [object()]
    _fallback_index = 0
    _credential_pool = None

    def __init__(self):
        self.activated = []

    def _has_pending_fallback(self):
        return True

    def _try_activate_fallback(self, **kwargs):
        self.activated.append(True)
        return True

    def _summarize_api_error(self, error):
        return str(error)

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def _settle(agent, err, provider, status_code):
    retry = SimpleNamespace(provider_stale_refresh_attempted=False, primary_recovery_attempted=False)
    classified = classify_api_error(err, provider=provider)
    with patch("agent.conversation_loop._credential_refresh_profile", lambda a: None):
        return settle_unrecovered_error(
            agent, api_error=err, classified=classified, _retry=retry, status_code=status_code, error_msg=str(err),
            is_context_length_error=False, is_rate_limited=False, _is_zai_coding_overload=False,
            _provider=provider, _base="http://127.0.0.1:11434/v1", _model="glm", messages=[], api_messages=[],
            api_kwargs={}, active_system_prompt="", conversation_history=None, approx_tokens=10,
            retry_count=0, max_retries=3, compression_attempts=0, api_call_count=1,
        )


def test_client_error_settlement_skips_fallback_when_classifier_says_so():
    agent = _Agent()
    verdict = _settle(agent, _Err("invalid tool call arguments"), "ollama", 400)
    assert verdict.action == "return"
    assert verdict.result["failure_reason"] == FailoverReason.format_error.value
    assert agent.activated == []


def test_malformed_tool_args_repairs_the_wire_copy_and_reasks_once():
    """FORK: the model's own bad tool-call JSON is stochastic (the same body can 400 then 200),
    so the no-fallback verdict still gets one same-route re-ask with the replayed arguments
    repaired — never a provider switch, and never a loop."""
    from agent.turn_recovery import _recover_format_errors
    from agent.turn_retry_state import TurnRetryState

    agent = _Agent()
    agent.model = "deepseek-v4.1-flash:1017"
    bad_args = '{"content":"build caab2fa1f8 \\uZZZZ done"}'
    api_messages = [
        {"role": "assistant", "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "write_file", "arguments": bad_args}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
    ]
    retry = TurnRetryState()
    err = _Err("Error code: 400 - {'error': {'message': 'invalid tool call arguments (ref: abc)'}}")
    classified = classify_api_error(err, provider="ollama")

    assert classified.reason is FailoverReason.format_error
    assert classified.should_fallback is False

    assert _recover_format_errors(agent, err, classified, retry, [], api_messages) is True
    # The wire copy is healed; canonical history is never touched here.
    assert api_messages[0]["tool_calls"][0]["function"]["arguments"] == "{}"
    assert retry.malformed_tool_args_retry_attempted is True

    # One-shot: the second miss settles the error instead of re-asking.
    assert _recover_format_errors(agent, err, classified, retry, [], api_messages) is False


def test_unrelated_format_errors_are_not_retried_by_the_malformed_rung():
    from agent.turn_recovery import _recover_format_errors
    from agent.turn_retry_state import TurnRetryState

    agent = _Agent()
    err = _Err("Unsupported parameter: 'max_tokens'")
    classified = classify_api_error(err, provider="ollama")
    retry = TurnRetryState()
    assert _recover_format_errors(agent, err, classified, retry, [], []) is False
    assert retry.malformed_tool_args_retry_attempted is False


def test_fallback_free_verdict_wins_over_local_valueerror_shape():
    """A recognised no-fallback verdict raised as a ValueError subclass (MoA preset missing,
    #55933) must not sneak through the unclassified-local-error fallback allowance; a truly
    unclassified ValueError keeps it."""
    from agent.errors import MoAPresetNotFoundError

    agent = _Agent()
    verdict = _settle(agent, MoAPresetNotFoundError("MoA preset 'old' was not found"), "moa", None)
    assert verdict.action == "return"
    assert agent.activated == []

    agent = _Agent()
    verdict = _settle(agent, ValueError("some local bug"), "openai", None)
    assert verdict.action == "break"
    assert agent.activated == [True]
