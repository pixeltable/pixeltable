# type: ignore
"""
Tests to observe and track the over-throttling bug caused by inflated token estimates.

Background
----------
RateLimitsScheduler.`_resource_delay()` computes a pre-request wait time using:

    target_remaining = ceil(limit * 0.05 + next_request_estimate + est_usage)

where `next_request_estimate` comes from `_chat_completions_get_request_resources()`.
That function uses `_default_max_tokens(model)` (e.g. 16 384 for gpt-4o-mini) when the
caller hasn't set `max_tokens`. Actual completions are often only ~100-300 tokens, so
the estimate can be 50-80x too large.

As a result the scheduler starts throttling at remaining ≈ 18 584 (for a 40 000 TPM
limit) instead of at remaining ≈ 2 200, wasting 40 % of available token budget with
unnecessary sleeps.

Running
-------
    pytest tests/test_rate_limit_estimation.py -v
"""

import asyncio
import datetime
import math
from unittest.mock import AsyncMock, Mock, patch

import pytest

from pixeltable import env, func
from pixeltable.env import TARGET_RATE_LIMIT_RESOURCE_FRACT, RateLimitInfo, RateLimitsInfo
from pixeltable.exec.expr_eval.globals import Dispatcher, ExprEvalCtx, FnCallArgs
from pixeltable.exec.expr_eval.schedulers import RateLimitsScheduler
from pixeltable.functions.openai import _chat_completions_get_request_resources, _default_max_tokens

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rate_limit_info(limit: int, remaining: int, window_secs: float = 60.0) -> RateLimitInfo:
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    reset_at = now + datetime.timedelta(seconds=window_secs)
    return RateLimitInfo(resource='tokens', request_start_ts=now, limit=limit, remaining=remaining, reset_at=reset_at)


def _make_scheduler_with_pool(
    limit: int, remaining: int, window_secs: float = 60.0, est_usage: int = 0
) -> RateLimitsScheduler:
    """
    Return a RateLimitsScheduler with pool_info pre-seeded, bypassing __init__
    (which calls asyncio.create_task and requires a running event loop).
    Only the attributes needed by _resource_delay are populated.
    """
    # Bypass __init__ to avoid asyncio.create_task requirement.
    scheduler = object.__new__(RateLimitsScheduler)

    rate_info = _make_rate_limit_info(limit, remaining, window_secs)

    mock_pool = Mock(spec=RateLimitsInfo)
    mock_pool.resource_limits = {'tokens': rate_info}
    mock_pool.is_initialized = Mock(return_value=True)
    mock_pool.get_request_resources = Mock(return_value={'tokens': 0})

    scheduler.pool_info = mock_pool
    scheduler.est_usage = {'tokens': est_usage}
    return scheduler


# ---------------------------------------------------------------------------
# 1. Low-level: RateLimitInfo.estimated_resource_refill_delay
# ---------------------------------------------------------------------------


class TestEstimatedResourceRefillDelay:
    """Directly probe the refill-delay math so we can reason about each input."""

    def test_no_wait_when_remaining_is_sufficient(self) -> None:
        info = _make_rate_limit_info(limit=40_000, remaining=20_000)
        # Target well below current remaining → no wait.
        assert info.estimated_resource_refill_delay(5_000) == 0

    def test_wait_when_remaining_below_target(self) -> None:
        info = _make_rate_limit_info(limit=40_000, remaining=5_000)
        # Ask for 10 000 to be available → must wait.
        delay = info.estimated_resource_refill_delay(10_000)
        assert delay is not None and delay > 0

    def test_returns_none_when_target_exceeds_limit(self) -> None:
        """If the target itself exceeds the limit it can never be satisfied linearly."""
        info = _make_rate_limit_info(limit=40_000, remaining=10_000)
        assert info.estimated_resource_refill_delay(50_000) is None


# ---------------------------------------------------------------------------
# 2. Core bug: over-estimated tokens → unnecessary throttle
# ---------------------------------------------------------------------------


class TestOverThrottlingDueToTokenEstimate:
    """
    Shows that _resource_delay() returns a positive wait when the per-request
    token estimate is much larger than actual usage, even when real capacity is fine.
    """

    # Scenario: 40 000 TPM, 15 000 remaining (62.5 % still available).
    # Actual requests use ~200 tokens each. Default estimate is 16 584 tokens.
    LIMIT = 40_000
    REMAINING = 15_000  # plenty of real capacity
    ACTUAL_TOKENS = 200  # realistic short completion
    DEFAULT_ESTIMATED_TOKENS = 16_584  # _default_max_tokens('gpt-4o-mini') + ~200 prompt

    def _delay_for(self, per_request_tokens: int, est_usage: int = 0) -> float:
        scheduler = _make_scheduler_with_pool(limit=self.LIMIT, remaining=self.REMAINING, est_usage=est_usage)
        return scheduler._resource_delay({'tokens': per_request_tokens})

    def test_no_wait_with_accurate_estimate(self) -> None:
        """With accurate token estimates, no wait should be added."""
        delay = self._delay_for(self.ACTUAL_TOKENS)
        assert delay == 0, (
            f'Expected 0s delay with accurate estimate ({self.ACTUAL_TOKENS} tokens) '
            f'but got {delay}s. remaining={self.REMAINING}/{self.LIMIT}'
        )

    def test_unnecessary_wait_with_default_estimate(self) -> None:
        """
        BUG: With the default (inflated) estimate, the scheduler adds a positive
        wait even though actual capacity is well above what is needed.

        When this test FAILS (delay == 0), the bug is fixed.
        """
        delay = self._delay_for(self.DEFAULT_ESTIMATED_TOKENS)
        # Document current (broken) behaviour: delay > 0 despite plenty of capacity.
        assert delay > 0, (
            f'BUG APPEARS FIXED: Expected unnecessary delay > 0 with default estimate '
            f'({self.DEFAULT_ESTIMATED_TOKENS} tokens) but got {delay}s. '
            f'remaining={self.REMAINING}/{self.LIMIT}'
        )

    def test_est_usage_accumulation_flips_to_no_throttle(self) -> None:
        """
        When est_usage (in-flight) + next_request estimate exceeds the limit,
        estimated_resource_refill_delay() returns None and _resource_delay() falls
        through to 0 — the scheduler stops throttling entirely.

        This reveals a second failure mode: the overestimate causes the system to
        oscillate between unnecessary waiting (few in-flight) and no throttling at all
        (many in-flight), meaning actual 429 errors are hit instead.

        Timeline with LIMIT=40_000 and DEFAULT_ESTIMATED_TOKENS=16_584:
          0 in-flight: target = 18_584 → wait ~9s  (over-throttle)
          1 in-flight: target = 35_168 → wait (varies)
          2 in-flight: target = 51_752 > limit → None → wait = 0  (under-throttle → 429s)

        With accurate estimates (ACTUAL_TOKENS=200):
          Any number of in-flight: target stays well under limit → behaves correctly.
        """
        # With 2 in-flight using default estimates, target exceeds the limit.
        est_usage_2_requests_default = 2 * self.DEFAULT_ESTIMATED_TOKENS
        target_when_2_in_flight = math.ceil(
            self.LIMIT * TARGET_RATE_LIMIT_RESOURCE_FRACT + self.DEFAULT_ESTIMATED_TOKENS + est_usage_2_requests_default
        )
        assert target_when_2_in_flight > self.LIMIT, (
            'Precondition: target should exceed the limit when 2 requests are in-flight'
        )

        # Because target > limit, estimated_resource_refill_delay returns None,
        # _resource_delay returns 0 — no throttle, even though remaining is low.
        delay_2_in_flight_default = self._delay_for(
            self.DEFAULT_ESTIMATED_TOKENS, est_usage=est_usage_2_requests_default
        )
        assert delay_2_in_flight_default == 0, (
            f'Expected no throttle (None guard) with 2 in-flight default estimates, got {delay_2_in_flight_default}s'
        )

        # With accurate estimates, 2 in-flight is still well below the limit.
        delay_2_in_flight_accurate = self._delay_for(self.ACTUAL_TOKENS, est_usage=2 * self.ACTUAL_TOKENS)
        assert delay_2_in_flight_accurate == 0, (
            f'With accurate estimates and 2 in-flight, expected 0s delay but got {delay_2_in_flight_accurate}s'
        )


# ---------------------------------------------------------------------------
# 3. default_max_tokens values for common models
# ---------------------------------------------------------------------------


class TestDefaultMaxTokensValues:
    """
    Documents the mismatch between scheduling estimates and typical usage.
    These numbers feed directly into _chat_completions_get_request_resources.
    """

    @pytest.mark.parametrize(
        'model,expected_default',
        [
            ('gpt-4o', 16_384),
            ('gpt-4o-mini', 16_384),  # matches gpt-4o family via startswith('gpt-4o-')
            ('gpt-4-turbo', 4_096),
            ('o1', 100_000),
            ('o3-mini', 100_000),
        ],
    )
    def test_known_defaults(self, model: str, expected_default: int) -> None:
        assert _default_max_tokens(model) == expected_default

    def test_overestimate_ratio_for_typical_usage(self) -> None:
        """Quantify how far off the estimate is from a typical short completion."""
        typical_completion_tokens = 200
        model = 'gpt-4o-mini'
        default = _default_max_tokens(model)
        ratio = default / typical_completion_tokens
        # Ratio is currently ~82x. Once fixed this test should be updated.
        assert ratio >= 50, (
            f'Default estimate for {model} ({default}) is only {ratio:.0f}x typical usage — '
            f'perhaps the fix reduced the default? Update this test threshold accordingly.'
        )


# ---------------------------------------------------------------------------
# 4. _chat_completions_get_request_resources: prompt-only vs full estimate
# ---------------------------------------------------------------------------


class TestChatCompletionsResourceEstimate:
    """
    Shows what the scheduler actually receives as its per-request resource estimate.
    """

    PROMPT = [{'role': 'user', 'content': 'What is 2+2?'}]  # ~10 tokens

    def test_estimate_without_max_tokens_uses_default(self) -> None:
        """When max_tokens is not specified, estimate is dominated by the default max."""
        resources = _chat_completions_get_request_resources(
            messages=self.PROMPT, model='gpt-4o-mini', model_kwargs=None
        )
        assert resources['tokens'] >= _default_max_tokens('gpt-4o-mini'), (
            'Expected estimate to be at least _default_max_tokens when max_tokens is unset'
        )

    def test_estimate_with_max_tokens_is_bounded(self) -> None:
        """When max_tokens=256, the estimate is bounded to a sensible range."""
        resources = _chat_completions_get_request_resources(
            messages=self.PROMPT, model='gpt-4o-mini', model_kwargs={'max_tokens': 256}
        )
        assert resources['tokens'] < 512, (
            f'Expected token estimate to be <512 with max_tokens=256, got {resources["tokens"]}'
        )

    def test_estimate_gap_between_max_tokens_and_no_max_tokens(self) -> None:
        """Quantify the scheduling difference between specifying vs not specifying max_tokens."""
        with_max = _chat_completions_get_request_resources(
            messages=self.PROMPT, model='gpt-4o-mini', model_kwargs={'max_tokens': 256}
        )
        without_max = _chat_completions_get_request_resources(
            messages=self.PROMPT, model='gpt-4o-mini', model_kwargs=None
        )
        ratio = without_max['tokens'] / with_max['tokens']
        assert ratio > 30, (
            f'Expected large estimation gap (>30x) between no max_tokens and max_tokens=256, '
            f'got {ratio:.1f}x ({without_max["tokens"]} vs {with_max["tokens"]} tokens). '
            f'If ratio is small, the estimation default may have been reduced (good!).'
        )
