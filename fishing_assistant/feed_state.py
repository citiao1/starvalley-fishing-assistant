from __future__ import annotations

from dataclasses import dataclass


READY = "READY"
WAITING_TRANSITION = "WAITING_TRANSITION"
TRANSITION_SEEN = "TRANSITION_SEEN"
PAUSED = "PAUSED"


@dataclass(frozen=True)
class FeedDecision:
    should_send: bool
    transitioned: bool
    state: str
    reason: str


class FeedStateMachine:
    """Debounce feed actions while preserving the not_found transition state."""

    def __init__(
        self,
        confirm_frames: int = 3,
        min_interval_seconds: float = 0.35,
        retry_interval_seconds: float = 0.75,
        input_failure_limit: int = 3,
    ) -> None:
        self.confirm_frames = max(1, int(confirm_frames))
        self.min_interval_seconds = max(0.2, float(min_interval_seconds))
        self.retry_interval_seconds = max(
            self.min_interval_seconds,
            float(retry_interval_seconds),
        )
        self.input_failure_limit = max(1, int(input_failure_limit))
        self.reset()

    def reset(self) -> None:
        self.state = READY
        self.zero_streak = 0
        self.next_allowed_at = 0.0
        self.next_retry_at = 0.0
        self.input_failures = 0

    def reconfigure(
        self,
        confirm_frames: int,
        min_interval_seconds: float,
        retry_interval_seconds: float,
        input_failure_limit: int,
    ) -> None:
        self.confirm_frames = max(1, int(confirm_frames))
        self.min_interval_seconds = max(0.2, float(min_interval_seconds))
        self.retry_interval_seconds = max(
            self.min_interval_seconds,
            float(retry_interval_seconds),
        )
        self.input_failure_limit = max(1, int(input_failure_limit))

    def observe(
        self,
        label: str,
        now: float,
        enabled: bool = True,
    ) -> FeedDecision:
        if not enabled:
            self.reset()
            return FeedDecision(False, False, self.state, "disabled")

        if label == "zero":
            self.zero_streak += 1
        else:
            self.zero_streak = 0

        transitioned = False
        if self.state == WAITING_TRANSITION and label != "zero":
            self.state = TRANSITION_SEEN
            transitioned = True
        if self.state == TRANSITION_SEEN and now >= self.next_allowed_at:
            self.state = READY

        if self.state == PAUSED:
            return FeedDecision(False, transitioned, self.state, "input_failures")

        if self.state == READY:
            if self.zero_streak >= self.confirm_frames and now >= self.next_retry_at:
                return FeedDecision(True, transitioned, self.state, "zero_confirmed")
            return FeedDecision(False, transitioned, self.state, "waiting_zero")

        if (
            self.state == WAITING_TRANSITION
            and not transitioned
            and label == "zero"
            and self.zero_streak >= self.confirm_frames
            and now >= self.next_retry_at
        ):
            return FeedDecision(True, False, self.state, "retry_without_transition")

        return FeedDecision(False, transitioned, self.state, "waiting_transition")

    def mark_send(self, now: float, success: bool) -> None:
        self.zero_streak = 0
        self.next_retry_at = now + self.retry_interval_seconds
        if success:
            self.input_failures = 0
            self.state = WAITING_TRANSITION
            self.next_allowed_at = now + self.min_interval_seconds
            return

        self.input_failures += 1
        if self.input_failures >= self.input_failure_limit:
            self.state = PAUSED
        else:
            self.state = READY
