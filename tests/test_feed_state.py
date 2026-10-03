from __future__ import annotations

import unittest

from fishing_assistant.feed_state import (
    PAUSED,
    READY,
    TRANSITION_SEEN,
    WAITING_TRANSITION,
    FeedStateMachine,
)


class FeedStateMachineTests(unittest.TestCase):
    def make_machine(self) -> FeedStateMachine:
        return FeedStateMachine(
            confirm_frames=3,
            min_interval_seconds=0.35,
            retry_interval_seconds=0.75,
            input_failure_limit=2,
        )

    def test_not_found_is_transition_and_does_not_retrigger(self) -> None:
        machine = self.make_machine()
        machine.observe("zero", 0.0)
        machine.observe("zero", 0.0)
        self.assertTrue(machine.observe("zero", 0.0).should_send)
        machine.mark_send(0.0, True)
        self.assertEqual(machine.state, WAITING_TRANSITION)

        machine.observe("not_found", 0.1)
        self.assertEqual(machine.state, TRANSITION_SEEN)
        for now in (0.2, 0.3, 0.4, 0.5):
            self.assertFalse(machine.observe("zero", now).should_send)

    def test_zero_retries_only_after_interval_without_transition(self) -> None:
        machine = self.make_machine()
        machine.observe("zero", 0.0)
        machine.observe("zero", 0.0)
        self.assertTrue(machine.observe("zero", 0.0).should_send)
        machine.mark_send(0.0, True)
        for now in (0.1, 0.2, 0.7):
            machine.observe("zero", now)
        self.assertFalse(machine.observe("zero", 0.74).should_send)
        self.assertTrue(machine.observe("zero", 0.75).should_send)

    def test_failed_inputs_pause_after_limit(self) -> None:
        machine = self.make_machine()
        machine.observe("zero", 0.0)
        machine.observe("zero", 0.0)
        self.assertTrue(machine.observe("zero", 0.0).should_send)
        machine.mark_send(0.0, False)
        self.assertEqual(machine.state, READY)
        for now in (0.75, 1.5):
            machine.observe("zero", now)
        machine.mark_send(1.5, False)
        self.assertEqual(machine.state, PAUSED)
        self.assertFalse(machine.observe("zero", 4.0).should_send)


if __name__ == "__main__":
    unittest.main()
