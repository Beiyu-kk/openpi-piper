import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import main_rtc  # noqa: E402


def test_realtime_action_stream_skips_actions_executed_while_inference_was_pending():
    initial_chunk = np.arange(30 * 7, dtype=np.float32).reshape(30, 7)
    next_chunk = initial_chunk + 1000.0
    stream = main_rtc.RealtimeActionStream(initial_chunk)

    request = stream.start_request(action_step=0)
    np.testing.assert_array_equal(stream.next_action(), initial_chunk[0])
    np.testing.assert_array_equal(stream.next_action(), initial_chunk[1])
    np.testing.assert_array_equal(stream.next_action(), initial_chunk[2])

    delay_steps = stream.accept_response(request, next_chunk)

    assert delay_steps == 3
    np.testing.assert_array_equal(stream.next_action(), next_chunk[3])


def test_realtime_action_stream_uses_zero_delay_when_response_returns_before_next_action():
    initial_chunk = np.arange(30 * 7, dtype=np.float32).reshape(30, 7)
    next_chunk = initial_chunk + 1000.0
    stream = main_rtc.RealtimeActionStream(initial_chunk)

    request = stream.start_request(action_step=0)
    delay_steps = stream.accept_response(request, next_chunk)

    assert delay_steps == 0
    np.testing.assert_array_equal(stream.next_action(), next_chunk[0])


def test_realtime_action_stream_rejects_too_late_responses():
    initial_chunk = np.arange(3 * 7, dtype=np.float32).reshape(3, 7)
    next_chunk = initial_chunk + 1000.0
    stream = main_rtc.RealtimeActionStream(initial_chunk)
    request = stream.start_request(action_step=0)

    for _ in range(3):
        stream.next_action()

    try:
        stream.accept_response(request, next_chunk)
    except RuntimeError as exc:
        assert "arrived after 3 executed actions" in str(exc)
    else:
        raise AssertionError("Expected a too-late response to be rejected.")
