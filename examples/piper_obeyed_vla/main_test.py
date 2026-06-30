from __future__ import annotations

import numpy as np

from examples.piper_obeyed_vla import main
from examples.piper_obeyed_vla import adapter


def test_action_chunk_cache_calls_perception_only_when_replanning() -> None:
    calls = []

    class FakePerception:
        def process(self, **kwargs):
            calls.append((kwargs["is_base_init"], kwargs["is_wrist_init"]))
            return adapter.PerceptionResult(
                base_rgb=kwargs["base_rgb"] + 10,
                wrist_rgb=kwargs["wrist_rgb"] + 20,
            )

    cache = main.GroundedObservationBuilder(
        perception_client=FakePerception(),
        prompt="pick book",
        image_size=8,
        select_objects="book",
        exclude_objects="",
        wrist_init_period=2,
    )
    head = np.ones((4, 4, 3), dtype=np.uint8)
    wrist = np.ones((4, 4, 3), dtype=np.uint8)
    state = np.zeros((7,), dtype=np.float32)

    first = cache.build_if_replanning(
        should_replan=True,
        step=0,
        head_rgb=head,
        wrist_rgb=wrist,
        state=state,
    )
    second = cache.build_if_replanning(
        should_replan=False,
        step=1,
        head_rgb=head,
        wrist_rgb=wrist,
        state=state,
    )
    third = cache.build_if_replanning(
        should_replan=True,
        step=2,
        head_rgb=head,
        wrist_rgb=wrist,
        state=state,
    )

    assert first is second
    assert third is not first
    assert calls == [(True, True), (False, True)]
    assert first["observation/image"].shape == (8, 8, 3)
    assert first["prompt"] == "pick book"
