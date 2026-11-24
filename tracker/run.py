"""Run a whole scenario through a tracker, frame by frame, arrival order applied."""

from __future__ import annotations

from .engine import GreedyAssociator, TrackerConfig, WindowedAssociator
from .model import Detection


def detections_for(scenario: dict, frame: int) -> list[Detection]:
    late = {int(k) for k in (scenario.get("late_schedule") or {})}
    return [
        Detection(id=d["id"], frame=frame, x=d["x"], y=d["y"], late=frame in late)
        for d in scenario["frames"][frame]["detections"]
    ]


def make_tracker(mode: str, scenario: dict, cfg: TrackerConfig | None = None):
    cls = GreedyAssociator if mode == "greedy" else WindowedAssociator
    return cls(cfg or TrackerConfig(), scenario)


def run_scenario(scenario: dict, mode: str, cfg: TrackerConfig | None = None,
                 start: int | None = None, end: int | None = None) -> dict:
    tracker = make_tracker(mode, scenario, cfg)
    start = 0 if start is None else start
    end = scenario["n_frames"] - 1 if end is None else end
    for step in range(start, end + 1):
        # Step == frame for on-time frames; a scheduled-late frame is handed over
        # at its scheduled step, which is exactly how the window sees a delay.
        tracker.ingest(step, detections_for(scenario, step), step=step)
    # Flush frames still held back by the schedule.
    held = sorted(f for f in tracker.det_log if f not in tracker.delivered) \
        if hasattr(tracker, "delivered") else []
    for f in held:
        tracker.ingest(f, tracker.det_log[f], step=max(end, f) + 1)
    return tracker.result()
