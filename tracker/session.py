"""Session layer: one scenario + one tracker + a lock set, rebuilt deterministically.

Everything a request needs is derived from ``(scenario, mode, config, locks,
frames_delivered)``.  Changing a lock rebuilds the run from frame 0, so a lock
provably reaches every later hypothesis; the rebuild is diffed against the
previous output to produce an append-only revision log instead of silently
rewriting history.

Locks are user input, not tracker output.  The tracker is the only judge of
whether a lock is satisfiable: it reports ``lock_shadowed`` / ``lock_track_absent``
/ ``detection_claimed_twice`` with the detection id when it is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .engine import GreedyAssociator, TrackerConfig, WindowedAssociator
from .evaluate import evaluate
from .model import Detection
from .scenarios import build_scenario, scenario_catalog


@dataclass
class Session:
    scenario_key: str
    mode: str = "window"
    overrides: dict = field(default_factory=dict)
    locks: list[dict] = field(default_factory=list)   # {frame, det_id, track_id, note}
    delivered_until: int | None = None                # None = all frames
    seed: int | None = None
    id: str = ""

    def __post_init__(self):
        self.scenario = build_scenario(self.scenario_key, seed=self.seed)
        self.rebuild()

    # -- configuration ------------------------------------------------------
    def config(self) -> TrackerConfig:
        return TrackerConfig().with_overrides(**self.overrides)

    def _make_tracker(self):
        cls = GreedyAssociator if self.mode == "greedy" else WindowedAssociator
        tracker = cls(self.config(), self.scenario)
        for lk in self.locks:
            tracker.add_lock(int(lk["frame"]), lk["det_id"], lk["track_id"],
                             lk.get("note", ""))
        return tracker

    # -- rebuild ------------------------------------------------------------
    def rebuild(self) -> None:
        previous = getattr(self, "result", None)
        tracker = self._make_tracker()
        n = self.scenario["n_frames"]
        last = n - 1 if self.delivered_until is None else min(self.delivered_until, n - 1)
        for step in range(0, last + 1):
            tracker.ingest(step, self.detections(step), step=step)
        if self.delivered_until is None and hasattr(tracker, "delivered"):
            # Flush frames held back by the late schedule.
            for f in sorted(f for f in tracker.det_log if f not in tracker.delivered):
                tracker.ingest(f, tracker.det_log[f], step=n + 5)
        self.tracker = tracker
        self.result = tracker.result()
        self.result["scenario"] = {
            "key": self.scenario["key"],
            "name": self.scenario["name"],
            "description": self.scenario["description"],
            "notes": self.scenario["notes"],
            "n_frames": n,
            "sigma": self.scenario["sigma"],
            "clutter_rate": self.scenario["clutter_rate"],
            "late_schedule": self.scenario["late_schedule"],
            "arena": self.scenario["arena"],
            "seed": self.scenario["seed"],
        }
        self.result["locks"] = [lk.to_json() for lk in tracker.locks.values()]
        self.evaluation = evaluate(self.result, self.scenario)
        self.changed_frames: list[dict] = []
        if previous is not None:
            self.changed_frames = self._diff(previous, self.result)
        self.revisions = getattr(self, "revisions", []) + [
            {**c, "kind": "lock_revision", "note": "锁变更后重建，此帧的已提交关联与上一次结果不同"}
            for c in self.changed_frames if c["was_sealed"]
        ]

    def _diff(self, old: dict, new: dict) -> list[dict]:
        old_by = {f["frame"]: f for f in old["frames"]}
        changed = []
        for f in new["frames"]:
            o = old_by.get(f["frame"])
            oc = (o or {}).get("committed") or {}
            nc = f.get("committed") or {}
            if oc.get("track_det") != nc.get("track_det"):
                changed.append({
                    "frame": f["frame"],
                    "was_sealed": bool((o or {}).get("sealed")),
                    "before": oc.get("track_det"),
                    "after": nc.get("track_det"),
                })
        return changed

    # -- delivery -----------------------------------------------------------
    def detections(self, frame: int) -> list[Detection]:
        late = {int(k) for k in (self.scenario.get("late_schedule") or {})}
        return [
            Detection(id=d["id"], frame=frame, x=d["x"], y=d["y"], late=frame in late)
            for d in self.scenario["frames"][frame]["detections"]
        ]

    def pending_frames(self) -> list[int]:
        delivered = getattr(self.tracker, "delivered", set())
        return sorted(f for f in getattr(self.tracker, "det_log", {}) if f not in delivered)

    # -- locks --------------------------------------------------------------
    def add_lock(self, frame: int, det_id: str, track_id: str, note: str = "") -> dict:
        if frame < 0 or frame >= self.scenario["n_frames"]:
            return {"ok": False, "error": f"帧 {frame} 超出范围"}
        det_ids = {d["id"] for d in self.scenario["frames"][frame]["detections"]}
        if det_id not in det_ids:
            return {"ok": False, "error": f"检测点 {det_id} 不在第 {frame} 帧"}
        for lk in self.locks:
            if lk["frame"] == frame and lk["det_id"] == det_id and lk["track_id"] == track_id:
                return {"ok": True, "duplicate": True, "lock": lk,
                        "conflicts": self._conflicts_for(frame)}
        self.locks.append({"frame": frame, "det_id": det_id,
                           "track_id": track_id, "note": note})
        self.rebuild()
        return {
            "ok": True,
            "lock": next(lk for lk in self.result["locks"]
                         if lk["frame"] == frame and lk["det_id"] == det_id
                         and lk["track_id"] == track_id),
            "conflicts": self._conflicts_for(frame),
            "changed_frames": self.changed_frames,
        }

    def remove_lock(self, lock_id: str) -> dict:
        match = next((lk for lk in self.locks
                      if f"L{self.locks.index(lk) + 1:02d}" == lock_id), None)
        if match is None:
            return {"ok": False, "error": f"锁定 {lock_id} 不存在"}
        frame = match["frame"]
        self.locks.remove(match)
        self.rebuild()
        return {"ok": True, "changed_frames": self.changed_frames,
                "conflicts": self._conflicts_for(frame)}

    def clear_locks(self) -> None:
        self.locks = []
        self.rebuild()

    def _conflicts_for(self, frame: int) -> list[dict]:
        return [c for c in self.result.get("conflicts", []) if c.get("frame") == frame]

    # -- views --------------------------------------------------------------
    def frame_view(self, frame: int) -> dict:
        entry = next((f for f in self.result["frames"] if f["frame"] == frame), None)
        truth = self.scenario["truth"][frame]["targets"] if frame < len(self.scenario["truth"]) else []
        dets = self.scenario["frames"][frame]["detections"] if frame < len(self.scenario["frames"]) else []
        conflicts = [c for c in self.result.get("conflicts", []) if c.get("frame") == frame]
        corrections = [c for c in self.result.get("corrections", []) if c.get("frame") == frame]
        pruned = [d for ev in self.result.get("events", []) if ev.get("frame") == frame
                  for d in ev.get("dropped", [])]
        return {
            "frame": frame,
            "detections": dets,
            "truth": truth,
            "tracker": entry,
            "locks": [lk for lk in self.result.get("locks", []) if lk["frame"] == frame],
            "conflicts": conflicts,
            "corrections": corrections,
            "pruned": pruned,
            "replays": [r for r in self.result.get("replays", []) if r.get("frame") == frame],
            "sealed_until": self.result.get("sealed_until", -1),
            "changed": [c for c in self.changed_frames if c["frame"] == frame],
            "pending": frame in self.pending_frames(),
            "late_schedule": self.scenario["late_schedule"],
        }

    def overview(self) -> dict:
        return {
            "session_id": self.id,
            "scenario": self.result["scenario"],
            "catalog": scenario_catalog(),
            "mode": self.mode,
            "config": self.result["config"],
            "locks": self.result.get("locks", []),
            "sealed_until": self.result.get("sealed_until", -1),
            "replays": self.result.get("replays", []),
            "corrections": self.result.get("corrections", []),
            "conflicts": self.result.get("conflicts", []),
            "prune_events": self.result.get("events", []),
            "late_notes": self.result.get("late_notes", []),
            "revisions": getattr(self, "revisions", []),
            "evaluation": self.evaluation,
            "pending_frames": self.pending_frames(),
            "optimal_claim": False,
            "truncated_search": self.result.get("truncated_search", False),
            "frames": [
                {
                    "frame": f["frame"],
                    "sealed": f.get("sealed"),
                    "sealed_kind": f.get("sealed_kind"),
                    "provisional": f.get("provisional"),
                    "branches": f.get("branches"),
                    "late": f.get("late"),
                    "replayed": f.get("replayed"),
                    "committed": (f.get("committed") or {}).get("track_det"),
                    "alternatives": len(f.get("alternatives") or []),
                    "n_detections": len(self.scenario["frames"][f["frame"]]["detections"]),
                }
                for f in self.result["frames"]
            ],
        }

    def track_history(self) -> list[dict]:
        """Per-frame committed (or MAP-provisional) track positions.

        Small enough to hand to the browser in one request so the canvas can
        draw identity polylines without fetching every detection frame.
        """
        out = []
        for entry in self.result["frames"]:
            node = entry.get("committed")
            if node is None:
                alts = entry.get("alternatives") or []
                node = alts[0] if alts else None
            tracks = []
            for tr in (node or {}).get("tracks", []):
                tracks.append({
                    "id": tr["id"], "x": tr["x"], "y": tr["y"],
                    "det": tr.get("det"), "state": tr.get("state"),
                    "confirmed": tr.get("confirmed"),
                    "forced": tr.get("forced"),
                })
            out.append({
                "frame": entry["frame"],
                "sealed": bool(entry.get("sealed")),
                "tracks": tracks,
            })
        return out

    def export(self) -> dict:
        """Original detections plus association results, frame by frame."""
        frames = []
        by_frame = {f["frame"]: f for f in self.result["frames"]}
        for fr in self.scenario["frames"]:
            f = fr["frame"]
            entry = by_frame.get(f, {})
            node = entry.get("committed") or {}
            frames.append({
                "frame": f,
                "detections": fr["detections"],
                "truth": self.scenario["truth"][f]["targets"],
                "association": node.get("track_det", {}),
                "assignment": node.get("assignment", {}),
                "tracks": node.get("tracks", []),
                "sealed": entry.get("sealed", False),
                "late": entry.get("late", False),
                "replayed": entry.get("replayed", False),
                "alternatives": [
                    {"key": a["key"], "score": a["score"], "weight": a.get("weight"),
                     "outcome": a.get("outcome"), "track_det": a.get("track_det")}
                    for a in (entry.get("alternatives") or [])
                ],
                "conflicts": [c for c in self.result.get("conflicts", [])
                              if c.get("frame") == f],
                "corrections": [c for c in self.result.get("corrections", [])
                                if c.get("frame") == f],
            })
        return {
            "scenario": self.result["scenario"],
            "mode": self.mode,
            "config": self.result["config"],
            "locks": self.result.get("locks", []),
            "revisions": getattr(self, "revisions", []),
            "replays": self.result.get("replays", []),
            "frames": frames,
            "evaluation": self.evaluation,
            "optimal_claim": False,
        }


def compare_scenario(scenario_key: str, overrides: dict | None = None,
                     seed: int | None = None) -> dict:
    """Run greedy and window inference on one scenario and compare metrics."""
    overrides = overrides or {}
    out = {}
    for mode in ("greedy", "window"):
        s = Session(scenario_key, mode, dict(overrides), seed=seed)
        out[mode] = {
            "evaluation": s.evaluation,
            "config": s.result["config"],
            "overview": s.overview(),
        }
    g = out["greedy"]["evaluation"]
    w = out["window"]["evaluation"]
    out["delta"] = {
        "idsw": w["identity"]["idsw"] - g["identity"]["idsw"],
        "clean_ratio": round(w["identity"]["clean_ratio"] - g["identity"]["clean_ratio"], 3),
        "ghosts": w["ghost_count"] - g["ghost_count"],
        "branch_frames": (w["branching"]["frames_with_alternatives"]
                          - g["branching"]["frames_with_alternatives"]),
        "replays": len(w["replays"]) - len(g["replays"]),
        "corrections": len(w["corrections"]) - len(g["corrections"]),
    }
    return out
