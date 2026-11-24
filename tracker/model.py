"""Data structures shared by the association engines."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .kalman import KalmanTrack


@dataclass
class Detection:
    id: str
    frame: int
    x: float
    y: float
    arrival_step: int = 0
    late: bool = False

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "frame": self.frame,
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "arrival_step": self.arrival_step,
            "late": self.late,
        }


@dataclass
class TrackState:
    id: str
    kf: KalmanTrack
    birth_frame: int
    birth_det: str
    hits: int = 0
    misses: int = 0
    age: int = 0
    confirmed: bool = False
    last_hit: int = -1
    det_history: list[tuple[int, str]] = field(default_factory=list)
    missed_frames: list[int] = field(default_factory=list)
    forced_frames: list[int] = field(default_factory=list)

    def clone(self) -> "TrackState":
        out = TrackState.__new__(TrackState)
        out.id = self.id
        out.kf = self.kf.clone()
        out.birth_frame = self.birth_frame
        out.birth_det = self.birth_det
        out.hits = self.hits
        out.misses = self.misses
        out.age = self.age
        out.confirmed = self.confirmed
        out.last_hit = self.last_hit
        out.det_history = list(self.det_history)
        out.missed_frames = list(self.missed_frames)
        out.forced_frames = list(self.forced_frames)
        return out

    def json_at(self, frame: int, det_id: str | None, *, forced: bool = False) -> dict:
        x, y = self.kf.position
        vx, vy = self.kf.velocity
        if det_id is not None:
            state = "confirmed" if self.confirmed else "tentative"
        else:
            state = "coasting"
        return {
            "id": self.id,
            "x": round(x, 3),
            "y": round(y, 3),
            "vx": round(vx, 3),
            "vy": round(vy, 3),
            "hits": self.hits,
            "misses": self.misses,
            "confirmed": self.confirmed,
            "birth_frame": self.birth_frame,
            "birth_det": self.birth_det,
            "age": self.age,
            "det": det_id,
            "state": state,
            "forced": forced,
            "det_history": [[f, d] for f, d in self.det_history],
        }


@dataclass
class Lock:
    id: str
    frame: int
    det_id: str
    track_id: str
    note: str = ""
    origin: str = "human"

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "frame": self.frame,
            "det_id": self.det_id,
            "track_id": self.track_id,
            "note": self.note,
            "origin": self.origin,
        }


@dataclass
class CandidatePair:
    track: str
    det: str
    d2: float
    euclid: float
    nll: float
    feasible: bool
    cost: float
    gated_by: str | None = None
    forced: bool = False
    note: str = ""


@dataclass
class FrameMatrix:
    frame: int
    tracks: list[str]
    detections: list[str]
    pairs: list[CandidatePair]
    miss_cost: dict[str, float]
    gate_chi2: float

    def by_pair(self) -> dict[tuple[str, str], CandidatePair]:
        return {(p.track, p.det): p for p in self.pairs}


def softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    mx = max(values)
    exps = [math.exp(v - mx) for v in values]
    total = sum(exps)
    return [e / total for e in exps]
