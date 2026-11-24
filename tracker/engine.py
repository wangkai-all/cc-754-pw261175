"""Association engines: gating, k-best global assignment, hypothesis windows.

Two trackers share one cost model:

* :class:`GreedyAssociator` - nearest-first local matching.  It commits each
  frame immediately and cannot revise an earlier decision.
* :class:`WindowedAssociator` - keeps a tree of association hypotheses over a
  sliding window, branches on ambiguous frames, commits (seals) frames that
  leave the window, and reconciles late arrivals by replaying the open window.

All costs are negative log-likelihoods; **lower is better**.  The search is
bounded (k-best truncation, beam width, score window) and deliberately makes no
claim of global optimality: every prune is recorded with its score and gap.

Cost model (documented so every number in the UI is auditable)
---------------------------------------------------------------
The frame cost is the negative log-likelihood of the hypothesis relative to
explaining every detection as clutter:

* association ``(i, j)``: ``NLL(z; pred, S) - log Pd + log(clutter_density)``
* track misses its detection: ``-log(1 - Pd)``
* detection left unexplained (clutter): ``0`` - the reference cost
* detection starts a track: ``birth_cost`` (prior against track initiation)

``clutter_density`` is clutter points per px^2 per frame; for the 100x100 arena
with ~2 clutter points per frame it is about ``2e-4``.  Scoring association
against a *density* (not against a fixed miss penalty) is what makes an
uncertain, coasting track still prefer a plausible detection: the Gaussian
spread is what pays for the distance, not an arbitrary constant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .kbest import k_best_assignments
from .kalman import KalmanTrack
from .model import CandidatePair, Detection, FrameMatrix, Lock, TrackState

INF = float("inf")


@dataclass
class TrackerConfig:
    gate_chi2: float = 16.0
    meas_var: float = 0.9
    dt: float = 1.0
    process_pos: float = 0.12
    process_vel: float = 0.05
    pd: float = 0.95
    clutter_density: float = 2.0e-4   # clutter points per px^2 per frame
    miss_cost: float = 3.0            # -log(1 - Pd), Pd = 0.95
    birth_cost: float = 5.0           # relative log-prior against track birth
    gate_violation_cost: float = 25.0
    lock_bonus: float = -12.0
    confirm_hits: int = 5
    confirm_window: int = 10    # M-of-N confirmation: hits inside the last N frames
    init_window: int = 5        # unconfirmed tracks older than this are deleted
    init_gate_px: float = 8.0   # tentative tracks use a fixed initiation radius
    max_misses: int = 6
    birth_limit: int = 4
    window_len: int = 8
    beam_width: int = 6
    k_best: int = 4
    branch_margin: float = 3.0  # max frame-cost gap for spawning a branch
    prune_margin: float = 9.0   # max cumulative-cost gap for keeping a branch
    kbest_node_budget: int = 400

    def with_overrides(self, **overrides) -> "TrackerConfig":
        data = dict(self.__dict__)
        for k, v in overrides.items():
            if v is not None and k in data:
                data[k] = v
        return TrackerConfig(**data)

    def to_json(self) -> dict:
        return dict(self.__dict__)


def predict_track(tr: TrackState, cfg: TrackerConfig) -> TrackState:
    live = tr.clone()
    live.kf.predict(cfg.dt, cfg.process_pos, cfg.process_vel)
    live.age += 1
    return live


@dataclass
class Node:
    """One association hypothesis at one frame (a node of the hypothesis tree)."""

    key: str
    frame: int
    parent: "Node | None"
    score: float                       # cumulative cost (lower = better)
    frame_cost: float
    tracks: dict[str, TrackState]      # state *after* observing this frame
    ended: dict[str, dict]
    assignment: dict[str, str]         # detection id -> track id
    track_det: dict[str, str | None]   # track id -> detection id (None = missed)
    births: list[str]
    deaths: list[str]
    violations: list[dict]
    forced: list[str]
    matrix: FrameMatrix
    reason: str
    truncated: bool = False

    def history(self) -> list["Node"]:
        out: list[Node] = []
        node: Node | None = self
        while node is not None:
            out.append(node)
            node = node.parent
        out.reverse()
        return out

    def ancestor_at(self, frame: int) -> "Node | None":
        node: Node | None = self
        while node is not None and node.frame > frame:
            node = node.parent
        return node if node is not None and node.frame == frame else None

    def to_json(self) -> dict:
        tracks = []
        for tid in sorted(self.tracks):
            tr = self.tracks[tid]
            det = self.track_det.get(tid)
            tracks.append(tr.json_at(self.frame, det, forced=det in self.forced))
        return {
            "key": self.key,
            "frame": self.frame,
            "parent": self.parent.key if self.parent else None,
            "score": round(self.score, 3),
            "frame_cost": round(self.frame_cost, 3),
            "assignment": dict(self.assignment),
            "track_det": {k: v for k, v in self.track_det.items() if v is not None},
            "births": list(self.births),
            "deaths": list(self.deaths),
            "forced": list(self.forced),
            "violations": list(self.violations),
            "tracks": tracks,
            "reason": self.reason,
            "truncated": self.truncated,
        }


class _AssociatorBase:
    mode = "base"

    def __init__(self, cfg: TrackerConfig | None = None, scenario: dict | None = None):
        self.cfg = cfg or TrackerConfig()
        self.scenario = scenario or {}
        self.tracks: dict[str, TrackState] = {}
        self.ended: dict[str, dict] = {}
        self.locks: dict[str, Lock] = {}
        self._lock_seq = 0
        self.step = 0

    # -- identifiers --------------------------------------------------------
    def track_name(self, frame: int, det_id: str, existing: dict[str, TrackState]) -> str:
        """Deterministic id from the birth detection.

        Keying names on the birth detection (not a running counter) keeps the
        same identity label across hypotheses, which is what lets the UI compare
        branches and report identity changes honestly.
        """
        idx = 0
        if "-d" in det_id:
            try:
                idx = int(det_id.rsplit("-d", 1)[1])
            except ValueError:
                idx = 0
        tid = f"T{frame:02d}{chr(ord('a') + idx % 26)}"
        if tid in existing:
            k = 1
            while f"{tid}x{k}" in existing:
                k += 1
            tid = f"{tid}x{k}"
        return tid

    # -- locks --------------------------------------------------------------
    def add_lock(self, frame: int, det_id: str, track_id: str, note: str = "") -> Lock:
        for lk in self.locks.values():
            if lk.frame == frame and lk.det_id == det_id and lk.track_id == track_id:
                return lk
        self._lock_seq += 1
        lk = Lock(id=f"L{self._lock_seq:02d}", frame=frame, det_id=det_id,
                  track_id=track_id, note=note)
        self.locks[lk.id] = lk
        return lk

    def remove_lock(self, lock_id: str) -> bool:
        return self.locks.pop(lock_id, None) is not None

    def locks_for_frame(self, frame: int) -> list[Lock]:
        return [lk for lk in self.locks.values() if lk.frame == frame]

    def lock_conflicts(self, frame: int, dets: list[Detection]) -> list[dict]:
        """Locks on this frame that cannot all hold, naming the detection(s)."""
        det_ids = {d.id for d in dets}
        conflicts: list[dict] = []
        by_det: dict[str, list[Lock]] = {}
        by_track: dict[str, list[Lock]] = {}
        for lk in self.locks_for_frame(frame):
            if lk.det_id not in det_ids:
                conflicts.append({
                    "kind": "detection_absent", "frame": frame,
                    "det_id": lk.det_id, "track_id": lk.track_id,
                    "locks": [lk.to_json()],
                    "detail": f"锁定 {lk.id} 引用的检测点 {lk.det_id} 不在第 {frame} 帧",
                })
                continue
            by_det.setdefault(lk.det_id, []).append(lk)
            by_track.setdefault(lk.track_id, []).append(lk)
        for det_id, group in sorted(by_det.items()):
            tids = sorted({lk.track_id for lk in group})
            if len(tids) > 1:
                conflicts.append({
                    "kind": "detection_claimed_twice", "frame": frame,
                    "det_id": det_id, "track_id": None,
                    "locks": [lk.to_json() for lk in group],
                    "detail": f"检测点 {det_id} 被锁定给多个轨迹: {', '.join(tids)}",
                })
        for tid, group in sorted(by_track.items()):
            dids = sorted({lk.det_id for lk in group})
            if len(dids) > 1:
                conflicts.append({
                    "kind": "track_claimed_twice", "frame": frame,
                    "det_id": None, "track_id": tid,
                    "locks": [lk.to_json() for lk in group],
                    "detail": f"轨迹 {tid} 被锁定到多个检测点: {', '.join(dids)}",
                })
        return conflicts

    def forced_map(self, frame: int, dets: list[Detection],
                   live_tracks: dict[str, TrackState]) -> tuple[dict[str, str], list[dict]]:
        """Satisfiable track->detection locks for one branch, plus why others are not.

        Conflicts are resolved deterministically (lowest lock id wins) so every
        branch applies the same override.
        """
        det_ids = {d.id for d in dets}
        chosen: dict[str, str] = {}
        used_det: set[str] = set()
        unmet: list[dict] = []
        for lk in sorted(self.locks_for_frame(frame), key=lambda x: x.id):
            if lk.det_id not in det_ids:
                continue  # reported by lock_conflicts
            if lk.track_id not in live_tracks:
                unmet.append({
                    "kind": "lock_track_absent", "frame": frame,
                    "det_id": lk.det_id, "track_id": lk.track_id,
                    "lock": lk.id,
                    "detail": (f"锁定 {lk.id} 要求把 {lk.det_id} 给轨迹 {lk.track_id}，"
                               "但该轨迹在此假设中不存在（分支已丢弃它）"),
                })
                continue
            if lk.det_id in used_det or lk.track_id in chosen:
                unmet.append({
                    "kind": "lock_shadowed", "frame": frame,
                    "det_id": lk.det_id, "track_id": lk.track_id,
                    "lock": lk.id,
                    "detail": (f"锁定 {lk.id} 与更早的锁定冲突（检测点或轨迹已被占用），"
                               "冲突已单独列出"),
                })
                continue
            chosen[lk.track_id] = lk.det_id
            used_det.add(lk.det_id)
        return chosen, unmet

    # -- gating -------------------------------------------------------------
    def build_matrix(self, frame: int, tracks: dict[str, TrackState],
                     dets: list[Detection], forced: dict[str, str]) -> FrameMatrix:
        cfg = self.cfg
        pairs: list[CandidatePair] = []
        costs = {tid: cfg.miss_cost for tid in tracks}
        for tid in sorted(tracks):
            live = predict_track(tracks[tid], cfg)
            for d in dets:
                nll, d2 = live.kf.log_likelihood(d.x, d.y, cfg.meas_var)
                euclid = math.hypot(d.x - live.kf.mean[0], d.y - live.kf.mean[1])
                feasible = d2 <= cfg.gate_chi2
                if not tracks[tid].confirmed:
                    # A tentative track may only take nearby detections: this is
                    # what stops a clutter point from "poaching" a real target's
                    # measurement on its first association.
                    feasible = feasible and euclid <= cfg.init_gate_px
                is_forced = forced.get(tid) == d.id
                cost = (nll - math.log(cfg.pd)
                        + math.log(cfg.clutter_density))
                note = ""
                if not feasible:
                    cost += cfg.gate_violation_cost
                    note = (f"超出门控 Δ²={d2:.1f} > {cfg.gate_chi2:g}"
                            if d2 > cfg.gate_chi2
                            else f"超出初始化半径 {euclid:.1f}px > {cfg.init_gate_px:g}px（轨迹未确认）")
                if is_forced:
                    cost += cfg.lock_bonus
                    note = "人工锁定（代价已下调）"
                pairs.append(CandidatePair(
                    track=tid, det=d.id, d2=round(d2, 3), euclid=round(euclid, 3),
                    nll=round(nll, 4), feasible=feasible,
                    gated_by=None if feasible else f"chi2>{cfg.gate_chi2:g}",
                    cost=round(cost, 4), forced=is_forced, note=note,
                ))
        return FrameMatrix(
            frame=frame, tracks=sorted(tracks), detections=[d.id for d in dets],
            pairs=pairs, miss_cost=costs, gate_chi2=cfg.gate_chi2,
        )

    @staticmethod
    def assignment_cost_matrix(rows: list[str], cols: list[str], matrix: FrameMatrix):
        """Cost matrix for the k-best solver.

        Each track gets its **own** miss column (a single shared miss column
        would allow only one track to be unmatched).  Returns the matrix and the
        number of real detection columns.
        """
        lookup = matrix.by_pair()
        n, k = len(rows), len(cols)
        cost = [[0.0] * (k + n) for _ in range(n)]
        for i, tid in enumerate(rows):
            for j, did in enumerate(cols):
                cost[i][j] = lookup[(tid, did)].cost
            for q in range(n):
                cost[i][k + q] = matrix.miss_cost[tid] if q == i else INF
        return cost, k

    # -- transition ---------------------------------------------------------
    @staticmethod
    def _confirmable(tr: TrackState, frame: int, cfg: TrackerConfig) -> bool:
        """M-of-N confirmation: enough hits inside a recent window.

        A clutter point that starts a track usually collects no further hits, so
        it fails this test and is deleted by ``init_window``.
        """
        recent = sum(1 for f, _d in tr.det_history if f > frame - cfg.confirm_window)
        return recent >= cfg.confirm_hits

    def transition(self, tracks: dict[str, TrackState], ended: dict[str, dict],
                   dets: list[Detection], frame: int, assignment: dict[str, str],
                   forced: dict[str, str], births_wanted: list[str]):
        cfg = self.cfg
        det_by_id = {d.id: d for d in dets}
        out = {tid: predict_track(tr, cfg) for tid, tr in tracks.items()}
        det_to_track = dict(assignment)
        deaths: list[str] = []
        births: list[str] = []
        violations: list[dict] = []
        for tid in sorted(out):
            tr = out[tid]
            det_id = next((d for d, t in det_to_track.items() if t == tid), None)
            if det_id is None:
                tr.misses += 1
                tr.missed_frames.append(frame)
                if tr.misses > cfg.max_misses:
                    deaths.append(tid)
                elif not tr.confirmed and frame - tr.birth_frame >= cfg.init_window:
                    deaths.append(tid)
                continue
            d = det_by_id[det_id]
            _nll, d2 = tr.kf.log_likelihood(d.x, d.y, cfg.meas_var)
            if d2 > cfg.gate_chi2:
                violations.append({
                    "kind": "outside_gate", "frame": frame, "track": tid, "det": det_id,
                    "d2": round(d2, 3), "forced": forced.get(tid) == det_id,
                    "detail": ("人工锁定越过门控" if forced.get(tid) == det_id
                               else "该关联的 Δ² 高于门控阈值"),
                })
            tr.kf.update(d.x, d.y, cfg.meas_var)
            tr.hits += 1
            tr.misses = 0
            tr.last_hit = frame
            tr.det_history.append((frame, det_id))
            if not tr.confirmed and self._confirmable(tr, frame, cfg):
                tr.confirmed = True
        for tid in deaths:
            tr = out.pop(tid)
            ended[tid] = {
                "id": tid, "birth_frame": tr.birth_frame, "end_frame": frame,
                "cause": ("max_misses" if tr.misses > cfg.max_misses else "unconfirmed"),
                "hits": tr.hits, "last_det": tr.det_history[-1][1] if tr.det_history else None,
            }
        for det_id in births_wanted:
            d = det_by_id[det_id]
            tid = self.track_name(frame, det_id, out)
            tr = TrackState(id=tid, kf=KalmanTrack(d.x, d.y), birth_frame=frame,
                            birth_det=det_id, hits=1, last_hit=frame,
                            det_history=[(frame, det_id)])
            tr.confirmed = cfg.confirm_hits <= 1
            out[tid] = tr
            births.append(tid)
        return out, ended, births, deaths, violations

    # -- audit --------------------------------------------------------------
    def audit_pairs(self, node: Node) -> list[dict]:
        """Why each (track, detection) pair was or was not used in this node."""
        chosen = {(det, tid) for det, tid in node.assignment.items()}
        track_hit = {tid: det for tid, det in node.track_det.items() if det is not None}
        out = []
        for p in sorted(node.matrix.pairs, key=lambda x: x.cost):
            used = (p.det, p.track) in chosen
            if used:
                why = "已关联" + ("（人工锁定）" if p.forced else "")
            elif not p.feasible:
                why = f"超出门控 Δ²={p.d2:.1f} > {node.matrix.gate_chi2:g}；若被采用会记一次门控违反"
            elif track_hit.get(p.track):
                why = f"该轨迹选择了代价更低的检测点 {track_hit[p.track]}"
            elif p.det in node.assignment:
                why = f"该检测点被判给轨迹 {node.assignment[p.det]}（全局指派代价更优）"
            elif p.forced:
                why = "人工锁定未能生效（见冲突/说明）"
            else:
                why = "双方均未指派：轨迹滑行、检测点作为新轨迹或杂点"
            out.append({
                "track": p.track, "det": p.det, "d2": p.d2, "euclid": p.euclid,
                "cost": p.cost, "feasible": p.feasible, "forced": p.forced,
                "note": p.note, "chosen": used, "reason": why,
            })
        return out


class GreedyAssociator(_AssociatorBase):
    """Nearest-first local association; commits immediately, no backtracking."""

    mode = "greedy"

    def __init__(self, cfg: TrackerConfig | None = None, scenario: dict | None = None):
        super().__init__(cfg, scenario)
        self.frames: list[dict] = []
        self.det_log: dict[int, list[Detection]] = {}
        self.late_notes: list[dict] = []
        self.max_frame = -1

    def ingest(self, frame: int, dets: list[Detection], *, late: bool = False,
               step: int | None = None) -> dict:
        cfg = self.cfg
        self.step = step if step is not None else self.step + 1
        self.det_log[frame] = dets
        self.max_frame = max(self.max_frame, frame)
        forced, unmet = self.forced_map(frame, dets, self.tracks)
        conflicts = self.lock_conflicts(frame, dets) + unmet
        matrix = self.build_matrix(frame, self.tracks, dets, forced)

        assignment: dict[str, str] = dict((det, tid) for tid, det in forced.items())
        used_tracks: set[str] = set(forced)
        rejected: list[dict] = []
        candidates = sorted(
            (p for p in matrix.pairs
             if p.feasible and p.det not in assignment and p.track not in used_tracks),
            key=lambda p: (p.euclid, p.cost, p.det),
        )
        for p in candidates:
            if p.det in assignment:
                rejected.append({
                    "det": p.det, "track": p.track, "reason": "det_taken",
                    "detail": f"检测点已被更近的轨迹 {assignment[p.det]} 占用（Δ²={p.d2:.1f}）",
                })
                continue
            if p.track in used_tracks:
                rejected.append({
                    "det": p.det, "track": p.track, "reason": "track_taken",
                    "detail": "该轨迹已匹配到更近的检测点（贪心不比较全局代价）",
                })
                continue
            assignment[p.det] = p.track
            used_tracks.add(p.track)
        leftovers = [d for d in dets if d.id not in assignment]
        births_wanted = [d.id for d in leftovers[: cfg.birth_limit]]
        for d in dets:
            if d.id in births_wanted:
                rejected.append({"det": d.id, "track": None, "reason": "new_track",
                                 "detail": "无轨迹通过门控 → 立即新建轨迹（贪心无回溯）"})
            elif d.id not in assignment:
                rejected.append({"det": d.id, "track": None, "reason": "clutter",
                                 "detail": "未关联且超出每帧新建上限 → 当作杂点"})
        new_tracks, ended, births, deaths, violations = self.transition(
            self.tracks, self.ended, dets, frame, assignment, forced, births_wanted)
        self.tracks, self.ended = new_tracks, ended
        track_det = {tid: next((d for d, t in assignment.items() if t == tid), None)
                     for tid in new_tracks}
        node = Node(
            key=f"G{self.step:03d}", frame=frame, parent=None, score=0.0,
            frame_cost=self._frame_cost(matrix, assignment, leftovers, births_wanted),
            tracks=new_tracks, ended=ended, assignment=assignment, track_det=track_det,
            births=births, deaths=deaths, violations=violations,
            forced=sorted(forced.values()), matrix=matrix, reason="greedy",
        )
        if late:
            self.late_notes.append({
                "frame": frame, "step": self.step,
                "detail": ("贪心算法只能把迟到帧当作当前帧处理，"
                           "已提交的较早关联无法回溯修正"),
            })
        entry = {
            "frame": frame,
            "mode": "greedy",
            "sealed": True,
            "provisional": False,
            "late": late,
            "replayed": False,
            "branches": 1,
            "committed": node.to_json(),
            "alternatives": [dict(node.to_json(), weight=1.0, is_map=True)],
            "audit": self.audit_pairs(node),
            "rejected": rejected,
            "conflicts": conflicts,
            "pruned": [],
            "events": [],
            "corrections": [],
            "sealed_until": frame,
        }
        self.frames.append(entry)
        return entry

    def _frame_cost(self, matrix: FrameMatrix, assignment: dict[str, str],
                    leftovers: list[Detection], births: list[str]) -> float:
        cfg = self.cfg
        by_pair = matrix.by_pair()
        total = sum(by_pair[(tid, det)].cost for det, tid in assignment.items())
        chosen = set(assignment.values())
        total += cfg.miss_cost * sum(1 for tid in matrix.tracks if tid not in chosen)
        total += cfg.birth_cost * len(births)
        return total

    def result(self) -> dict:
        return {
            "mode": "greedy",
            "frames": self.frames,
            "locks": [lk.to_json() for lk in self.locks.values()],
            "events": [],
            "replays": [],
            "corrections": [],
            "conflicts": [],
            "late_notes": self.late_notes,
            "sealed_until": self.max_frame,
            "config": self.cfg.to_json(),
            "optimal_claim": False,
            "truncated_search": False,
        }


class WindowedAssociator(_AssociatorBase):
    """Beam-of-hypotheses tracker with a sliding inference window."""

    mode = "window"

    def __init__(self, cfg: TrackerConfig | None = None, scenario: dict | None = None):
        super().__init__(cfg, scenario)
        late = self.scenario.get("late_schedule") or {}
        self.late_schedule: dict[int, int] = {int(k): int(v) for k, v in late.items()}
        cfg0 = self.cfg
        self.root = Node(
            key="H0000", frame=-1, parent=None, score=0.0, frame_cost=0.0,
            tracks={}, ended={}, assignment={}, track_det={}, births=[], deaths=[],
            violations=[], forced=[], matrix=FrameMatrix(-1, [], [], [], {}, cfg0.gate_chi2),
            reason="root",
        )
        self.leaves: list[Node] = [self.root]
        self._seq = 0
        self.nodes_by_frame: dict[int, list[Node]] = {}
        self.archive: dict[int, list[Node]] = {}      # every node ever created
        self.seal_snapshots: dict[int, dict] = {}     # alternatives frozen at seal time
        self.det_log: dict[int, list[Detection]] = {}
        self.delivered: set[int] = set()
        self.max_delivered = -1
        self.sealed_until = -1
        self.sealed: dict[int, Node] = {}
        self.frame_reports: dict[int, dict] = {}
        self.prune_events: list[dict] = []
        self.replay_events: list[dict] = []
        self.corrections: list[dict] = []
        self.late_notes: list[dict] = []
        self.conflict_log: list[dict] = []
        self._truncated_seen = False
        self._revision = 0

    # -- delivery -----------------------------------------------------------
    def ingest(self, frame: int, dets: list[Detection], *, late: bool = False,
               step: int | None = None) -> list[dict]:
        """Deliver one frame (possibly late).  Returns reports for frames processed."""
        self.step = step if step is not None else self.step + 1
        if frame in self.delivered:
            raise ValueError(f"frame {frame} already delivered")
        self.det_log[frame] = dets
        reports: list[dict] = []
        arrival = frame + self.late_schedule.get(frame, 0)
        if arrival > self.step:
            self.late_notes.append({
                "frame": frame, "scheduled_step": arrival, "current_step": self.step,
                "detail": f"第 {frame} 帧按计划在第 {arrival} 步送达，先挂起（尚未进入窗口）",
            })
            return reports
        # Deliver this frame, plus any older frame whose scheduled arrival has come.
        pending = sorted(f for f in self.det_log if f not in self.delivered
                         and f + self.late_schedule.get(f, 0) <= self.step)
        for f in pending:
            reports.append(self._deliver(f, self.det_log[f],
                                         late=self.late_schedule.get(f, 0) > 0))
        return reports

    def _deliver(self, frame: int, dets: list[Detection], *, late: bool) -> dict:
        self.delivered.add(frame)
        self.max_delivered = max(self.max_delivered, frame)
        replayed = False
        if frame <= self.sealed_until:
            self._append_correction(frame, dets)
            entry = self._frame_entry(frame)
            entry["late"] = True
            entry["replayed"] = False
            entry["provisional"] = False
            entry["sealed_kind"] = "late_correction"
            entry["corrections"] = [c for c in self.corrections if c["frame"] == frame]
            return entry
        later_open = sorted(f for f in self.nodes_by_frame if f > frame)
        if later_open:
            # Still open but out of order: rewind, process the late frame, then
            # replay every later frame that is still inside the open window.
            replayed = True
            self._rewind(frame)
            self._expand(frame, dets, late=late, replayed=True)
            for f in later_open:
                if f in self.det_log and f not in self.sealed:
                    self._expand(f, self.det_log[f], late=True, replayed=True)
        else:
            self._expand(frame, dets, late=late, replayed=False)
        return self._frame_entry(frame)

    def _append_correction(self, frame: int, dets: list[Detection]) -> None:
        node = self.sealed.get(frame)
        new_ids = {d.id for d in dets}
        if node is not None:
            for det_id, tid in sorted(node.assignment.items()):
                if det_id not in new_ids:
                    self.corrections.append({
                        "kind": "detection_withdrawn", "frame": frame,
                        "det_id": det_id, "track_id": tid,
                        "sealed_until": self.sealed_until,
                        "detail": (f"第 {frame} 帧已封存（封存到第 {self.sealed_until} 帧）："
                                   f"轨迹 {tid} 引用的检测点 {det_id} 被更正撤回。"
                                   "只追加这条更正关系，不改写已封存轨迹"),
                    })
            for d in dets:
                if d.id not in node.assignment:
                    self.corrections.append({
                        "kind": "detection_added", "frame": frame,
                        "det_id": d.id, "track_id": None,
                        "sealed_until": self.sealed_until,
                        "detail": (f"第 {frame} 帧已封存：迟到检测点 {d.id} 只能登记为更正，"
                                   "不会再参与已封存轨迹的关联"),
                    })
        else:
            for d in dets:
                self.corrections.append({
                    "kind": "frame_after_seal", "frame": frame,
                    "det_id": d.id, "track_id": None,
                    "sealed_until": self.sealed_until,
                    "detail": (f"第 {frame} 帧（已越过封存线）迟到，只记录更正，不改写历史"),
                })

    def _replay_from(self, frame: int) -> None:
        horizon = frame - 1
        if horizon < 0:
            self.leaves = [self.root]
        else:
            ancestors: dict[str, Node] = {}
            for leaf in self.leaves:
                anc = leaf.ancestor_at(horizon)
                if anc is not None:
                    ancestors[anc.key] = anc
            if not ancestors:
                return
            best = min(a.score for a in ancestors.values())
            self.leaves = [a for a in ancestors.values()
                           if a.score <= best + self.cfg.prune_margin]
        for f in [f for f in self.nodes_by_frame if f >= frame]:
            self.nodes_by_frame.pop(f, None)
            self.frame_reports.pop(f, None)
        self.replay_events.append({
            "frame": frame, "step": self.step,
            "sealed_until": self.sealed_until,
            "horizon": horizon,
            "restored_hypotheses": [a.key for a in self.leaves],
            "detail": (f"第 {frame} 帧迟到，但仍在开放窗口内（封存到第 {self.sealed_until} 帧）："
                       f"回滚到第 {horizon} 帧并重放开放窗口"),
        })
        # Re-process already-delivered later frames against the restored leaves.
        later = sorted(f for f in self.delivered if f > frame)
        for f in later:
            self._expand(f, self.det_log[f], late=True, replayed=True)

    def _rewind(self, frame: int) -> None:
        """Roll the open window back to just before ``frame``; history stays put."""
        horizon = frame - 1
        if horizon < 0:
            self.leaves = [self.root]
        else:
            ancestors: dict[str, Node] = {}
            for leaf in self.leaves:
                anc = leaf.ancestor_at(horizon)
                if anc is not None:
                    ancestors[anc.key] = anc
            if not ancestors:
                return
            best = min(a.score for a in ancestors.values())
            self.leaves = [a for a in ancestors.values()
                           if a.score <= best + self.cfg.prune_margin]
        # Drop the open window's derived state; sealed snapshots stay intact.
        for f in [f for f in self.nodes_by_frame if f >= frame]:
            self.nodes_by_frame.pop(f, None)
        for f in [f for f in self.frame_reports if f >= frame]:
            self.frame_reports.pop(f, None)
        self.replay_events.append({
            "frame": frame, "step": self.step,
            "sealed_until": self.sealed_until,
            "horizon": horizon,
            "restored_hypotheses": [a.key for a in self.leaves],
            "detail": (f"第 {frame} 帧迟到且仍在开放窗口内（封存到第 {self.sealed_until} 帧）："
                       f"回滚到第 {horizon} 帧，插入迟到帧后重放后续开放帧"),
        })

    # -- expansion ----------------------------------------------------------
    def _expand(self, frame: int, dets: list[Detection], *, late: bool, replayed: bool) -> None:
        conflicts = self.lock_conflicts(frame, dets)
        children: list[Node] = []
        for leaf in self.leaves:
            forced, unmet = self.forced_map(frame, dets, leaf.tracks)
            conflicts = conflicts + [u for u in unmet if u not in conflicts]
            matrix = self.build_matrix(frame, leaf.tracks, dets, forced)
            children.extend(self._alternatives(leaf, matrix, dets, frame, forced, unmet))
        if conflicts:
            for c in conflicts:
                c["step"] = self.step
            self.conflict_log.extend(conflicts)
        if not children:
            self.nodes_by_frame[frame] = []
            return
        children.sort(key=lambda n: n.score)
        best = children[0].score
        survivors: list[Node] = []
        dropped: list[Node] = []
        for n in children:
            if len(survivors) >= self.cfg.beam_width:
                dropped.append(n)
            elif n.score > best + self.cfg.prune_margin:
                dropped.append(n)
            else:
                survivors.append(n)
        self.leaves = survivors
        self.nodes_by_frame[frame] = survivors
        self.archive.setdefault(frame, []).extend(survivors)
        if dropped:
            self.archive.setdefault(frame, []).extend(dropped)
            self.prune_events.append({
                "frame": frame,
                "best_score": round(best, 3),
                "dropped": [
                    {
                        "key": n.key, "score": round(n.score, 3),
                        "gap": round(n.score - best, 3),
                        "parent": n.parent.key if n.parent else None,
                        "reason": (f"超出 beam 宽度 {self.cfg.beam_width}"
                                   if len(survivors) >= self.cfg.beam_width
                                   else f"累计代价差 {n.score - best:.2f} > prune_margin {self.cfg.prune_margin:g}"),
                    }
                    for n in sorted(dropped, key=lambda x: x.score)[:24]
                ],
            })
        entry = self._frame_entry(frame)
        entry["late"] = late
        entry["replayed"] = replayed
        entry["conflicts"] = [c for c in self.conflict_log if c["frame"] == frame]
        entry["pruned"] = [d for ev in self.prune_events if ev["frame"] == frame
                           for d in ev["dropped"]]
        # Seal frames that have left the window.
        while self.sealed_until < frame - self.cfg.window_len:
            nxt = self.sealed_until + 1
            self._seal(nxt)

    def _alternatives(self, leaf: Node, matrix: FrameMatrix, dets: list[Detection],
                      frame: int, forced: dict[str, str], unmet: list[dict]) -> list[Node]:
        cfg = self.cfg
        rows = [t for t in matrix.tracks if t not in forced]
        used_dets = set(forced.values())
        cols = [d.id for d in dets if d.id not in used_dets]
        cost, ncols = self.assignment_cost_matrix(rows, cols, matrix)
        sols, truncated = k_best_assignments(cost, cfg.k_best,
                                             max_examined=cfg.kbest_node_budget)
        if truncated:
            self._truncated_seen = True
        if not sols and rows:
            return []
        if not rows:
            sols = [{"cost": 0.0, "columns": ()}]

        scored: list[dict] = []
        for sol in sols:
            track_det: dict[str, str | None] = {}
            assignment: dict[str, str] = {}
            for tid, det_id in forced.items():
                track_det[tid] = det_id
                assignment[det_id] = tid
            for i, tid in enumerate(rows):
                j = sol["columns"][i]
                if j < ncols:
                    track_det[tid] = cols[j]
                    assignment[cols[j]] = tid
                else:
                    track_det[tid] = None
            leftovers = [d for d in dets if d.id not in assignment]
            births = self._choose_births(leftovers, matrix, leaf.tracks)
            # Clutter costs are the zero reference; only births pay a prior.
            frame_cost = sol["cost"] + cfg.birth_cost * len(births)
            scored.append({"cost": frame_cost, "assignment": assignment,
                           "track_det": track_det, "births": births})
        scored.sort(key=lambda x: x["cost"])
        best_cost = scored[0]["cost"]
        out: list[Node] = []
        for idx, item in enumerate(scored):
            if idx > 0 and item["cost"] - best_cost > cfg.branch_margin:
                break
            tracks, ended, births, deaths, viol = self.transition(
                leaf.tracks, leaf.ended, dets, frame, item["assignment"], forced, item["births"])
            self._seq += 1
            reason = "extend" if idx == 0 else "branch"
            if truncated and idx == len(scored) - 1:
                reason += "+truncated"
            out.append(Node(
                key=f"H{self._seq:04d}", frame=frame, parent=leaf,
                score=leaf.score + item["cost"], frame_cost=item["cost"],
                tracks=tracks, ended=ended, assignment=item["assignment"],
                track_det=item["track_det"], births=births, deaths=deaths,
                violations=viol + [u for u in unmet if u["kind"] == "lock_track_absent"],
                forced=sorted(d for d in forced.values() if d in item["assignment"]),
                matrix=matrix, reason=reason, truncated=truncated,
            ))
        return out

    def _choose_births(self, leftovers: list[Detection], matrix: FrameMatrix,
                       tracks: dict[str, TrackState]) -> list[str]:
        """Pick which unassociated detections start tracks.

        Ordered by distance to the nearest prediction (farthest first): a real
        new target is usually farther from an existing track than clutter is
        from one.  This is a heuristic and is stated as such.
        """
        if not leftovers:
            return []
        limit = min(self.cfg.birth_limit, len(leftovers))
        scored = []
        for d in leftovers:
            nearest = min((math.hypot(d.x - tr.kf.mean[0], d.y - tr.kf.mean[1])
                           for tr in tracks.values()), default=INF)
            scored.append((nearest, d.id))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [det for _dist, det in scored[:limit]]

    # -- sealing ------------------------------------------------------------
    def _seal(self, frame: int) -> None:
        nodes = self.nodes_by_frame.get(frame) or []
        if not nodes:
            self.sealed_until = max(self.sealed_until, frame)
            return
        live_keys: set[str] = set()
        for leaf in self.leaves:
            node: Node | None = leaf
            while node is not None:
                live_keys.add(node.key)
                node = node.parent
        candidates = [n for n in nodes if n.key in live_keys] or nodes
        chosen = min(candidates, key=lambda n: n.score)

        kept: list[Node] = []
        dropped: list[Node] = []
        for leaf in self.leaves:
            anc = leaf.ancestor_at(frame)
            if anc is not None and anc.key == chosen.key:
                kept.append(leaf)
            else:
                dropped.append(leaf)
        self.leaves = kept
        if dropped:
            self.prune_events.append({
                "frame": frame,
                "best_score": round(chosen.score, 3),
                "reason": "sealed_commit",
                "dropped": [
                    {"key": leaf.key, "score": round(leaf.score, 3),
                     "gap": round(leaf.score - chosen.score, 3),
                     "parent": leaf.parent.key if leaf.parent else None,
                     "reason": (f"封存第 {frame} 帧时与 MAP 假设 {chosen.key} "
                                f"不一致（累计代价 {leaf.score:.2f} vs {chosen.score:.2f}）")}
                    for leaf in sorted(dropped, key=lambda n: n.score)[:24]
                ],
            })
        # Descendant nodes of dropped leaves are no longer live.
        for g in [g for g in self.nodes_by_frame if g > frame]:
            filtered = [n for n in self.nodes_by_frame[g]
                        if n.ancestor_at(frame) is not None
                        and n.ancestor_at(frame).key == chosen.key]
            self.nodes_by_frame[g] = filtered
        # Snapshot the alternatives that existed at seal time: after the commit
        # only one branch remains live, but the analysis station must still be
        # able to show what the competing hypotheses were and why they lost.
        alts = self._alternatives_json(frame)
        for i, a in enumerate(alts):
            a["outcome"] = "committed" if a["key"] == chosen.key else "pruned"
            a["outcome_reason"] = (
                "封存时成为 MAP 假设"
                if a["key"] == chosen.key
                else f"封存第 {frame} 帧时与 MAP 假设 {chosen.key} 不一致"
                     f"（累计代价差 {a['score'] - chosen.score:.2f}）"
            )
        self.seal_snapshots[frame] = {
            "chosen": chosen.key,
            "alternatives": alts,
            "opened_until": self.sealed_until,
        }
        self.sealed[frame] = chosen
        self.sealed_until = max(self.sealed_until, frame)
        self._refresh_entries(frame)

    def _refresh_entries(self, from_frame: int) -> None:
        for f in sorted(f for f in self.frame_reports if f >= from_frame):
            entry = self.frame_reports[f]
            if entry.get("sealed_kind") == "late_correction":
                # Arrived after its own frame was sealed away: it never entered
                # the window, so there is no live hypothesis to describe.  The
                # only output for this frame is its appended corrections.
                entry["sealed"] = False
                entry["provisional"] = False
                entry["committed"] = None
                entry["audit"] = []
                entry["alternatives"] = []
                entry["sealed_until"] = self.sealed_until
                continue
            if f in self.sealed:
                snap = self.seal_snapshots.get(f) or {}
                entry["sealed"] = True
                entry["sealed_kind"] = "sealed"
                entry["provisional"] = False
                entry["committed"] = self.sealed[f].to_json()
                entry["audit"] = self.audit_pairs(self.sealed[f])
                entry["alternatives"] = [dict(a) for a in snap.get("alternatives", [])]
            else:
                entry["sealed"] = False
                entry["sealed_kind"] = "open"
                entry["provisional"] = True
                entry["committed"] = None
                alts = self._alternatives_json(f)
                for a in alts:
                    a["outcome"] = "open"
                    a["outcome_reason"] = "窗口未封闭，仍在比较中"
                entry["alternatives"] = alts
                nodes = self.nodes_by_frame.get(f) or []
                entry["audit"] = self.audit_pairs(nodes[0]) if nodes else []
            entry["branches"] = len(self.nodes_by_frame.get(f, []))
            entry["sealed_until"] = self.sealed_until

    def _frame_entry(self, frame: int) -> dict:
        entry = self.frame_reports.get(frame)
        if entry is not None:
            self._refresh_entries(frame)
            return self.frame_reports[frame]
        entry = {
            "frame": frame, "mode": self.mode,
            "sealed": False, "provisional": True, "late": False, "replayed": False,
            "branches": len(self.nodes_by_frame.get(frame, [])),
            "committed": None, "alternatives": [], "audit": [], "rejected": [],
            "conflicts": [], "pruned": [], "events": [], "corrections": [],
            "sealed_until": self.sealed_until,
        }
        self.frame_reports[frame] = entry
        self._refresh_entries(frame)
        return self.frame_reports[frame]

    def _alternatives_json(self, frame: int) -> list[dict]:
        nodes = self.nodes_by_frame.get(frame) or []
        if not nodes:
            return []
        nodes = sorted(nodes, key=lambda n: n.score)
        best = nodes[0].score
        exps = [math.exp(-(n.score - best)) for n in nodes]
        total = sum(exps)
        out = []
        for n, e in zip(nodes, exps):
            item = n.to_json()
            item["weight"] = round(e / total, 4)
            item["is_map"] = n is nodes[0]
            out.append(item)
        return out

    def node_by_key(self, frame: int, key: str) -> Node | None:
        for n in self.archive.get(frame, []):
            if n.key == key:
                return n
        return None

    def alternative_detail(self, frame: int, key: str) -> dict | None:
        """Full detail for one alternative: tracks, audit, children, outcome."""
        node = self.node_by_key(frame, key)
        if node is None:
            return None
        snap = self.seal_snapshots.get(frame)
        outcome = "open"
        reason = "窗口未封闭"
        if snap is not None:
            outcome = "committed" if snap["chosen"] == key else "pruned"
            reason = next((a.get("outcome_reason", "") for a in snap["alternatives"]
                           if a["key"] == key), "")
        children = [n.key for n in self.nodes_by_frame.get(frame + 1, [])
                    if n.parent is not None and n.parent.key == key]
        item = node.to_json()
        item.update({
            "weight": next((a["weight"] for a in (snap or {}).get("alternatives", [])
                            if a["key"] == key), None),
            "is_map": outcome == "committed" or (
                frame not in self.sealed and self.nodes_by_frame.get(frame)
                and self.nodes_by_frame[frame][0].key == key),
            "outcome": outcome,
            "outcome_reason": reason,
            "audit": self.audit_pairs(node),
            "history": [
                {
                    "key": n.key, "frame": n.frame,
                    "assignment": dict(n.assignment),
                    "track_det": {k: v for k, v in n.track_det.items() if v is not None},
                    "score": round(n.score, 3),
                    "reason": n.reason,
                }
                for n in node.history()[1:]  # skip the synthetic root
            ],
            "child_keys": children,
        })
        return item

    # -- output -------------------------------------------------------------
    def result(self) -> dict:
        self._refresh_entries(0)
        frames = [self.frame_reports[f] for f in sorted(self.frame_reports)]
        return {
            "mode": self.mode,
            "frames": frames,
            "locks": [lk.to_json() for lk in self.locks.values()],
            "events": list(self.prune_events),
            "replays": list(self.replay_events),
            "corrections": list(self.corrections),
            "conflicts": list(self.conflict_log),
            "late_notes": self.late_notes,
            "sealed_until": self.sealed_until,
            "config": self.cfg.to_json(),
            "optimal_claim": False,
            "truncated_search": self._truncated_seen,
        }
