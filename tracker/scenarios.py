"""Synthetic scenarios: point detections only, no images.

Each scenario yields per-frame noisy 2-D detections with missed detections and
false alarms.  Ground truth is kept separately for evaluation; the tracker only
ever sees coordinates and detection ids.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

ARENA = (100.0, 100.0)


@dataclass
class TargetScript:
    truth_id: str
    p0: tuple[float, float]
    v: tuple[float, float]
    pd: float = 0.9
    visible_from: int = 0
    visible_to: int = 999
    blind: frozenset[int] = frozenset()
    turns: dict[int, tuple[float, float]] = field(default_factory=dict)


@dataclass
class ScenarioSpec:
    key: str
    name: str
    description: str
    targets: list[TargetScript]
    n_frames: int = 40
    sigma: float = 0.7
    clutter_rate: float = 0.3
    late_schedule: dict[int, int] = field(default_factory=dict)
    seed: int = 7
    notes: str = ""


def _pos(script: TargetScript, frame: int) -> tuple[float, float]:
    x, y = script.p0
    vx, vy = script.v
    for t in range(1, frame + 1):
        if t in script.turns:
            vx, vy = script.turns[t]
        x += vx
        y += vy
    return x, y


def _phase(truth_id: str, mod: int) -> float:
    # Deterministic per-target phase; avoid hash(), which is salted per process.
    return (sum(ord(c) for c in truth_id) * 1.7) % mod


def _poisson(rng: random.Random, lam: float, cap: int = 8) -> int:
    if lam <= 0:
        return 0
    p = math.exp(-lam)
    acc = p
    u = rng.random()
    k = 0
    while u > acc and k < cap:
        k += 1
        p *= lam / k
        acc += p
    return k


def _make(spec: ScenarioSpec) -> dict:
    rng = random.Random(spec.seed)
    w, h = ARENA
    frames: list[dict] = []
    truth_frames: list[dict] = []
    for f in range(spec.n_frames):
        targets: list[dict] = []
        dets: list[dict] = []
        for t in spec.targets:
            if not (t.visible_from <= f <= t.visible_to):
                continue
            tx, ty = _pos(t, f)
            tx += 0.8 * math.sin(0.21 * f + _phase(t.truth_id, 7))
            ty += 0.8 * math.cos(0.17 * f + _phase(t.truth_id, 5))
            targets.append({"truth_id": t.truth_id, "x": round(tx, 3), "y": round(ty, 3)})
            if f in t.blind or rng.random() > t.pd:
                continue
            dets.append({
                "x": tx + rng.gauss(0, spec.sigma),
                "y": ty + rng.gauss(0, spec.sigma),
                "truth_id": t.truth_id,
                "clutter": False,
            })
        for _ in range(_poisson(rng, spec.clutter_rate)):
            dets.append({
                "x": rng.uniform(2, w - 2),
                "y": rng.uniform(2, h - 2),
                "truth_id": None,
                "clutter": True,
            })
        dets.sort(key=lambda d: (round(d["y"], 3), round(d["x"], 3)))
        out = []
        for i, d in enumerate(dets):
            out.append({
                "id": f"f{f:02d}-d{i:02d}",
                "frame": f,
                "x": round(d["x"], 3),
                "y": round(d["y"], 3),
                "truth_id": d["truth_id"],
                "clutter": d["clutter"],
            })
        frames.append({"frame": f, "detections": out})
        truth_frames.append({"frame": f, "targets": targets})
    return {
        "key": spec.key,
        "name": spec.name,
        "description": spec.description,
        "notes": spec.notes,
        "n_frames": spec.n_frames,
        "sigma": spec.sigma,
        "clutter_rate": spec.clutter_rate,
        "late_schedule": {str(k): v for k, v in spec.late_schedule.items()},
        "arena": {"width": w, "height": h},
        "frames": frames,
        "truth": truth_frames,
        "seed": spec.seed,
    }


def _specs() -> dict[str, ScenarioSpec]:
    return {
        "crossing": ScenarioSpec(
            key="crossing",
            name="交叉相遇",
            description="两目标相向而行、在画面中部近距交叉；交会前后单帧最近邻几乎不可分。",
            targets=[
                TargetScript("A", (10.0, 44.5), (2.0, 0.16), pd=0.93),
                TargetScript("B", (90.0, 51.5), (-1.9, -0.16), pd=0.93),
            ],
            sigma=0.62,
            clutter_rate=0.45,
            notes="第 19-22 帧 A/B 相距 < 3 px，门控区域重叠；速度相近，交叉帧本身几乎没有信息量。",
        ),
        "vanish": ScenarioSpec(
            key="vanish",
            name="短暂消失",
            description="目标 A 连续 5 帧漏检，B 从 A 的预测位置附近经过后离开；检验滑行与复捕。",
            targets=[
                TargetScript(
                    "A", (14.0, 50.0), (1.7, 0.22), pd=0.95,
                    blind=frozenset({15, 16, 17, 18, 19}),
                ),
                TargetScript(
                    "B", (78.0, 66.0), (-1.05, -1.1), pd=0.95,
                    turns={20: (-0.4, 1.35)},
                ),
            ],
            sigma=0.62,
            clutter_rate=0.35,
            notes="A 第 15-19 帧无检测；B 第 20 帧转向，从 A 的预测走廊上方穿过。",
        ),
        "clutter": ScenarioSpec(
            key="clutter",
            name="高杂点",
            description="每帧约 4 个虚警、3 个真实目标，其中 A/B 交叉；检验门控与虚警轨迹抑制。",
            targets=[
                TargetScript("A", (8.0, 30.0), (2.05, 0.75), pd=0.93),
                TargetScript("B", (92.0, 55.0), (-2.0, -0.7), pd=0.93),
                TargetScript("C", (30.0, 92.0), (0.55, -1.65), pd=0.93),
            ],
            sigma=0.72,
            clutter_rate=4.2,
            notes="真实检测约 2.8/帧、虚警约 4.2/帧；关联必须依赖时间一致性而非单帧距离。",
        ),
        "combined": ScenarioSpec(
            key="combined",
            name="综合场景",
            description="四个目标、一次交叉、一次短暂消失、高杂点；第 24 帧按计划延迟 4 帧送达。",
            targets=[
                TargetScript("A", (8.0, 24.0), (1.95, 0.9), pd=0.92),
                TargetScript("B", (92.0, 58.0), (-1.85, -0.75), pd=0.92),
                TargetScript("C", (46.0, 10.0), (0.35, 1.6), pd=0.92),
                TargetScript(
                    "D", (70.0, 40.0), (-1.4, 0.9), pd=0.95,
                    blind=frozenset({18, 19, 20, 21}),
                ),
            ],
            sigma=0.72,
            clutter_rate=2.4,
            late_schedule={24: 4},
            notes="第 24 帧在第 28 步才送达，迟到时仍在窗口内（窗口 8 帧）→ 回溯重放；窗口外则只追加更正。",
        ),
    }


SCENARIOS = _specs()


def build_scenario(key: str, *, seed: int | None = None) -> dict:
    spec = SCENARIOS[key]
    if seed is not None:
        spec = ScenarioSpec(
            key=spec.key, name=spec.name, description=spec.description,
            targets=spec.targets, n_frames=spec.n_frames, sigma=spec.sigma,
            clutter_rate=spec.clutter_rate, late_schedule=spec.late_schedule,
            seed=seed, notes=spec.notes,
        )
    return _make(spec)


def scenario_catalog() -> list[dict]:
    return [
        {
            "key": s.key,
            "name": s.name,
            "description": s.description,
            "notes": s.notes,
            "n_frames": s.n_frames,
            "sigma": s.sigma,
            "clutter_rate": s.clutter_rate,
            "late_schedule": {str(k): v for k, v in s.late_schedule.items()},
        }
        for s in SCENARIOS.values()
    ]
