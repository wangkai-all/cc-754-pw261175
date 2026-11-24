"""Evaluation against ground truth: identity continuity, ghosts, branch recovery.

Ground truth (``truth_id`` on detections) is only read here; the trackers never
see it.  The figures are deliberately blunt and explainable - no MOTA-style
single number that would hide the identity failures this tool exists to expose.
"""

from __future__ import annotations

from collections import defaultdict


def _det_truth(scenario: dict) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for fr in scenario["frames"]:
        for d in fr["detections"]:
            out[d["id"]] = d["truth_id"]
    return out


def evaluate(result: dict, scenario: dict) -> dict:
    det_truth = _det_truth(scenario)
    frames = result["frames"]

    # Every committed (frame, track -> detection) assignment, with whether the
    # track was confirmed at that frame.  Tentative-track churn is reported
    # separately: a track that dies before confirmation is initiation noise,
    # not an identity switch.
    track_dets: dict[str, list[tuple[int, str]]] = defaultdict(list)
    confirmed_at: dict[tuple[int, str], bool] = {}
    for entry in frames:
        node = entry.get("committed")
        if not node:
            continue
        confirmed_ids = {t["id"] for t in node.get("tracks", []) if t.get("confirmed")}
        for tid, det in (node.get("track_det") or {}).items():
            if det is None:
                continue
            track_dets[tid].append((entry["frame"], det))
            confirmed_at[(entry["frame"], tid)] = tid in confirmed_ids

    per_track_truth: dict[str, dict[str, int]] = {}
    dominant: dict[str, str | None] = {}
    for tid, pairs in track_dets.items():
        counts: dict[str, int] = defaultdict(int)
        for _f, det in pairs:
            t = det_truth.get(det)
            if t is not None:
                counts[t] += 1
        per_track_truth[tid] = dict(counts)
        dominant[tid] = max(counts, key=counts.get) if counts else None

    # Identity switches: for each truth target, walk its committed detections in
    # time order and count changes of claiming *confirmed* track.
    truth_tracks: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for tid, pairs in track_dets.items():
        for f, det in pairs:
            t = det_truth.get(det)
            if t is not None and confirmed_at.get((f, tid)):
                truth_tracks[t].append((f, tid))
    tentative_claims: dict[str, int] = defaultdict(int)
    for tid, pairs in track_dets.items():
        for f, det in pairs:
            t = det_truth.get(det)
            if t is not None and not confirmed_at.get((f, tid)):
                tentative_claims[t] += 1
    switches: list[dict] = []
    per_truth: dict[str, dict] = {}
    total_clean = 0
    total_hits = 0
    for t, seq in truth_tracks.items():
        seq.sort()
        counts: dict[str, int] = defaultdict(int)
        for _f, tid in seq:
            counts[tid] += 1
        dom = max(counts, key=counts.get)
        clean = 0
        prev: tuple[int, str] | None = None
        switches_here = []
        for f, tid in seq:
            if tid == dom:
                clean += 1
            if prev is not None and tid != prev[1]:
                ev = {"truth": t, "frame": f, "from_track": prev[1],
                      "to_track": tid, "gap_frames": f - prev[0]}
                switches.append(ev)
                switches_here.append(ev)
            prev = (f, tid)
        total_clean += clean
        total_hits += len(seq)
        per_truth[t] = {
            "detections": len(seq),
            "tracks": sorted(counts),
            "dominant_track": dom,
            "continuity": round(clean / len(seq), 3) if seq else None,
            "switches": switches_here,
            "tentative_claims": tentative_claims.get(t, 0),
        }

    ghosts = [
        {"track": tid, "detections": len(pairs),
         "det_ids": [d for _f, d in pairs[:8]],
         "birth_frame": next(iter(sorted(f for f, _d in pairs)), None)}
        for tid, pairs in track_dets.items()
        if dominant.get(tid) is None and len(pairs) >= 2
    ]

    truth_frames = sum(len(fr["targets"]) for fr in scenario["truth"])
    fragmentation = {t: len({tid for _f, tid in seq})
                     for t, seq in truth_tracks.items()}

    # Branch recovery: on each frame with >1 alternative, is a pairing that
    # disagrees with the committed one still present?  Computed on committed
    # frame alternatives (which survive sealing as snapshots).
    branch_details = []
    rec_frames = 0
    rec_recovered = 0
    for entry in frames:
        alts = entry.get("alternatives") or []
        if len(alts) < 2:
            continue
        rec_frames += 1
        committed = entry.get("committed") or alts[0]
        base_sig = _signature(committed)
        variants = [a for a in alts if _signature(a) != base_sig]
        best = min(alts, key=lambda a: a["score"])
        if variants:
            rec_recovered += 1
        branch_details.append({
            "frame": entry["frame"],
            "alternatives": len(alts),
            "committed_is_map": committed.get("is_map", True),
            "committed_gap_to_map": round(committed["score"] - best["score"], 3),
            "identity_variant_recorded": bool(variants),
            "variant_keys": [a["key"] for a in variants],
            "variant_outcomes": {a["key"]: a.get("outcome") for a in variants},
        })

    return {
        "identity": {
            "idsw": len(switches),
            "switches": switches,
            "fragmentation": fragmentation,
            "per_truth": per_truth,
            "clean_ratio": round(total_clean / max(1, total_hits), 3),
            "detections_matched": total_hits,
            "truth_frames": truth_frames,
        },
        "ghosts": ghosts,
        "ghost_count": len(ghosts),
        "branching": {
            "frames_with_alternatives": rec_frames,
            "frames_with_recorded_variant": rec_recovered,
            "details": branch_details,
        },
        "sealed_until": result.get("sealed_until", -1),
        "late_frames": [e["frame"] for e in frames if e.get("late")],
        "replays": result.get("replays", []),
        "corrections": result.get("corrections", []),
        "locks": result.get("locks", []),
    }


def _signature(node: dict) -> tuple:
    """Assignment shape ignoring which track id was used.

    Compares the *pairing of detections to each other*: two hypotheses share a
    signature when every detection is grouped with the same partner detection.
    """
    assignment = node.get("assignment") or {}
    groups: list[tuple[str, ...]] = []
    by_track: dict[str, list[str]] = defaultdict(list)
    for det, tid in assignment.items():
        by_track[tid].append(det)
    for dets in by_track.values():
        groups.append(tuple(sorted(dets)))
    return tuple(sorted(groups))
