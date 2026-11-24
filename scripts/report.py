#!/usr/bin/env python3
"""Terminal comparison report: greedy vs windowed inference, per scenario.

Usage:
    python3 scripts/report.py                 # all scenarios, default seed
    python3 scripts/report.py --seed 23       # one alternative noise draw
    python3 scripts/report.py --json out.json # also dump the full comparison

The numbers come from ground truth (the tracker never sees it).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tracker.scenarios import SCENARIOS, scenario_catalog  # noqa: E402
from tracker.session import compare_scenario  # noqa: E402


def _line(label: str, ev: dict) -> str:
    ident = ev["identity"]
    frags = " ".join(f"{k}×{v}" for k, v in sorted(ident["fragmentation"].items()))
    return (
        f"  {label:8s} idsw={ident['idsw']:<2d} cont={ident['clean_ratio'] * 100:5.1f}% "
        f"ghosts={ev['ghost_count']:<2d} branch={ev['branching']['frames_with_alternatives']:<3d} "
        f"replay={len(ev['replays']):<2d} corr={len(ev['corrections']):<2d} "
        f"frag=[{frags}]"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=None, help="scenario noise seed override")
    ap.add_argument("--json", default=None, help="write the full comparison payload here")
    args = ap.parse_args()

    catalog = {c["key"]: c for c in scenario_catalog()}
    payload = {}
    for key in SCENARIOS:
        data = compare_scenario(key, seed=args.seed)
        payload[key] = data
        spec = catalog[key]
        print(f"\n{spec['name']} ({key}) — {spec['description']}")
        print(f"  {spec['notes']}")
        print(_line("greedy", data["greedy"]["evaluation"]))
        print(_line("window", data["window"]["evaluation"]))
        d = data["delta"]
        print(
            f"  delta    idsw={d['idsw']:+d} ghosts={d['ghosts']:+d} "
            f"branch_frames={d['branch_frames']:+d} replays={d['replays']:+d}"
        )
        w = data["window"]["overview"]
        print(
            f"  window   sealed_until={w['sealed_until']} "
            f"truncated_search={w['truncated_search']} optimal_claim={w['optimal_claim']}"
        )

    if args.json:
        Path(args.json).write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        print(f"\nwrote {args.json}")
    print(
        "\nNote: neither mode is globally optimal. Greedy never backtracks; window "
        "inference prunes hypotheses outside its beam/budget and records every "
        "discarded hypothesis with its score gap."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
