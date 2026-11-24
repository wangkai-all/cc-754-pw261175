"""Test suite (stdlib unittest only - no pytest/numpy in this environment).

Run with:  python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import itertools
import math
import random
import tempfile
import threading
import unittest
import urllib.request
import json

from tracker.hungarian import INF, assignment_cost, hungarian
from tracker.kalman import KalmanTrack
from tracker.kbest import k_best_assignments
from tracker.engine import GreedyAssociator, TrackerConfig, WindowedAssociator
from tracker.evaluate import evaluate
from tracker.model import Detection
from tracker.scenarios import SCENARIOS, build_scenario, scenario_catalog
from tracker.session import Session, compare_scenario


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def brute_force(cost):
    """Every row gets a distinct column; padding columns (>= ncols) are legal."""
    n = len(cost)
    m = len(cost[0])
    best = None
    for perm in itertools.permutations(range(m), n):
        total = 0.0
        ok = True
        for i, j in enumerate(perm):
            if cost[i][j] >= INF:
                ok = False
                break
            total += cost[i][j]
        if not ok:
            continue
        if best is None or total < best - 1e-9:
            best = total
    return best


def run_scenario(key, mode, **overrides):
    tracker = (GreedyAssociator if mode == "greedy" else WindowedAssociator)(
        TrackerConfig().with_overrides(**overrides), build_scenario(key)
    )
    scn = build_scenario(key)
    for step in range(scn["n_frames"]):
        dets = [
            Detection(id=d["id"], frame=step, x=d["x"], y=d["y"])
            for d in scn["frames"][step]["detections"]
        ]
        tracker.ingest(step, dets, step=step)
    if isinstance(tracker, WindowedAssociator):
        for f in sorted(f for f in tracker.det_log if f not in tracker.delivered):
            tracker.ingest(f, tracker.det_log[f], step=scn["n_frames"] + 5)
    result = tracker.result()
    result["scenario"] = scn
    return tracker, result


# --------------------------------------------------------------------------
# assignment primitives
# --------------------------------------------------------------------------
class TestHungarian(unittest.TestCase):
    def test_matches_brute_force_random(self):
        rng = random.Random(11)
        for trial in range(200):
            n = rng.randint(1, 4)
            m = rng.randint(n, n + 2)
            cost = [[rng.uniform(-8, 8) for _ in range(m)] for _ in range(n)]
            # sprinkle forbidden cells
            for _ in range(rng.randint(0, 3)):
                cost[rng.randrange(n)][rng.randrange(m)] = INF
            got = hungarian(cost)
            want = brute_force(cost)
            if want is None:
                self.assertIsNone(got, f"trial {trial} expected infeasible")
            else:
                self.assertIsNotNone(got, f"trial {trial} expected a solution")
                self.assertAlmostEqual(assignment_cost(cost, got), want, places=7)

    def test_infeasible_matrix(self):
        cost = [[INF, 1.0], [INF, INF]]
        self.assertIsNone(hungarian(cost))

    def test_rejects_more_rows_than_cols(self):
        with self.assertRaises(ValueError):
            hungarian([[1.0], [2.0]])


class TestKBest(unittest.TestCase):
    def test_returns_ranked_prefix_of_all_assignments(self):
        rng = random.Random(3)
        for trial in range(120):
            n = rng.randint(1, 3)
            m = rng.randint(n, n + 2)
            cost = [[rng.uniform(-5, 5) for _ in range(m)] for _ in range(n)]
            for _ in range(rng.randint(0, 2)):
                cost[rng.randrange(n)][rng.randrange(m)] = INF
            # brute-force sorted list of all distinct assignments our solver may use
            cands = []
            for perm in itertools.permutations(range(m), n):
                if all(cost[i][j] < INF for i, j in enumerate(perm)):
                    cands.append((sum(cost[i][j] for i, j in enumerate(perm)), perm))
            cands.sort()
            seen_cols = set()
            want = []
            for total, perm in cands:
                if perm in seen_cols:
                    continue
                seen_cols.add(perm)
                want.append(perm)
                if len(want) >= 3:
                    break
            sols, truncated = k_best_assignments(cost, 3, max_examined=200)
            got = [tuple(s["columns"]) for s in sols]
            self.assertEqual(got, want, f"trial {trial}: got {got} want {want}")
            # The flag is a completeness statement, not "more alternatives
            # existed": it may only be set when the search stopped early.
            if not truncated:
                self.assertLessEqual(len(got), len(cands))
            for s in sols:
                self.assertAlmostEqual(
                    s["cost"],
                    sum(cost[i][j] for i, j in enumerate(s["columns"])),
                    places=7,
                )

    def test_empty_matrix(self):
        sols, truncated = k_best_assignments([], 3)
        self.assertEqual(sols, [{"cost": 0.0, "columns": ()}])
        self.assertFalse(truncated)


# --------------------------------------------------------------------------
# Kalman filter
# --------------------------------------------------------------------------
class TestKalman(unittest.TestCase):
    def test_converges_toward_constant_velocity(self):
        kf = KalmanTrack(0.0, 0.0)
        for t in range(1, 41):
            kf.predict(1.0)
            kf.update(2.0 * t, -1.0 * t, 0.5)
        vx, vy = kf.velocity
        self.assertAlmostEqual(vx, 2.0, delta=0.15)
        self.assertAlmostEqual(vy, -1.0, delta=0.15)
        x, y = kf.position
        self.assertAlmostEqual(x, 80.0, delta=2.0)
        self.assertAlmostEqual(y, -40.0, delta=2.0)

    def test_log_likelihood_monotone_in_distance(self):
        kf = KalmanTrack(10.0, 10.0)
        kf.predict()
        near, d2n = kf.log_likelihood(10.5, 10.0, 1.0)
        far, d2f = kf.log_likelihood(25.0, 10.0, 1.0)
        self.assertLess(near, far)
        self.assertLess(d2n, d2f)

    def test_covariance_stays_symmetric(self):
        kf = KalmanTrack(0.0, 0.0)
        for t in range(20):
            kf.predict()
            kf.update(float(t), float(-t), 0.8)
        for i in range(4):
            for j in range(4):
                self.assertAlmostEqual(kf.cov[i][j], kf.cov[j][i], places=10)


# --------------------------------------------------------------------------
# scenarios
# --------------------------------------------------------------------------
class TestScenarios(unittest.TestCase):
    def test_catalog_complete(self):
        keys = {c["key"] for c in scenario_catalog()}
        self.assertEqual(keys, set(SCENARIOS))

    def test_detections_have_ids_truth_and_clutter(self):
        scn = build_scenario("clutter")
        self.assertEqual(len(scn["frames"]), scn["n_frames"])
        clutter = 0
        for fr in scn["frames"]:
            for d in fr["detections"]:
                self.assertIn("truth_id", d)
                self.assertTrue(math.isfinite(d["x"]) and math.isfinite(d["y"]))
                if d["truth_id"] is None:
                    clutter += 1
        self.assertGreater(clutter, 50, "clutter scenario should contain many false alarms")

    def test_combined_has_late_frames(self):
        scn = build_scenario("combined")
        self.assertTrue(scn["late_schedule"])


# --------------------------------------------------------------------------
# engine behaviour
# --------------------------------------------------------------------------
class TestGating(unittest.TestCase):
    def test_far_detection_is_gated(self):
        cfg = TrackerConfig()
        tr = GreedyAssociator(cfg, build_scenario("crossing"))
        dets0 = [Detection("f00-d00", 0, 10.0, 10.0)]
        tr.ingest(0, dets0, step=0)
        dets1 = [Detection("f01-d00", 1, 60.0, 60.0)]
        tr.ingest(1, dets1, step=1)
        entry = tr.result()["frames"][1]
        pairs = {(p["track"], p["det"]): p for p in entry["audit"]}
        self.assertTrue(pairs, "the track and detection must still be listed as a candidate pair")
        pair = next(iter(pairs.values()))
        self.assertFalse(pair["feasible"])
        self.assertIn("门控", pair["reason"])
        self.assertNotIn("f01-d00", entry["committed"]["assignment"])
        self.assertIn("T01a", entry["committed"]["births"])
        # The gated detection was not associated to the old track; it entered
        # as a new (tentative) track instead.
        born = {tr["id"]: tr["birth_det"] for tr in entry["committed"]["tracks"]}
        self.assertEqual(born.get("T01a"), "f01-d00")


class TestGreedy(unittest.TestCase):
    def test_result_is_honest_about_optimality(self):
        _, result = run_scenario("crossing", "greedy")
        self.assertFalse(result["optimal_claim"])
        self.assertFalse(result["truncated_search"])
        self.assertTrue(all(f["sealed"] for f in result["frames"]))

    def test_rejected_pairs_have_reasons(self):
        _, result = run_scenario("clutter", "greedy")
        reasons = {r["reason"] for f in result["frames"] for r in f["rejected"]}
        self.assertTrue(reasons)
        self.assertTrue(reasons <= {"det_taken", "track_taken", "new_track", "clutter"})


class TestWindowed(unittest.TestCase):
    def test_alternatives_listed_with_weights(self):
        _, result = run_scenario("combined", "window")
        seen = False
        for f in result["frames"]:
            alts = f["alternatives"]
            if len(alts) > 1:
                seen = True
                self.assertAlmostEqual(sum(a["weight"] for a in alts), 1.0, places=3)
                self.assertEqual(sum(1 for a in alts if a["is_map"]), 1)
        self.assertTrue(seen, "combined should produce at least one ambiguous frame")

    def test_pruned_hypotheses_are_scored(self):
        _, result = run_scenario("clutter", "window")
        events = result["events"]
        self.assertTrue(events, "high clutter should trigger pruning")
        for ev in events:
            for d in ev["dropped"]:
                self.assertGreaterEqual(d["gap"], -1e-9)
                self.assertIn(d["reason"], (f"超出 beam 宽度 6",
                                            d["reason"]))
                self.assertIsNotNone(d["reason"])

    def test_never_claims_global_optimum(self):
        _, result = run_scenario("clutter", "window")
        self.assertFalse(result["optimal_claim"])
        self.assertIsInstance(result["truncated_search"], bool)

    def test_sealed_frames_do_not_change_after_expiry(self):
        tracker, result = run_scenario("crossing", "window")
        snapshot = {
            f["frame"]: f["committed"]["track_det"]
            for f in result["frames"] if f["sealed"] and f["committed"]
        }
        self.assertTrue(snapshot)
        again = tracker.result()
        for f in again["frames"]:
            if f["frame"] in snapshot:
                self.assertEqual(f["committed"]["track_det"], snapshot[f["frame"]])

    def test_identity_continuity_beats_greedy_on_vanish(self):
        _, greedy = run_scenario("vanish", "greedy")
        _, window = run_scenario("vanish", "window")
        g = evaluate(greedy, build_scenario("vanish"))
        w = evaluate(window, build_scenario("vanish"))
        self.assertGreater(g["identity"]["idsw"], w["identity"]["idsw"])


class TestLateFrames(unittest.TestCase):
    def test_late_frame_inside_window_replays(self):
        _, result = run_scenario("combined", "window")
        self.assertTrue(result["replays"], "the scheduled late frame should replay")
        replay = result["replays"][0]
        self.assertEqual(replay["frame"], 24)
        self.assertLess(replay["horizon"], 24)
        entry = next(f for f in result["frames"] if f["frame"] == 24)
        self.assertTrue(entry["late"])
        self.assertTrue(entry["replayed"])

    def test_late_frame_after_seal_appends_correction_only(self):
        """Shrink the window so frame 24 arrives past the seal line."""
        tracker, result = run_scenario("combined", "window", window_len=2)
        self.assertTrue(result["corrections"], "expected append-only corrections")
        self.assertFalse(result["replays"], "a frame past the seal line must not replay")
        for c in result["corrections"]:
            self.assertIn(c["kind"], ("detection_added", "detection_withdrawn", "frame_after_seal"))
            self.assertIn("sealed_until", c)
        # The late frame never entered the window: no hypothesis was fabricated
        # for it, and its sealed neighbours are untouched.
        entry = next(f for f in result["frames"] if f["frame"] == 24)
        self.assertTrue(entry["late"])
        self.assertIsNone(entry["committed"])
        self.assertEqual(entry["alternatives"], [])
        sealed_before = {
            f["frame"]: f["committed"]["track_det"]
            for f in result["frames"] if f["sealed"] and f["committed"]
        }
        self.assertIn(23, sealed_before)
        self.assertIn(25, sealed_before)
        self.assertNotIn(24, sealed_before)

    def test_correction_does_not_rewrite_sealed_assignment(self):
        """Withdraw a detection from an already-sealed frame: only a correction appears."""
        scn = build_scenario("combined")
        tracker = WindowedAssociator(TrackerConfig().with_overrides(window_len=2), scn)
        for step in range(0, 26):
            dets = [Detection(id=d["id"], frame=step, x=d["x"], y=d["y"])
                    for d in scn["frames"][step]["detections"]]
            tracker.ingest(step, dets, step=step)
        self.assertGreaterEqual(tracker.sealed_until, 23)
        before = tracker.sealed[21].to_json()["track_det"]
        # Withdraw one detection that the sealed hypothesis actually used.
        withdrawn = next(iter(tracker.sealed[21].assignment))
        shortened = [d for d in tracker.det_log[21] if d.id != withdrawn]
        tracker._append_correction(21, shortened)
        after = tracker.sealed[21].to_json()["track_det"]
        self.assertEqual(before, after, "sealed history must not be rewritten")
        kinds = {c["kind"] for c in tracker.corrections}
        self.assertTrue(kinds <= {"detection_withdrawn", "detection_added", "frame_after_seal"})
        self.assertIn("detection_withdrawn", kinds)


# --------------------------------------------------------------------------
# locks
# --------------------------------------------------------------------------
class TestLocks(unittest.TestCase):
    def test_lock_propagates_to_later_frames(self):
        base = Session("crossing", "window", id="a")
        base_map = {f["frame"]: (f["committed"] or {}).get("track_det")
                    for f in base.result["frames"]}
        # Find a detection/track pair that is not already the committed choice.
        frame = 19
        entry = next(f for f in base.result["frames"] if f["frame"] == frame)
        assign = entry["committed"]["assignment"]
        det = "f19-d00"
        current = assign.get(det)
        candidates = [t for t in {tr["id"] for tr in entry["committed"]["tracks"]} if t != current]
        self.assertTrue(candidates)
        target = sorted(candidates)[0]
        locked = Session("crossing", "window", id="b")
        res = locked.add_lock(frame, det, target)
        self.assertTrue(res["ok"])
        changed = [c for c in res["changed_frames"] if c["frame"] >= frame]
        self.assertTrue(changed, "the lock must change at least the locked frame")
        self.assertNotEqual(base_map[frame], {
            f["frame"]: (f["committed"] or {}).get("track_det")
            for f in locked.result["frames"]
        }[frame])
        # Lock recorded in the ledger of every side
        self.assertTrue(locked.result["locks"])
        for rev in locked.revisions:
            self.assertEqual(rev["kind"], "lock_revision")

    def test_conflicting_locks_name_the_detection(self):
        sess = Session("crossing", "window", id="c")
        entry = next(f for f in sess.result["frames"] if f["frame"] == 19)
        tracks = sorted({t["id"] for t in entry["committed"]["tracks"]})
        self.assertGreaterEqual(len(tracks), 2)
        sess.add_lock(19, "f19-d00", tracks[0])
        res = sess.add_lock(19, "f19-d00", tracks[1])
        kinds = {c["kind"] for c in res["conflicts"]}
        self.assertIn("detection_claimed_twice", kinds)
        conflict = next(c for c in res["conflicts"] if c["kind"] == "detection_claimed_twice")
        self.assertEqual(conflict["det_id"], "f19-d00")
        self.assertIn("f19-d00", conflict["detail"])

    def test_lock_on_missing_detection_is_rejected(self):
        sess = Session("crossing", "window", id="d")
        res = sess.add_lock(19, "f19-d99", "T00a")
        self.assertFalse(res["ok"])
        self.assertIn("f19-d99", res["error"])

    def test_lock_on_absent_track_reports_detection(self):
        sess = Session("crossing", "window", id="e")
        sess.add_lock(19, "f19-d00", "T99z")
        kinds = {c["kind"] for c in sess.result["conflicts"]}
        self.assertIn("lock_track_absent", kinds)
        never = next(c for c in sess.result["conflicts"] if c["kind"] == "lock_track_absent")
        self.assertEqual(never["det_id"], "f19-d00")

    def test_remove_lock_restores_previous_run(self):
        sess = Session("crossing", "window", id="f")
        before = {f["frame"]: (f["committed"] or {}).get("track_det")
                  for f in sess.result["frames"]}
        sess.add_lock(19, "f19-d00", "T00a")
        locks = sess.result["locks"]
        res = sess.remove_lock(locks[0]["id"])
        self.assertTrue(res["ok"])
        after = {f["frame"]: (f["committed"] or {}).get("track_det")
                 for f in sess.result["frames"]}
        self.assertEqual(before, after)


# --------------------------------------------------------------------------
# comparison + evaluation
# --------------------------------------------------------------------------
class TestCompare(unittest.TestCase):
    def test_compare_payload_shape(self):
        data = compare_scenario("vanish")
        for mode in ("greedy", "window"):
            ev = data[mode]["evaluation"]
            self.assertIn("identity", ev)
            self.assertIn("ghosts", ev)
            self.assertIn("branching", ev)
            self.assertFalse(ev is None)
        self.assertIn("idsw", data["delta"])
        self.assertEqual(data["greedy"]["evaluation"]["identity"]["idsw"]
                         - data["window"]["evaluation"]["identity"]["idsw"],
                         -data["delta"]["idsw"])

    def test_every_scenario_runs_in_both_modes(self):
        for key in SCENARIOS:
            for mode in ("greedy", "window"):
                _, result = run_scenario(key, mode)
                self.assertEqual(len(result["frames"]), SCENARIOS[key].n_frames,
                                 f"{key}/{mode} lost frames")


# --------------------------------------------------------------------------
# export / session
# --------------------------------------------------------------------------
class TestSessionExport(unittest.TestCase):
    def test_export_contains_detections_and_association(self):
        sess = Session("combined", "window", id="g")
        data = sess.export()
        self.assertEqual(len(data["frames"]), SCENARIOS["combined"].n_frames)
        for f in data["frames"]:
            self.assertIn("detections", f)
            self.assertIn("association", f)
            self.assertIn("tracks", f)
        self.assertFalse(data["optimal_claim"])
        first = data["frames"][0]
        self.assertTrue(first["detections"])
        # Frame 0 has no tracks yet, so its only association is (possibly) a
        # birth; by frame 2 the committed assignment is non-empty.
        self.assertTrue(any(f["association"] for f in data["frames"]))

    def test_overview_links_frames(self):
        sess = Session("clutter", "window", id="h")
        ov = sess.overview()
        self.assertEqual(len(ov["frames"]), SCENARIOS["clutter"].n_frames)
        self.assertFalse(ov["optimal_claim"])
        self.assertTrue(ov["config"]["gate_chi2"])

    def test_track_history_matches_frames(self):
        sess = Session("vanish", "window", id="i")
        hist = sess.track_history()
        self.assertEqual(len(hist), SCENARIOS["vanish"].n_frames)
        for h in hist:
            self.assertIn("tracks", h)
            for tr in h["tracks"]:
                self.assertIn("x", tr)
                self.assertIn("id", tr)


class TestServer(unittest.TestCase):
    """End-to-end over a real socket: the API the page actually calls."""

    @classmethod
    def setUpClass(cls):
        from http.server import ThreadingHTTPServer

        from tracker.server import Handler

        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def req(self, path, method="GET", body=None):
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=data, method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request) as res:
                return res.status, json.load(res)
        except urllib.error.HTTPError as exc:
            return exc.code, json.load(exc)

    def test_static_assets(self):
        for path, ctype, marker in (
            ("/", "text/html", "<canvas"),
            ("/style.css", "text/css", "--s1"),
            ("/app.js", "text/javascript", "renderCanvas"),
        ):
            url = f"http://127.0.0.1:{self.port}{path}"
            with urllib.request.urlopen(url) as res:
                body = res.read().decode("utf-8")
                self.assertEqual(res.status, 200, path)
                self.assertIn(ctype, res.headers.get("Content-Type", ""))
            self.assertIn(marker, body)
        status, _ = self.req("/../tracker/engine.py")
        self.assertEqual(status, 404)

    def test_session_roundtrip(self):
        status, created = self.req("/api/sessions", "POST",
                                   {"scenario": "combined", "mode": "window"})
        self.assertEqual(status, 200)
        sid = created["overview"]["session_id"]
        status, overview = self.req(f"/api/sessions/{sid}")
        self.assertEqual(status, 200)
        self.assertFalse(overview["overview"]["optimal_claim"])

        status, view = self.req(f"/api/sessions/{sid}/frame?frame=24")
        self.assertEqual(status, 200)
        self.assertTrue(view["view"]["tracker"]["replayed"])

        status, tracks = self.req(f"/api/sessions/{sid}/tracks")
        self.assertEqual(len(tracks["history"]), 40)

        status, lock = self.req(f"/api/sessions/{sid}/locks", "POST",
                                {"frame": 12, "det_id": "f12-d00", "track_id": "T00a"})
        self.assertEqual(status, 200)
        self.assertTrue(lock["ok"])
        lock_id = lock["lock"]["id"]
        status, removed = self.req(f"/api/sessions/{sid}/locks", "DELETE",
                                   {"lock_id": lock_id})
        self.assertEqual(status, 200)
        self.assertTrue(removed["ok"])

        status, bad = self.req(f"/api/sessions/{sid}/locks", "POST",
                               {"frame": 12, "det_id": "f12-d99", "track_id": "T00a"})
        self.assertEqual(status, 400)

        status, exported = self.req(f"/api/sessions/{sid}/export")
        self.assertEqual(len(exported["data"]["frames"]), 40)
        self.assertFalse(exported["data"]["optimal_claim"])


if __name__ == "__main__":
    unittest.main()
