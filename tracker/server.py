"""Dependency-free HTTP server for the association workbench.

Uses only the Python standard library (http.server + json) so the workbench runs
anywhere Python 3.10+ is available.  The API is read/write against in-memory
session state; nothing is persisted.
"""

from __future__ import annotations

import json
import threading
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .scenarios import scenario_catalog
from .session import Session, compare_scenario

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"

_SESSIONS: dict[str, Session] = {}
_LOCK = threading.Lock()


def _new_session(params: dict) -> Session:
    key = params.get("scenario", "crossing")
    mode = params.get("mode", "window")
    if mode not in ("greedy", "window"):
        raise ValueError("mode must be greedy or window")
    seed = params.get("seed")
    seed = int(seed) if seed not in (None, "", "null") else None
    overrides = {}
    for name in ("gate_chi2", "window_len", "beam_width", "k_best",
                 "branch_margin", "prune_margin", "birth_limit",
                 "confirm_hits", "confirm_window", "init_gate_px", "max_misses"):
        if params.get(name) not in (None, ""):
            val = float(params[name])
            overrides[name] = int(val) if val.is_integer() and name not in (
                "branch_margin", "prune_margin", "gate_chi2", "init_gate_px") else val
    sid = params.get("session_id") or uuid.uuid4().hex[:10]
    sess = Session(key, mode, overrides, seed=seed, id=sid)
    return sess


def _get_session(sid: str) -> Session | None:
    return _SESSIONS.get(sid)


class Handler(BaseHTTPRequestHandler):
    server_version = "AssociationWorkbench/0.1"

    def log_message(self, fmt, *args):  # quieter default logging
        pass

    # -- helpers -----------------------------------------------------------
    def _send(self, payload, status=HTTPStatus.OK, content_type="application/json"):
        body = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, message, status=HTTPStatus.BAD_REQUEST):
        self._send({"ok": False, "error": str(message)}, status)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError(f"invalid JSON body: {exc}") from exc

    def _static(self, path: str):
        rel = path.lstrip("/") or "index.html"
        target = (WEB / rel).resolve()
        if not str(target).startswith(str(WEB.resolve())) or not target.is_file():
            self._error("not found", HTTPStatus.NOT_FOUND)
            return
        ctype = {
            ".html": "text/html", ".js": "text/javascript", ".css": "text/css",
            ".json": "application/json", ".svg": "image/svg+xml",
        }.get(target.suffix, "application/octet-stream")
        self._send(target.read_bytes(), content_type=ctype)

    # -- routing -----------------------------------------------------------
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        try:
            if path.startswith("/api/"):
                return self._api_get(path, query)
            return self._static(path)
        except Exception as exc:  # noqa: BLE001 - server boundary
            self._error(f"{type(exc).__name__}: {exc}", HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            body = self._body()
            return self._api_post(parsed.path, body)
        except ValueError as exc:
            self._error(exc)
        except Exception as exc:  # noqa: BLE001
            self._error(f"{type(exc).__name__}: {exc}", HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_DELETE(self):
        parsed = urlparse(self.path)
        parts = parsed.path.strip("/").split("/")
        try:
            if len(parts) == 4 and parts[:2] == ["api", "sessions"] and parts[3].startswith("locks"):
                sid = parts[2]
                body = self._body()
                sess = _get_session(sid)
                if sess is None:
                    return self._error("session not found", HTTPStatus.NOT_FOUND)
                with _LOCK:
                    out = sess.remove_lock(body.get("lock_id", ""))
                return self._send(out)
            self._error("unknown endpoint", HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            self._error(exc)

    def _api_get(self, path: str, query: dict):
        parts = path.strip("/").split("/")
        if parts == ["api", "scenarios"]:
            return self._send({"catalog": scenario_catalog()})
        if parts == ["api", "compare"]:
            key = query.get("scenario", "crossing")
            seed = query.get("seed")
            seed = int(seed) if seed not in (None, "", "null") else None
            with _LOCK:
                return self._send(compare_scenario(key, seed=seed))
        if len(parts) >= 3 and parts[1] == "sessions":
            sess = _get_session(parts[2])
            if sess is None:
                return self._error("session not found", HTTPStatus.NOT_FOUND)
            if len(parts) == 3:
                return self._send({"ok": True, "overview": sess.overview()})
            if parts[3] == "frame":
                frame = int(query.get("frame", 0))
                with _LOCK:
                    return self._send({"ok": True, "view": sess.frame_view(frame)})
            if parts[3] == "alternative":
                frame = int(query.get("frame", 0))
                key = query.get("key", "")
                with _LOCK:
                    node = sess.tracker.node_by_key(frame, key) if hasattr(sess.tracker, "node_by_key") else None
                    if node is None:
                        return self._send({"ok": True, "detail": None})
                    if hasattr(sess.tracker, "alternative_detail"):
                        return self._send({"ok": True,
                                           "detail": sess.tracker.alternative_detail(frame, key)})
                    return self._send({"ok": True, "detail": node.to_json()})
            if parts[3] == "tracks":
                with _LOCK:
                    return self._send({"ok": True, "history": sess.track_history()})
            if parts[3] == "export":
                with _LOCK:
                    return self._send({"ok": True, "data": sess.export()})
        return self._error("unknown endpoint", HTTPStatus.NOT_FOUND)

    def _api_post(self, path: str, body: dict):
        parts = path.strip("/").split("/")
        if parts == ["api", "sessions"]:
            sess = _new_session(body)
            with _LOCK:
                _SESSIONS[sess.id] = sess
            return self._send({"ok": True, "overview": sess.overview()})
        if len(parts) == 4 and parts[1] == "sessions":
            sess = _get_session(parts[2])
            if sess is None:
                return self._error("session not found", HTTPStatus.NOT_FOUND)
            if parts[3] == "locks":
                frame = int(body.get("frame", 0))
                det = str(body.get("det_id", ""))
                track = str(body.get("track_id", ""))
                with _LOCK:
                    res = sess.add_lock(frame, det, track, body.get("note", ""))
                status = HTTPStatus.OK if res.get("ok") else HTTPStatus.BAD_REQUEST
                return self._send(res, status)
            if parts[3] == "rebuild":
                with _LOCK:
                    sess.overrides.update(body.get("overrides") or {})
                    if body.get("mode"):
                        sess.mode = body["mode"]
                    sess.rebuild()
                return self._send({"ok": True, "overview": sess.overview()})
        return self._error("unknown endpoint", HTTPStatus.NOT_FOUND)


def serve(host: str = "127.0.0.1", port: int = 8765):
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"Association workbench on http://{host}:{port}/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
