"""A local, append-only labeling server for disputed mineral components (Python stdlib only).

Derived from the hydration cosmic-ray labeler (``cr_labeler``, 2026-10-21):
same persistence, freeze, binding and HTTP rules. Differences: four neutral classes, a mineral-window
payload with two fits (A and B) and the disputed band, and rows shown in manifest order (the manifest
builder fixes the order with its seed).
"""

import argparse
from datetime import datetime, timezone
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import secrets
import socket
import tempfile
import threading


BODY_LIMIT = 16 * 1024
COMPONENT_FIELDS = ("center_cm1", "amplitude", "fwhm_cm1", "snr")
CSP = ("default-src 'self'; img-src 'self' data:; object-src 'none'; "
       "base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
LABELS = {"present", "absent", "spike", "unsure"}
STATIC_DIR = Path(__file__).with_name("static")


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True) + "\n").encode()


def _atomic_write(path, data):
    """Publish a complete file and persist both its content and directory entry."""
    fd, temporary = tempfile.mkstemp(prefix=".labeler-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _number(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value))


def _trace(value):
    """Copy only numeric coordinates; never retain arbitrary nested metadata."""
    if not isinstance(value, dict):
        raise ValueError("Invalid spectrum")
    x, y = value.get("x"), value.get("y")
    if (not isinstance(x, list) or not isinstance(y, list) or len(x) < 2
            or len(x) != len(y) or not all(_number(v) for v in x + y)
            or any(a >= b for a, b in zip(x, x[1:]))):
        raise ValueError("Invalid spectrum")
    return {"x": list(x), "y": list(y)}


def _components(value):
    """Copy only the numeric component fields and the kept flag of one fit."""
    if not isinstance(value, list) or len(value) > 12:
        raise ValueError("Invalid fit")
    out = []
    for comp in value:
        if (not isinstance(comp, dict) or not isinstance(comp.get("kept"), bool)
                or not all(_number(comp.get(k)) for k in COMPONENT_FIELDS)
                or comp["fwhm_cm1"] <= 0):
            raise ValueError("Invalid fit component")
        out.append({k: comp[k] for k in COMPONENT_FIELDS} | {"kept": comp["kept"]})
    return out


def _public_item(item):
    """The served view of a manifest item: spectrum, band, both fits, review flags; nothing else."""
    band, fits, flags = item.get("band"), item.get("fits"), item.get("flags", {})
    if (not isinstance(band, dict) or not _number(band.get("center_cm1"))
            or not _number(band.get("fwhm_cm1")) or band["fwhm_cm1"] <= 0):
        raise ValueError("Invalid band")
    if not isinstance(fits, dict) or set(fits) != {"A", "B"}:
        raise ValueError("Invalid fits")
    mask = flags.get("mask_px_in_band", 0) if isinstance(flags, dict) else None
    if isinstance(mask, bool) or not isinstance(mask, int) or mask < 0:
        raise ValueError("Invalid flags")
    window = _trace(item.get("window"))
    models = item.get("models")
    if models is not None:
        if (not isinstance(models, dict) or set(models) != {"A", "B"}
                or not all(isinstance(m, list) and len(m) == len(window["x"])
                           and all(_number(v) for v in m) for m in models.values())):
            raise ValueError("Invalid models")
        models = {"A": list(models["A"]), "B": list(models["B"])}
    return {"window": window, "models": models,
            "band": {"center_cm1": band["center_cm1"], "fwhm_cm1": band["fwhm_cm1"]},
            "fits": {"A": _components(fits["A"]), "B": _components(fits["B"])},
            "flags": {"mask_px_in_band": mask}}


class Session:
    """Keep public payloads separate from the private final-export mapping."""

    def __init__(self, manifest, out, labeler):
        if not isinstance(labeler, str) or not labeler.strip():
            raise ValueError("A labeler name is required")
        raw = Path(manifest).read_bytes()
        document = json.loads(raw)
        if (not isinstance(document, dict) or not isinstance(document.get("header"), dict)
                or not isinstance(document["header"].get("public_check"), str)
                or not document["header"]["public_check"].strip()
                or not isinstance(document.get("items"), list)):
            raise ValueError("Invalid manifest or missing public check")
        self.manifest_sha256 = hashlib.sha256(raw).hexdigest()
        self.labeler = labeler
        self.out = Path(out)
        self.lock = threading.RLock()
        self.out.mkdir(parents=True, exist_ok=True, mode=0o700)
        key_path = self.out / "key"
        session_path = self.out / "session.json"
        journal = self.out / "labels.jsonl"
        final_path = self.out / "labels-final.json"
        if not key_path.exists():
            if any(p.exists() for p in (session_path, journal, final_path)):
                raise ValueError("Existing session has no key")
            fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(secrets.token_bytes(32))
                stream.flush()
                os.fsync(stream.fileno())
        os.chmod(key_path, 0o600)
        key = key_path.read_bytes()
        if len(key) != 32:
            raise ValueError("Invalid session key")
        public = {}
        self._units = {}
        for item in document["items"]:
            if (not isinstance(item, dict) or not isinstance(item.get("unit_id"), str)
                    or not item["unit_id"]):
                raise ValueError("Invalid manifest item")
            opaque = hmac.new(key, item["unit_id"].encode(), hashlib.sha256).hexdigest()[:16]
            if opaque in public:
                raise ValueError("Duplicate item or opaque ID collision")
            self._units[opaque] = item["unit_id"]
            public[opaque] = {"id": opaque} | _public_item(item)
        order = list(public)  # manifest order; the builder fixes it with its seed
        metadata = {"manifest_sha256": self.manifest_sha256, "labeler": labeler, "order": order}
        if session_path.exists():
            if json.loads(session_path.read_bytes()) != metadata:
                raise ValueError("Manifest or labeler differs from the saved session")
        else:
            if journal.exists() or final_path.exists():
                raise ValueError("Existing session has no saved order")
            _atomic_write(session_path, _json_bytes(metadata))
        self._items = [dict(public[opaque], k=k, n_total=len(order)) for k, opaque in enumerate(order)]
        self._latest = {}
        if journal.exists():
            history = journal.read_bytes()
            if history and not history.endswith(b"\n"):
                raise ValueError("Incomplete label journal; preserve it for recovery")
            for line in history.splitlines():
                entry = json.loads(line)
                if (not isinstance(entry, dict) or entry.get("id") not in self._units
                        or entry.get("label") not in LABELS or entry.get("labeler") != labeler
                        or not isinstance(entry.get("time"), str)):
                    raise ValueError("Invalid label journal")
                datetime.fromisoformat(entry["time"].replace("Z", "+00:00"))
                self._latest[entry["id"]] = entry["label"]
        self.frozen = final_path.exists()
        if self.frozen:
            final = final_path.read_bytes()
            if json.loads(final) != self._final_document():
                raise ValueError("Final labels differ from the saved session")
            digest_path = self.out / "labels-final.sha256"
            digest = hashlib.sha256(final).hexdigest()
            if digest_path.exists():
                if digest_path.read_text(encoding="ascii").strip() != digest:
                    raise ValueError("Final label checksum mismatch")
            else:
                # Recover a crash between publishing the final file and its checksum.
                _atomic_write(digest_path, (digest + "\n").encode("ascii"))

    def state(self):
        with self.lock:
            next_index = next((k for k, item in enumerate(self._items)
                               if item["id"] not in self._latest), len(self._items))
            return {"n_total": len(self._items), "n_labeled": len(self._latest),
                    "next_index": next_index, "frozen": self.frozen}

    def item(self, k):
        if 0 <= k < len(self._items):
            return self._items[k]
        return None

    def label(self, opaque, label):
        with self.lock:
            if self.frozen:
                return 409
            if not isinstance(opaque, str) or opaque not in self._units or label not in LABELS:
                return 400
            entry = {"id": opaque, "label": label, "labeler": self.labeler,
                     "time": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")}
            descriptor = os.open(self.out / "labels.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(descriptor, "ab") as stream:
                stream.write(_json_bytes(entry))
                stream.flush()
                os.fsync(stream.fileno())
            self._latest[opaque] = label
            return 200

    def _final_document(self):
        return {"manifest_sha256": self.manifest_sha256, "labeler": self.labeler,
                "labels": {self._units[opaque]: label for opaque, label in self._latest.items()}}

    def finish(self):
        with self.lock:
            if self.frozen:
                return 409
            if len(self._latest) < len(self._items):
                return 422  # every row needs a label before the set is frozen
            final = _json_bytes(self._final_document())
            final_path = self.out / "labels-final.json"
            try:
                _atomic_write(final_path, final)
            finally:
                # Replacement may succeed even if the following directory fsync fails.
                self.frozen = final_path.exists()
            digest = hashlib.sha256(final).hexdigest()
            _atomic_write(self.out / "labels-final.sha256", (digest + "\n").encode("ascii"))
            return 200


class Handler(BaseHTTPRequestHandler):
    """An explicit route allowlist; errors never echo requests or exceptions."""

    def setup(self):
        super().setup()
        self.connection.settimeout(5)

    def version_string(self):
        return "MineralsLabeler"

    def log_message(self, *_args):
        pass

    def end_headers(self):
        self.send_header("Content-Security-Policy", CSP)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        super().end_headers()

    def _send(self, status, body, content_type="application/json; charset=utf-8"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def send_error(self, code, message=None, explain=None):
        self.close_connection = True
        self._send(code, _json_bytes({"error": "Request refused"}))

    def _host_ok(self):
        """Refuse DNS-rebinding requests: Host must name the bound address, Origin must match it."""
        hosts = self.headers.get_all("Host", [])
        if len(hosts) != 1 or hosts[0].lower() not in self.server.allowed_hosts:
            return False
        origins = self.headers.get_all("Origin", [])
        if len(origins) > 1:
            return False
        return not origins or origins[0].lower() in {"http://" + h for h in self.server.allowed_hosts}

    def do_GET(self):
        if not self._host_ok():
            self.send_error(403)
            return
        if self.path in self.server.assets:
            body, content_type = self.server.assets[self.path]
            self._send(200, body, content_type)
        elif self.path == "/api/state":
            self._send(200, _json_bytes(self.server.session.state()))
        elif match := re.fullmatch(r"/api/item/(0|[1-9][0-9]{0,9})", self.path):
            item = self.server.session.item(int(match[1]))
            if item is None:
                self.send_error(404)
            else:
                self._send(200, _json_bytes(item))
        else:
            self.send_error(404)

    def _body(self):
        lengths = self.headers.get_all("Content-Length", [])
        if self.headers.get("Transfer-Encoding") is not None or len(lengths) > 1:
            self.send_error(400)
            return None
        if not lengths:
            self.send_error(411)
            return None
        if not re.fullmatch(r"[0-9]{1,10}", lengths[0]):
            self.send_error(400)
            return None
        length = int(lengths[0])
        if length > BODY_LIMIT:
            self.send_error(413)
            return None
        if length and self.headers.get_content_type() != "application/json":
            self.send_error(415)
            return None
        try:
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError("Incomplete body")
            value = json.loads(raw) if raw else {}
            if not isinstance(value, dict):
                raise ValueError("Expected object")
            return value
        except (ValueError, UnicodeError, RecursionError, TimeoutError):
            self.send_error(400)
            return None

    def do_POST(self):
        if not self._host_ok():
            self.send_error(403)
            return
        if self.server.session.state()["frozen"]:
            self.send_error(409)
            return
        if self.path not in {"/api/label", "/api/finish"}:
            self.send_error(404)
            return
        value = self._body()
        if value is None:
            return
        try:
            if self.path == "/api/label":
                if (set(value) != {"id", "label"} or not isinstance(value["label"], str)):
                    self.send_error(400)
                    return
                status = self.server.session.label(value["id"], value["label"])
            else:
                if value:
                    self.send_error(400)
                    return
                status = self.server.session.finish()
            if status == 200:
                self._send(status, _json_bytes({"ok": True}))
            else:
                self.send_error(status)
        except OSError:
            self.send_error(500)


class LabelingServer(ThreadingHTTPServer):
    daemon_threads = True


def _wildcard(host):
    """True when any address the host resolves to is unspecified (0.0.0.0, ::, or a shorthand like "0")."""
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except (OSError, UnicodeError):
        return False  # unresolvable: the bind itself fails later, before any request is served
    return any(ipaddress.ip_address(info[4][0].split("%")[0]).is_unspecified for info in infos)


def create_server(manifest, out, labeler, host="127.0.0.1", port=8591,
                  allow_nonloopback=False):
    """Validate the binding before touching session data or opening a socket."""
    if host.lower() == "localhost":
        host = "127.0.0.1"
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if not allow_nonloopback and (address is None or not address.is_loopback):
        raise ValueError("Non-loopback binding requires --allow-nonloopback")
    if _wildcard(host):
        # The Host check needs the address clients will use; a wildcard bind has none.
        raise ValueError("Non-loopback wildcard binding is refused; bind the concrete interface address")
    session = Session(manifest, out, labeler)
    assets = {}
    for name, content_type in (("index.html", "text/html; charset=utf-8"),
                               ("app.js", "text/javascript; charset=utf-8"),
                               ("app.css", "text/css; charset=utf-8")):
        assets["/static/" + name] = ((STATIC_DIR / name).read_bytes(), content_type)
    assets["/"] = assets["/static/index.html"]
    server_class = LabelingServer
    if address is not None and address.version == 6:
        class IPv6LabelingServer(LabelingServer):
            address_family = socket.AF_INET6
        server_class = IPv6LabelingServer
    server = server_class((host, port), Handler)
    bound = server.server_address[1]
    names = {host.lower(), f"[{host.lower()}]"}
    if address is not None and address.is_loopback:
        names |= {"localhost", "127.0.0.1", "[::1]"}
    server.allowed_hosts = {f"{name}:{bound}" for name in names}
    server.session = session
    server.assets = assets
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--labeler", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8591)
    parser.add_argument("--allow-nonloopback", action="store_true")
    args = parser.parse_args(argv)
    try:
        server = create_server(**vars(args))
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as error:
        if isinstance(error, ValueError) and str(error).startswith("Non-loopback"):
            parser.error(str(error))
        parser.error("Cannot open labeling session; check the manifest and saved session files")
    host, port = server.server_address[:2]
    print(f"Labeler ready at http://{'[' + host + ']' if ':' in host else host}:{port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
