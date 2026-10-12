"""PSL Progress Tracker - local app. Serves the UI on localhost and sends photos
to a local Ollama vision model. Nothing leaves your machine."""
import json, os, sys, threading, time, webbrowser, urllib.request, urllib.error
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
import statistics
import geometry

PORT = int(os.environ.get("PSL_PORT", 8765))
MODEL = os.environ.get("PSL_MODEL", "qwen2.5vl:3b")
OLLAMA = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
BASE = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
DATA_DIR = Path(os.environ.get("APPDATA") or Path.home() / ".local" / "share") / "PSLTracker"
DATA_DIR.mkdir(parents=True, exist_ok=True)
STORE = DATA_DIR / "store.json"
MAX_BODY = 25 * 1024 * 1024
SAMPLES = max(1, int(os.environ.get("PSL_SAMPLES", 3)))  # model runs per photo; the median is used
GEO_WEIGHT = 0.6  # share of harmony/dimorphism/angularity that comes from measured landmarks
CATS = ("harmony", "dimorphism", "angularity", "misc")


def ollama(path, payload=None, timeout=600):
    req = urllib.request.Request(
        OLLAMA + path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def status():
    try:
        names = [m["name"] for m in ollama("/api/tags", timeout=3).get("models", [])]
    except Exception:
        return {"ok": False, "help": "Ollama isn't running. Install it from ollama.com and start it, then reload."}
    if not any(n == MODEL or n.split(":")[0] == MODEL for n in names):
        return {"ok": False, "help": f"Model missing. Run in a terminal:  ollama pull {MODEL}"}
    return {"ok": True}


def parse_json(text):
    try:
        return json.loads(text)
    except Exception:
        a, b = text.find("{"), text.rfind("}")
        if a >= 0 and b > a:
            return json.loads(text[a:b + 1])
        raise


def shrink(v):
    """Small models over-spread and inflate. Pull toward the population mean of 4."""
    return 4 + (min(8.0, max(0.0, float(v))) - 4) * 0.65


def analyse(req):
    sex = req.get("sex", "male")
    geo = geometry.measure(req["image"], sex) if geometry.AVAILABLE else None
    if geo and geo.get("error"):
        return {"valid": False, "reason": geo["error"]}
    outs = []
    for i in range(SAMPLES):
        res = ollama("/api/chat", {
            "model": MODEL, "stream": False, "format": "json",
            "options": {"temperature": 0.3, "seed": 7 + i},
            "messages": [{"role": "user", "content": req["prompt"], "images": [req["image"]]}]})
        try:
            o = parse_json(res["message"]["content"])
        except Exception:
            continue
        if o.get("valid") is False:
            return o  # e.g. looks under 18: always honour a rejection
        if all(isinstance(o.get(k), (int, float)) for k in CATS):
            outs.append(o)
    if not outs:
        raise ValueError("no usable model output")
    out, spread = dict(outs[0]), 0.0
    for k in CATS:
        vals = [shrink(o[k]) for o in outs]
        spread = max(spread, max(vals) - min(vals))
        v = statistics.median(vals)
        if geo and k != "misc":
            v = GEO_WEIGHT * geo[k + "_g"] + (1 - GEO_WEIGHT) * v
        out[k] = round(v, 1)
    out["spread"] = round(spread, 2)
    if geo:
        out["geo"] = {x: geo[x] for x in ("raw", "scores", "pose")}
        out["photoQuality"] = list(geo["warnings"]) + list(out.get("photoQuality") or [])
    if spread > 1.0 or (geo and geo["warnings"]):
        out["confidence"] = "low"
    return out


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        if not isinstance(body, bytes):
            body = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _local(self):  # block DNS-rebinding / other sites poking the API
        return self.headers.get("Host", "").split(":")[0] in ("127.0.0.1", "localhost")

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        if n > MAX_BODY:
            raise ValueError("Request too large")
        return self.rfile.read(n)

    def do_GET(self):
        if not self._local():
            return self._send(403, {"error": "forbidden"})
        if self.path in ("/", "/index.html"):
            return self._send(200, (BASE / "index.html").read_bytes(), "text/html; charset=utf-8")
        if self.path == "/api/store":
            return self._send(200, STORE.read_bytes() if STORE.exists() else b"{}")
        if self.path == "/api/status":
            return self._send(200, status())
        self._send(404, {"error": "not found"})

    def do_PUT(self):
        if not self._local() or self.path != "/api/store":
            return self._send(403, {"error": "forbidden"})
        try:
            data = json.loads(self._body())
            tmp = STORE.with_suffix(".tmp")
            tmp.write_text(json.dumps(data))
            tmp.replace(STORE)
            self._send(200, {"ok": True})
        except Exception as e:
            self._send(400, {"error": str(e)})

    def do_POST(self):
        if not self._local() or self.path != "/api/analyze":
            return self._send(403, {"error": "forbidden"})
        st = status()
        if not st["ok"]:
            return self._send(503, {"error": st["help"]})
        try:
            self._send(200, {"out": analyse(json.loads(self._body()))})
        except urllib.error.URLError as e:
            self._send(502, {"error": f"Could not reach Ollama: {e.reason}"})
        except Exception as e:
            self._send(500, {"error": f"Model returned something unusable ({e}). Try again."})


def main():
    try:
        srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    except OSError:
        webbrowser.open(f"http://127.0.0.1:{PORT}")  # already running
        return
    url = f"http://127.0.0.1:{PORT}"
    print(f"PSL Progress Tracker running at {url}\nModel: {MODEL} x{SAMPLES} runs/photo, landmarks: {"on" if geometry.AVAILABLE else "OFF (pip install mediapipe)"}\nData: {STORE}\nClose this window to quit.")
    threading.Thread(target=lambda: (time.sleep(0.8), webbrowser.open(url)), daemon=True).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
