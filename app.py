"""PSL Progress Tracker - local app. Serves the UI on localhost and sends photos
to a local Ollama vision model. Nothing leaves your machine."""
import json, os, sys, threading, time, webbrowser, urllib.request, urllib.error
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path

PORT = int(os.environ.get("PSL_PORT", 8765))
MODEL = os.environ.get("PSL_MODEL", "qwen2.5vl:3b")
OLLAMA = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
BASE = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
DATA_DIR = Path(os.environ.get("APPDATA") or Path.home() / ".local" / "share") / "PSLTracker"
DATA_DIR.mkdir(parents=True, exist_ok=True)
STORE = DATA_DIR / "store.json"
MAX_BODY = 25 * 1024 * 1024


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
            req = json.loads(self._body())
            res = ollama("/api/chat", {
                "model": MODEL,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.2},
                "messages": [{"role": "user", "content": req["prompt"], "images": [req["image"]]}],
            })
            self._send(200, {"out": parse_json(res["message"]["content"])})
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
    print(f"PSL Progress Tracker running at {url}\nModel: {MODEL}\nData: {STORE}\nClose this window to quit.")
    threading.Thread(target=lambda: (time.sleep(0.8), webbrowser.open(url)), daemon=True).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
