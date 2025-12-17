import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse
# ensure project root is on path when running from web/ directory
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from voiceguard.pipeline import build_pipeline


PIPELINE = build_pipeline()


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, obj, status=200):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def _send_text(self, text, status=200):
        data = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        # basic CORS support
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            try:
                with open("web/index.html", "r", encoding="utf-8") as f:
                    content = f.read()
                self._send_text(content, 200)
            except Exception as e:
                self._send_text(f"<h1>VoiceGuard</h1><pre>{e}</pre>", 200)
        else:
            self._send_text("Not Found", 404)

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/predict":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                return self._send_json({"error": "invalid JSON"}, 400)
            voice_text = payload.get("voice_text", "")
            telemetry = payload.get("telemetry", {})
            customer = payload.get("customer", {"id": "unknown", "location": [12.9716, 77.5946]})
            result = PIPELINE.run(voice_text, telemetry, customer)
            return self._send_json(result, 200)
        else:
            self._send_json({"error": "not_found"}, 404)


def run_server(host="127.0.0.1", port=8000):
    server = HTTPServer((host, port), Handler)
    print(f"VoiceGuard server listening on http://{host}:{port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    run_server()
