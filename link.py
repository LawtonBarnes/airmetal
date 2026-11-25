"""Tells the rest of the fleet which aircraft AIR METAL is riding, so JOAN
JET (on P4) can circle it on the radar.

A tiny HTTP server on its own thread: GET /riding returns
{"hex": "a1b2c3", "callsign": "SWA113", "delay_sec": 2.5}, or
{"hex": null, ...} in standby. EXITROW (the side windows on P1/P3) reads
the same feed and flies the same aircraft `delay_sec` behind live, so all
the windows show one moment.
The render loop just assigns `riding`; serving costs it nothing.
"""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class RideServer:
    def __init__(self, port, delay_sec=0.0):
        self.riding = None  # (hex, callsign) or None
        self.delay_sec = delay_sec
        link = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path != "/riding":
                    self.send_error(404)
                    return
                ride = link.riding
                msg = {"hex": ride[0], "callsign": ride[1]} if ride else {"hex": None}
                msg["delay_sec"] = link.delay_sec
                body = json.dumps(msg).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass  # one request every couple of seconds, forever

        try:
            self.server = ThreadingHTTPServer(("", port), Handler)
        except OSError as exc:
            # Another copy already has the port (e.g. a dev run beside the
            # real one) -- ride on without publishing.
            print(f"Ride link not available on port {port}: {exc}", file=sys.stderr)
            self.server = None
            return
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
