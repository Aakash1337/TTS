#!/usr/bin/env python
"""Launch the TTS Reader web app.

    E:\\TTS\\.venv\\Scripts\\python.exe serve.py            # opens your browser
    E:\\TTS\\.venv\\Scripts\\python.exe serve.py --no-browser

Then open http://127.0.0.1:8000 if it doesn't open automatically.
"""

from __future__ import annotations

import socket
import sys
import threading
import webbrowser

import uvicorn

from webapp.server import app

HOST = "127.0.0.1"
# 8000 is reserved/held on this machine, so try friendly ports in order and fall
# back to an OS-assigned free one. Override with:  serve.py --port 5005
PREFERRED_PORTS = [8756, 8080, 5005, 7860, 8888, 3000]


def _pick_port() -> int:
    if "--port" in sys.argv:
        try:
            explicit = int(sys.argv[sys.argv.index("--port") + 1])
        except (ValueError, IndexError):
            sys.exit("Invalid --port value — usage: serve.py --port 8756")
        s = socket.socket()
        try:
            s.bind((HOST, explicit))
        except OSError:
            sys.exit(f"Port {explicit} is not available on this machine — "
                     "pick another (e.g. --port 8756) or omit --port to auto-select.")
        finally:
            s.close()
        return explicit
    for p in PREFERRED_PORTS:
        s = socket.socket()
        try:
            s.bind((HOST, p))
            return p
        except OSError:
            continue
        finally:
            s.close()
    s = socket.socket(); s.bind((HOST, 0)); p = s.getsockname()[1]; s.close()
    return p


def main() -> None:
    port = _pick_port()
    url = f"http://{HOST}:{port}/"
    print(f"\n  TTS Reader  ->  {url}\n  (press Ctrl+C to stop)\n")
    if "--no-browser" not in sys.argv:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=HOST, port=port, log_level="warning")


if __name__ == "__main__":
    main()
