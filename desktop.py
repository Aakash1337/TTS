#!/usr/bin/env python
"""TTS Reader — desktop app.

Wraps the exact same web UI in a native window (pywebview -> Edge WebView2 on
Windows): no browser chrome, no terminal, no code duplicated. The FastAPI
backend runs on an embedded uvicorn bound to a free localhost port; closing the
window exits the process (the server thread is a daemon).

    E:\\TTS\\.venv\\Scripts\\pythonw.exe desktop.py     # silent, no console
    E:\\TTS\\.venv\\Scripts\\python.exe  desktop.py     # with console (debug)
"""

from __future__ import annotations

import socket
import threading
import time

import uvicorn
import webview

from webapp.server import app

HOST = "127.0.0.1"


def _free_port() -> int:
    s = socket.socket()
    s.bind((HOST, 0))          # OS-assigned free port; avoids the blocked-8000 issue
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_until_up(url: str, timeout: float = 15.0) -> None:
    import urllib.request
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=2):
                return
        except Exception:
            time.sleep(0.15)
    # Window will show a load error if we get here; nothing better to do.


def main() -> None:
    port = _free_port()
    url = f"http://{HOST}:{port}/"

    print(f"TTS Reader desktop -> {url}")   # visible only when run with python.exe
    server = uvicorn.Server(uvicorn.Config(app, host=HOST, port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    _wait_until_up(url)

    webview.create_window(
        "TTS Reader", url,
        width=880, height=940,
        min_size=(420, 600),
        background_color="#0f1220",   # match the page so startup doesn't flash white
    )
    webview.start()                   # blocks until the window is closed


if __name__ == "__main__":
    main()
