# Copyright (C) 2026 KiharaLab, Purdue University
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Lightweight server for the CryoZeta JSON preparation UI.

Serves the built React SPA and provides an API endpoint to launch
CryoZeta jobs in a tmux session.  Uses only the Python standard library.
"""

from __future__ import annotations

import http.server
import json
import os
import shutil
import socketserver
import subprocess
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

UI_DIR = Path(__file__).parent / "ui" / "dist"
SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent.parent  # repo root
RUN_SCRIPT = SCRIPT_DIR / "run_cryozeta.sh"

# Session registry: session_name -> output_dir
_sessions: dict[str, str] = {}


class _SPAHandler(http.server.SimpleHTTPRequestHandler):
    """Serve static files + handle /api/launch, /api/logs, /api/status."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(UI_DIR), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/logs":
            return self._handle_logs(parsed)
        if parsed.path == "/api/status":
            return self._handle_status(parsed)
        path = UI_DIR / self.path.lstrip("/")
        if not path.is_file() and not self.path.startswith("/assets"):
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self):
        if self.path == "/api/launch":
            return self._handle_launch()
        if self.path == "/api/terminate":
            return self._handle_terminate()
        self._json_response(404, {"error": "Not found"})

    def _read_json_body(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length))

    def _json_response(self, code: int, data: dict):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_launch(self):
        try:
            payload = self._read_json_body()
        except Exception as e:
            self._json_response(400, {"error": f"Invalid JSON: {e}"})
            return

        entries = payload.get("entries")
        config = payload.get("config", {})

        json_path = config.get("json_path", "").strip()
        output_dir = config.get("output_dir", "").strip()
        gpu = config.get("gpu", "").strip()

        if not json_path or not output_dir or not gpu:
            self._json_response(400, {"error": "json_path, output_dir, and gpu are required"})
            return

        # Resolve paths
        json_path = os.path.expanduser(json_path)
        output_dir = os.path.expanduser(output_dir)

        # Find run_cryozeta.sh
        script = str(RUN_SCRIPT)
        if not os.path.isfile(script):
            self._json_response(500, {"error": f"run_cryozeta.sh not found at {script}"})
            return

        # Save input JSON
        try:
            os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
            with open(json_path, "w") as f:
                json.dump(entries, f, indent=4)
        except Exception as e:
            self._json_response(500, {"error": f"Failed to save JSON: {e}"})
            return

        # Build command
        cmd_parts = [
            "bash", script,
            "--gpu", gpu,
        ]

        seeds = config.get("seeds", "101")
        if seeds:
            cmd_parts += ["--seeds", str(seeds)]
        if config.get("n_sample"):
            cmd_parts += ["--n-sample", str(config["n_sample"])]
        if config.get("n_step"):
            cmd_parts += ["--n-step", str(config["n_step"])]
        if config.get("n_cycle"):
            cmd_parts += ["--n-cycle", str(config["n_cycle"])]
        if config.get("num_select"):
            cmd_parts += ["--num-select", str(config["num_select"])]
        if config.get("interpolation"):
            cmd_parts.append("--interpolation")
        if config.get("skip_detection"):
            cmd_parts.append("--skip-detection")
        if config.get("skip_inference"):
            cmd_parts.append("--skip-inference")
        if config.get("skip_combine"):
            cmd_parts.append("--skip-combine")

        cmd_parts += [json_path, output_dir]

        # Derive a tmux session name from the json filename
        session_name = "cryozeta-" + Path(json_path).stem

        # Check tmux is available
        if not shutil.which("tmux"):
            self._json_response(500, {"error": "tmux is not installed"})
            return

        # Ensure output dir exists for log file
        os.makedirs(output_dir, exist_ok=True)
        log_file = os.path.join(output_dir, "cryozeta.log")

        # Launch in tmux, piping output through tee to the log file
        shell_cmd = " ".join(cmd_parts) + f" 2>&1 | tee {log_file}"
        tmux_cmd = [
            "tmux", "new-session", "-d", "-s", session_name,
            shell_cmd,
        ]

        try:
            subprocess.run(tmux_cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            self._json_response(500, {
                "error": f"tmux launch failed: {e.stderr.strip() or str(e)}"
            })
            return

        # Register the session
        _sessions[session_name] = output_dir

        self._json_response(200, {
            "message": f"Job launched in tmux session '{session_name}'. "
                       f"Attach with: tmux attach -t {session_name}",
            "session": session_name,
        })

    def _handle_status(self, parsed):
        params = parse_qs(parsed.query)
        session_name = params.get("session", [None])[0]
        if not session_name:
            self._json_response(400, {"error": "session parameter required"})
            return
        running = _is_tmux_session_alive(session_name)
        self._json_response(200, {"running": running})

    def _handle_terminate(self):
        try:
            payload = self._read_json_body()
        except Exception as e:
            self._json_response(400, {"error": f"Invalid JSON: {e}"})
            return

        session_name = payload.get("session", "")
        if not session_name:
            self._json_response(400, {"error": "session parameter required"})
            return

        if not _is_tmux_session_alive(session_name):
            self._json_response(200, {"message": "Session already finished"})
            return

        try:
            subprocess.run(
                ["tmux", "kill-session", "-t", session_name],
                check=True, capture_output=True, text=True,
            )
            self._json_response(200, {"message": f"Session '{session_name}' terminated"})
        except subprocess.CalledProcessError as e:
            self._json_response(500, {
                "error": f"Failed to terminate: {e.stderr.strip() or str(e)}"
            })

    def _handle_logs(self, parsed):
        params = parse_qs(parsed.query)
        session_name = params.get("session", [None])[0]
        if not session_name:
            self._json_response(400, {"error": "session parameter required"})
            return

        output_dir = _sessions.get(session_name)
        if not output_dir:
            self._json_response(404, {"error": f"Unknown session: {session_name}"})
            return

        log_file = os.path.join(output_dir, "cryozeta.log")

        # SSE response headers
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        offset = 0
        try:
            while True:
                # Read new content from the log file
                if os.path.isfile(log_file):
                    with open(log_file, "r") as f:
                        f.seek(offset)
                        new_data = f.read()
                        if new_data:
                            offset += len(new_data.encode())
                            for line in new_data.splitlines():
                                self.wfile.write(f"data: {line}\n\n".encode())
                            self.wfile.flush()

                # Check if session is still running
                if not _is_tmux_session_alive(session_name):
                    # Read any remaining data
                    if os.path.isfile(log_file):
                        with open(log_file, "r") as f:
                            f.seek(offset)
                            remaining = f.read()
                            if remaining:
                                for line in remaining.splitlines():
                                    self.wfile.write(f"data: {line}\n\n".encode())
                    self.wfile.write(b"event: done\ndata: finished\n\n")
                    self.wfile.flush()
                    break

                time.sleep(1)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, format, *args):
        if not self.path.startswith("/assets"):
            super().log_message(format, *args)


def _is_tmux_session_alive(session_name: str) -> bool:
    try:
        subprocess.run(
            ["tmux", "has-session", "-t", session_name],
            check=True, capture_output=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


class _ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


def run_server(port: int = 8501) -> None:
    """Start the server for the React UI."""
    if not (UI_DIR / "index.html").is_file():
        raise FileNotFoundError(
            f"UI build not found at {UI_DIR}. "
            "Run 'npm run build' in src/cryozeta/runner/ui/ first."
        )

    with _ThreadingHTTPServer(("0.0.0.0", port), _SPAHandler) as httpd:
        print(f"Serving CryoZeta UI at http://localhost:{port}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down.")


if __name__ == "__main__":
    run_server()
