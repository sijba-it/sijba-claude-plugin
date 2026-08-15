#!/usr/bin/env python3
"""Claude Code plugin for sijba — standalone, no dependency on
sijba-agent-controller's own code. Runnable on its own (env vars below) or
spawned by the controller as a `kind: git` catalog plugin.

Plugin contract (documented in sijba-agent-controller/README.md):
  - read SIJBA_URL / SIJBA_TOKEN / PROJECT_NAME / WORK_DIR / REGISTRY_FILE
    from the environment
  - self-register exactly one sijba channel over its own WebSocket connection
  - handle /archive and /pause locally: write REGISTRY_FILE's own status
    entry directly (no shared import — see _set_registry_status below) and
    exit with the reserved code (90 archived / 91 paused) *before* the
    process actually stops, so the controller can tell "stopped on
    purpose" apart from "crashed" purely from the exit code
  - handle /log [n] locally from an in-memory ring buffer

Environment variables
----------------------
SIJBA_URL       sijba WebSocket URL (default: wss://sijba.edv-bartl.at)
SIJBA_TOKEN     Auth token (required)
PROJECT_NAME    Registry key / trigger word (required) — e.g. "sijba-web"
WORK_DIR        Working directory for Claude (required)
REGISTRY_FILE   Path to the controller's registry.json (required)
CLAUDE_CMD      Claude CLI command (default: claude)
"""
from __future__ import annotations

import asyncio
import collections
import fcntl
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import websockets

# Exit-code contract with the controller — see sijba-agent-controller/exit_codes.py.
EXIT_ARCHIVED = 90
EXIT_PAUSED = 91

PLUGIN_NAME = "claude"  # this repo *is* the claude plugin — no TOOL switch needed

SIJBA_URL = os.environ.get("SIJBA_URL", "wss://sijba.edv-bartl.at")
SIJBA_TOKEN = os.environ.get("SIJBA_TOKEN", "")
PROJECT_NAME = os.environ.get("PROJECT_NAME", "")
WORK_DIR = Path(os.environ.get("WORK_DIR", ""))
REGISTRY_FILE = os.environ.get("REGISTRY_FILE", "")
CLAUDE_CMD = os.environ.get("CLAUDE_CMD", "claude")

if not SIJBA_TOKEN:
    print("ERROR: SIJBA_TOKEN environment variable required", file=sys.stderr)
    sys.exit(1)
if not PROJECT_NAME:
    print("ERROR: PROJECT_NAME environment variable required", file=sys.stderr)
    sys.exit(1)
if not WORK_DIR.is_dir():
    print(f"ERROR: WORK_DIR {WORK_DIR} does not exist", file=sys.stderr)
    sys.exit(1)
if not REGISTRY_FILE:
    print("ERROR: REGISTRY_FILE environment variable required", file=sys.stderr)
    sys.exit(1)

PLUGIN_ID = f"{PLUGIN_NAME}-{PROJECT_NAME}"
CHANNEL_ID = f"{PLUGIN_NAME}:{PROJECT_NAME}"

_LOG_BUFFER: collections.deque[str] = collections.deque(maxlen=200)


def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    line = f"{ts} [{PROJECT_NAME}] {msg}"
    _LOG_BUFFER.append(line)
    print(line, flush=True)


def _set_registry_status(name: str, status: str) -> None:
    """Read-modify-write REGISTRY_FILE directly — same JSON shape as
    sijba-agent-controller/registry.py, but no import of it: this repo has
    to stand on its own, same as any other git-sourced plugin would."""
    path = Path(REGISTRY_FILE)
    with open(path, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            raw = f.read().strip()
            data = json.loads(raw) if raw else {"roots": [], "projects": {}}
            data.setdefault("projects", {}).setdefault(name, {})["status"] = status
            data["projects"][name]["status_changed_at"] = time.time()
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
            os.replace(tmp, path)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# Claude backend (persistent subprocess, JSON-lines over stdin/stdout)
# ---------------------------------------------------------------------------

class ClaudeSession:
    def __init__(self, work_dir: Path):
        self.work_dir = work_dir
        self.proc: asyncio.subprocess.Process | None = None
        self._pending: dict[int, asyncio.Future] = {}
        self._stdout_task: asyncio.Task | None = None
        self._stderr_task: asyncio.Task | None = None

    async def start(self):
        self.proc = await asyncio.create_subprocess_exec(
            CLAUDE_CMD,
            "--dangerously-skip-permissions",
            "--model", "sonnet",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.work_dir,
        )
        self._stdout_task = asyncio.create_task(self._read_stdout())
        self._stderr_task = asyncio.create_task(self._read_stderr())
        _log(f"Claude started (pid={self.proc.pid})")

    async def _read_stdout(self):
        while self.proc and self.proc.stdout:
            line = await self.proc.stdout.readline()
            if not line:
                break
            try:
                msg = json.loads(line.decode())
                req_id = msg.get("request_id")
                if req_id in self._pending:
                    self._pending[req_id].set_result(msg)
                    del self._pending[req_id]
            except json.JSONDecodeError:
                pass

    async def _read_stderr(self):
        while self.proc and self.proc.stderr:
            line = await self.proc.stderr.readline()
            if not line:
                break
            _log(f"stderr: {line.decode().strip()}")

    async def send(self, text: str, request_id: int, on_chunk=None) -> dict:
        if not self.proc or not self.proc.stdin:
            raise RuntimeError("Claude process not running")
        future = asyncio.get_event_loop().create_future()
        self._pending[request_id] = future
        payload = {"request_id": request_id, "text": text}
        self.proc.stdin.write((json.dumps(payload) + "\n").encode())
        await self.proc.stdin.drain()
        try:
            return await asyncio.wait_for(future, timeout=120)
        except asyncio.TimeoutError:
            raise RuntimeError("Claude response timeout")

    async def close(self):
        if self.proc:
            self.proc.terminate()
            await self.proc.wait()
        for task in [self._stdout_task, self._stderr_task]:
            if task:
                task.cancel()
        _log("Claude stopped")

    async def reset(self) -> None:
        """Kill the current persistent process (if any) and clear its state
        so the next turn spawns a brand-new `claude` with no prior context —
        it's a fresh invocation, not --resume/--continue, so a new process
        is inherently a clean conversation."""
        if self.proc:
            self.proc.terminate()
            try:
                await asyncio.wait_for(self.proc.wait(), timeout=10)
            except asyncio.TimeoutError:
                self.proc.kill()
        for task in [self._stdout_task, self._stderr_task]:
            if task:
                task.cancel()
        self.proc = None
        self._stdout_task = None
        self._stderr_task = None
        self._pending.clear()
        _log("Claude session reset")


# ---------------------------------------------------------------------------
# sijba plugin protocol (single project channel, self-registered)
# ---------------------------------------------------------------------------

class PluginClient:
    def __init__(self):
        self.ws = None
        self.session = ClaudeSession(WORK_DIR)
        self.session_started = False
        self.shutdown_event = asyncio.Event()
        self.exit_code = 0
        self._reconnect_backoff = 5
        self._request_id = 0

    async def connect(self):
        ws_url = SIJBA_URL.replace("https://", "wss://").replace("http://", "ws://")
        ws_url = f"{ws_url}/api/plugin/ws?token={SIJBA_TOKEN}"
        _log(f"Connecting to {ws_url} as {PLUGIN_ID}")
        self.ws = await websockets.connect(ws_url, ping_interval=20, ping_timeout=10, close_timeout=5)
        _log("Connected!")

        await self.ws.send(json.dumps({
            "type": "register",
            "plugin_id": PLUGIN_ID,
            "capabilities": ["conversation", "tools", "files", "code", "agent"],
            "channels": {
                CHANNEL_ID: {
                    "label": f"Claude: {PROJECT_NAME}",
                    "hint": (
                        f"Claude Code agent for project '{PROJECT_NAME}' at {WORK_DIR}. "
                        f"Trigger: '{PROJECT_NAME} ...'. In-channel commands: /archive, /pause, /log [n]."
                    ),
                    "triggers": [PROJECT_NAME, CHANNEL_ID],
                    "capabilities": ["code", "analysis", "debugging", "files", "terminal"],
                    "metadata": {"plugin": PLUGIN_NAME, "project": PROJECT_NAME, "working_dir": str(WORK_DIR)},
                }
            },
        }))
        self._reconnect_backoff = 5
        await self.message_loop()

    async def message_loop(self):
        try:
            async for raw in self.ws:
                msg = json.loads(raw)
                mtype = msg.get("type")
                if mtype == "registered":
                    _log(f"Plugin registered: {msg['plugin_id']}")
                elif mtype == "channel_registered":
                    _log(f"Channel registered: {msg['channel_id']} (ok={msg.get('ok')})")
                elif mtype == "message":
                    await self.handle_message(msg)
                elif mtype == "ping":
                    await self.ws.send(json.dumps({"type": "pong"}))
                elif mtype == "error":
                    _log(f"Server error: {msg.get('message')}")
        except websockets.ConnectionClosed:
            _log("Connection closed")
        except Exception as exc:
            _log(f"Message loop error: {exc}")

    async def _reply(self, channel_id: str, envelope_id: str, text: str) -> None:
        try:
            await self.ws.send(json.dumps({
                "type": "reply", "channel": channel_id, "envelope_id": envelope_id,
                "text": text, "continue": True,
            }))
        except Exception as exc:
            _log(f"Failed to send reply: {exc}")

    async def _handle_local_command(self, channel_id: str, envelope_id: str, text: str) -> bool:
        cmd, _, rest = text[1:].partition(" ")
        cmd = cmd.lower().strip()

        if cmd == "log":
            n = 10
            if rest.strip().isdigit():
                n = min(int(rest.strip()), len(_LOG_BUFFER) or 10)
            lines = list(_LOG_BUFFER)[-n:]
            await self._reply(channel_id, envelope_id, "\n".join(lines) or "(no logs yet)")
            return True

        if cmd in ("archive", "pause"):
            status = "archived" if cmd == "archive" else "paused"
            code = EXIT_ARCHIVED if cmd == "archive" else EXIT_PAUSED
            try:
                _set_registry_status(PROJECT_NAME, status)
            except Exception as exc:
                await self._reply(channel_id, envelope_id, f"Error updating registry: {exc}")
                return True
            await self._reply(channel_id, envelope_id, f"{PROJECT_NAME} {status}. Stopping.")
            _log(f"{status} via /{cmd} — shutting down (exit {code})")
            self.exit_code = code
            self.shutdown_event.set()
            await self.ws.close()
            return True

        if cmd == "reset":
            await self.session.reset()
            self.session_started = False
            await self._reply(channel_id, envelope_id, f"{PROJECT_NAME} reset — next message starts fresh, no prior context.")
            return True

        if cmd:
            await self._reply(
                channel_id, envelope_id,
                "Unsupported here: /archive, /pause, /reset, /log [n]. Anything else — just plain text.",
            )
            return True
        return False

    async def handle_message(self, msg: dict):
        channel_id = msg["channel"]
        payload = msg["payload"]
        envelope_id = payload["envelope_id"]
        text = payload["text"]

        for trigger in (PROJECT_NAME, CHANNEL_ID):
            if text.lower().startswith(trigger.lower()):
                text = text[len(trigger):].strip()
                break
        if not text:
            return

        if text.lstrip().startswith("/"):
            if await self._handle_local_command(channel_id, envelope_id, text.lstrip()):
                return

        if not self.session_started:
            await self.session.start()
            self.session_started = True

        _log(f"handling envelope={envelope_id} text={text[:120]!r}")
        self._request_id += 1
        reply_text = ""
        try:
            response = await self.session.send(text, request_id=self._request_id)
            reply_text = response.get("text", "")
        except Exception as exc:
            reply_text = f"Error: {exc}"

        await self._reply(channel_id, envelope_id, reply_text)

    async def run_forever(self):
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, self.shutdown_event.set)
            except NotImplementedError:
                pass

        while not self.shutdown_event.is_set():
            try:
                await self.connect()
            except Exception as exc:
                _log(f"Connection failed: {exc}")
            if self.shutdown_event.is_set():
                break
            _log(f"Reconnecting in {self._reconnect_backoff}s...")
            try:
                await asyncio.wait_for(self.shutdown_event.wait(), timeout=self._reconnect_backoff)
            except asyncio.TimeoutError:
                pass
            self._reconnect_backoff = min(self._reconnect_backoff * 2, 300)

        await self.session.close()
        _log("Shutdown complete")


async def main() -> int:
    client = PluginClient()
    await client.run_forever()
    return client.exit_code


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
