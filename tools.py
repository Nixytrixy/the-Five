"""Tool implementations for the agent: sandboxed Python execution and
persistent per-task HTTP sessions, plus their Anthropic tool-use schemas.

Design notes
------------
- `run_python` executes in a fresh subprocess per call (isolated, a real
  kill-on-timeout, can't corrupt another tile's state) with cwd set to the
  task's workdir, so the model can read downloaded files by plain filename.
  Nothing persists between calls except files written to disk -- that's the
  intended way for the model to carry state across turns.
- `http_request` is backed by a `requests.Session` kept per task_id, so
  cookies/auth persist across calls automatically. The model never manages
  a cookie jar itself; it just keeps calling this tool the way a browser
  would click through a login/multi-step flow.
- `submit_answer` deliberately has NO answer text field. See agent.py:
  the final value is captured programmatically from an `ANSWER: ...` line
  in run_python's stdout, never retyped by the model into a tool argument.
  That's the structural fix the hackathon README calls out: "verify your
  work" prompting doesn't stop a transcription slip, but never asking the
  model to retype the token does.
"""
from __future__ import annotations

import subprocess
import sys
import threading

import requests

MAX_OUTPUT_CHARS = 8000
DEFAULT_TIMEOUT_S = 20

_sessions: dict[str, requests.Session] = {}
_sessions_lock = threading.Lock()


def _session_for(task_id: str) -> requests.Session:
    with _sessions_lock:
        s = _sessions.get(task_id)
        if s is None:
            s = requests.Session()
            s.headers["User-Agent"] = "Mozilla/5.0 (agent-jeopardy)"
            _sessions[task_id] = s
        return s


def _truncate(s: str | None, limit: int = MAX_OUTPUT_CHARS) -> str:
    if not s:
        return ""
    if len(s) <= limit:
        return s
    half = limit // 2
    return s[:half] + f"\n...[{len(s) - limit} chars truncated]...\n" + s[-half:]


def run_python(code: str, workdir: str, timeout: int = DEFAULT_TIMEOUT_S) -> dict:
    """Run `code` as a standalone Python script with cwd=workdir.

    Fresh interpreter per call. Truncates stdout/stderr so a huge dump never
    blows the model's context.
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "stdout": _truncate(proc.stdout),
            "stderr": _truncate(proc.stderr),
            "returncode": proc.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": f"TIMEOUT after {timeout}s", "returncode": -1}
    except Exception as e:  # noqa: BLE001
        return {"stdout": "", "stderr": f"execution error: {e!r}", "returncode": -1}


def http_request(task_id: str, method: str, url: str, headers: dict | None = None,
                  params: dict | None = None, data=None, json_body=None,
                  allow_redirects: bool = True, timeout: int = 20) -> dict:
    """One HTTP call using the persistent session for this task.

    Cookies the server sets are kept automatically and sent on the next call
    for the same task_id -- a login/multi-step flow is just several of these
    calls in a row.
    """
    s = _session_for(task_id)
    try:
        r = s.request(
            method.upper(), url,
            headers=headers or None,
            params=params or None,
            data=data if json_body is None else None,
            json=json_body,
            allow_redirects=allow_redirects,
            timeout=timeout,
        )
        body = r.text
        return {
            "status_code": r.status_code,
            "url": r.url,
            "headers": dict(r.headers),
            "cookies": s.cookies.get_dict(),
            "body": _truncate(body),
            "body_truncated": len(body) > MAX_OUTPUT_CHARS,
        }
    except Exception as e:  # noqa: BLE001
        return {"status_code": None, "error": repr(e)}


# ---------------------------------------------------------------- schemas

TOOLS = [
    {
        "name": "run_python",
        "description": (
            "Execute a standalone Python 3.12 script. cwd is the task's "
            "workdir, which already contains any downloaded task files -- "
            "read them by plain filename. Each call is a FRESH interpreter: "
            "nothing persists between calls except files you write to disk, "
            "so save intermediate results to a file if you need them next "
            "turn. Packages available: requests, beautifulsoup4 (bs4), lxml, "
            "numpy, pandas. stdout/stderr are returned, truncated if huge. "
            "When you have the final answer, print it as the LAST line of "
            "stdout in the exact form `ANSWER: <value>` -- that line is read "
            "programmatically and is the only thing that gets submitted, so "
            "it must hold exactly the right string/number/literal and "
            "nothing else."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python source to run."},
                "timeout": {"type": "integer",
                            "description": "Seconds before it's killed (default 20, max 60)."},
            },
            "required": ["code"],
        },
    },
    {
        "name": "http_request",
        "description": (
            "Make one HTTP request. Cookies persist automatically across "
            "calls for this tile -- you do not need to read or resend "
            "Set-Cookie yourself, just keep calling this tool as you would "
            "click through a site: GET the page, POST the login form, GET "
            "the next page, etc. Returns status_code, headers, cookies, and "
            "the response body (truncated if huge)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "method": {"type": "string",
                           "enum": ["GET", "POST", "PUT", "DELETE", "HEAD", "PATCH"]},
                "url": {"type": "string"},
                "headers": {"type": "object"},
                "params": {"type": "object", "description": "URL query params."},
                "data": {"type": "object", "description": "Form-encoded body."},
                "json_body": {"type": "object", "description": "JSON body (sets Content-Type)."},
            },
            "required": ["method", "url"],
        },
    },
    {
        "name": "submit_answer",
        "description": (
            "Confirm you are ready to submit. Call this once, only after "
            "run_python has printed an `ANSWER: <value>` line and you have "
            "verified it (re-derived it a second, independent way if the "
            "format allows). There is no answer field here on purpose -- "
            "the value submitted is pulled directly from your ANSWER line, "
            "never retyped, so a one-character transcription slip can't "
            "happen. If you haven't printed an ANSWER line yet, do that "
            "first."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "confidence": {
                    "type": "string",
                    "enum": ["verified", "unverified"],
                    "description": "'verified' if you cross-checked the answer a second way.",
                },
            },
            "required": ["confidence"],
        },
    },
]
