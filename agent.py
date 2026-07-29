"""The tool-use loop: one tile in, a verified answer out.

This is the piece the naive baseline main.py doesn't have. Given a task_id,
`solve_tile` builds a category-tailored system prompt, hands the model three
tools (run_python, http_request, submit_answer), and drives the Anthropic
tool-use loop -- call, execute, feed results back, repeat -- until the model
calls submit_answer with a captured ANSWER line, or MAX_TURNS runs out.

Deliberately does NOT submit to the game server. main.py owns submission so
many tiles can be solved concurrently while all their submits still funnel
through one rate-limited gate.
"""
from __future__ import annotations

import json
import re

import jeopardy as jp
import tools

MAX_TURNS = 14
ANSWER_RE = re.compile(r"^ANSWER:\s*(.*)$", re.MULTILINE)

# Short, category-specific steering. Keeps a small model (Haiku) from
# reaching for the wrong tool or guessing instead of computing.
CATEGORY_HINTS = {
    "Needle in the Haystack": (
        "Messy-data wrangling at a size you cannot eyeball. Load the file(s) "
        "in workdir with pandas/numpy and filter/aggregate in code -- do not "
        "guess at contents from a partial read."
    ),
    "The Dark Web": (
        "A real HTTP flow: sessions, forms, cookies, possibly multi-step "
        "logins. Use http_request repeatedly -- cookies persist automatically "
        "across calls for this tile. Inspect each response body (forms, "
        "hidden fields, redirects) before guessing the next step."
    ),
    "Ship It": (
        "There is code in workdir that is broken or needs to be run. Read "
        "it, run it, and fix it with run_python; derive the answer from its "
        "actual output, not from reading the source and guessing what it "
        "would print."
    ),
    "Ancient Scrolls": (
        "A long document with cross-references or amendments. Load the "
        "whole thing in run_python and search/index it programmatically -- "
        "do not rely on skimming a summary of it."
    ),
    "Cryptic": (
        "An encoding, archive, binary format, or light cipher. Identify the "
        "format first (magic bytes, structure, byte layout) before writing "
        "decode code -- do not guess plaintext."
    ),
    "Heavy Compute": (
        "Needs a real search/optimization algorithm in code -- brute force "
        "only if the space is small enough, otherwise write something "
        "smarter. Verify the candidate answer against the problem's own "
        "constraints before printing it."
    ),
}

_FORMAT_HINTS = {
    "exact": "Whitespace-normalized exact string match.",
    "exact_ci": "Case-insensitive exact string match.",
    "numeric": "Parsed as a number, small tolerance; $ and commas are fine.",
    "literal": "Parsed as a Python/JSON literal and compared by value.",
    "validator": "Checked by the server against required properties, not a fixed string.",
}


def _system_prompt(detail: dict, workdir: str) -> str:
    category = detail.get("category") or ""
    fmt = detail.get("answer_format", "exact")
    return (
        "You are solving one tile of Agent Jeopardy using tools. You cannot "
        "solve this from memory or by guessing -- the data is real and "
        f"lives on disk at {workdir}.\n\n"
        f"Category: {category or 'unknown'}. {CATEGORY_HINTS.get(category, '')}\n\n"
        f"Answer checking: {fmt}. {_FORMAT_HINTS.get(fmt, '')}\n\n"
        "Rules:\n"
        "- Use run_python and http_request to gather ground truth. Never "
        "answer from recall.\n"
        "- Before finalizing, verify: re-derive the answer a second, "
        "independent way if the format allows (different logic, a "
        "count/checksum, re-parsing from a different angle).\n"
        "- When you have the final value, print it as the LAST stdout line "
        "of a run_python call in the exact form `ANSWER: <value>` -- make "
        "sure that line holds exactly the right string/number/literal and "
        "nothing else appended.\n"
        "- Then call submit_answer to confirm. You do not retype the answer "
        "there; it is pulled from your ANSWER line automatically.\n"
        f"- You have at most {MAX_TURNS} tool calls total. Work efficiently; "
        "don't re-explore what you've already established."
    )


def _run_tool(tu, task_id: str, workdir: str, state: dict) -> dict:
    """Execute one tool_use block, return its tool_result content dict."""
    if tu.name == "run_python":
        code = tu.input.get("code", "")
        timeout = min(int(tu.input.get("timeout") or 20), 60)
        result = tools.run_python(code, workdir, timeout=timeout)
        m = ANSWER_RE.findall(result.get("stdout", "") or "")
        if m:
            state["answer"] = m[-1].strip()
        return {"type": "tool_result", "tool_use_id": tu.id,
                "content": json.dumps(result)[:8000]}

    if tu.name == "http_request":
        result = tools.http_request(
            task_id, tu.input.get("method", "GET"), tu.input.get("url", ""),
            headers=tu.input.get("headers"), params=tu.input.get("params"),
            data=tu.input.get("data"), json_body=tu.input.get("json_body"),
        )
        return {"type": "tool_result", "tool_use_id": tu.id,
                "content": json.dumps(result)[:8000]}

    if tu.name == "submit_answer":
        if state.get("answer") is None:
            return {"type": "tool_result", "tool_use_id": tu.id, "is_error": True,
                    "content": ("No ANSWER: line captured yet. Print one via "
                                "run_python (as the last stdout line) before "
                                "calling submit_answer.")}
        state["done"] = True
        return {"type": "tool_result", "tool_use_id": tu.id, "content": "acknowledged"}

    return {"type": "tool_result", "tool_use_id": tu.id, "is_error": True,
            "content": f"unknown tool {tu.name}"}


def solve_tile(task_id: str, verbose: bool = False) -> tuple[str | None, dict]:
    """Attempt one tile end-to-end. Returns (answer_or_None, task_detail).

    Never raises jp.TileUnavailable/AuthError itself for the "no answer"
    case -- those propagate to the caller, which knows how to route them
    (skip vs. fatal).
    """
    detail = jp.task(task_id)
    workdir = jp.workdir(task_id)
    names = jp.fetch_files(task_id, detail, workdir)

    client = jp.anthropic_client()
    system = _system_prompt(detail, str(workdir))
    messages = [{
        "role": "user",
        "content": (
            f"{detail.get('prompt', '')}\n\n"
            f"Files available in {workdir}: {names or 'none'}"
        ),
    }]
    if verbose:
        jp.log(f"{task_id} system:\n{system}\n---\nuser:\n{messages[0]['content']}\n---")

    state: dict = {"answer": None, "done": False}

    for _turn in range(MAX_TURNS):
        resp = client.messages.create(
            model=jp.MODEL, max_tokens=2048,
            system=system, tools=tools.TOOLS, messages=messages,
        )
        messages.append({"role": "assistant", "content": resp.content})

        tool_uses = [b for b in resp.content if b.type == "tool_use"]
        if not tool_uses:
            if verbose:
                text = "".join(b.text for b in resp.content if b.type == "text")
                jp.log(f"{task_id} model (no tool call): {text[:300]}")
            messages.append({"role": "user", "content": (
                "Use a tool. If you already have the answer, run it through "
                "run_python with an ANSWER: line, then call submit_answer.")})
            continue

        tool_results = [_run_tool(tu, task_id, str(workdir), state) for tu in tool_uses]
        if verbose:
            for tu, tr in zip(tool_uses, tool_results):
                jp.log(f"{task_id} tool {tu.name}({tu.input}) -> "
                       f"{str(tr.get('content'))[:300]}")

        messages.append({"role": "user", "content": tool_results})
        if state["done"]:
            return state["answer"], detail

    if verbose:
        jp.log(f"{task_id}: MAX_TURNS hit, answer so far: {state['answer']!r}")
    return state["answer"], detail
