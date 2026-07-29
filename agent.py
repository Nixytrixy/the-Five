"""The tool-use loop for one Agent Jeopardy tile."""
from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import jeopardy as jp
import tools

MAX_TURNS = 14
# The message history is re-sent in full every turn, so fed-back tool output is
# the biggest per-tile lever under the shared 95k tokens/min limit. Trim it; the
# tiny captured_answer field is kept whole (placed first so truncation can't eat
# it). Both env-tunable for calibration.
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "1536"))
TOOL_RESULT_CHARS = int(os.environ.get("TOOL_RESULT_CHARS", "3000"))
ANSWER_RE = re.compile(r"^ANSWER:\s*(.*?)\s*$", re.MULTILINE)

CATEGORY_MAX_TURNS: dict[str, int] = {
    "Needle in the Haystack": 8,
    "Ship It": 8,
    "Cryptic": 10,
    "Heavy Compute": 14,
    "Ancient Scrolls": 10,
    "The Dark Web": 14,
}

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
        f"Category: {category or 'unknown'}. "
        f"{CATEGORY_HINTS.get(category, '')}\n\n"
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
        "- The harness captures the ANSWER line programmatically. Never "
        "retype the answer into another tool call.\n"
        "- Once the ANSWER line is printed, call submit_answer. The harness "
        "uses the captured string, so you do not type the answer there.\n"
        f"- You have at most {MAX_TURNS} tool calls total. Work efficiently; "
        "don't re-explore what you've already established."
    )


def _capture_answer(stdout: str) -> str | None:
    """Capture the final ANSWER line exactly once, in Python."""
    matches = ANSWER_RE.findall(stdout or "")
    return matches[-1] if matches else None


def _run_tools_parallel(
    tool_uses,
    task_id: str,
    workdir: str,
    state: dict,
) -> list[dict]:
    """Run non-submit tools concurrently; submit_answer always runs last."""
    non_submit = [tu for tu in tool_uses if tu.name != "submit_answer"]
    submits = [tu for tu in tool_uses if tu.name == "submit_answer"]
    result_map: dict[str, dict] = {}

    if non_submit:
        with ThreadPoolExecutor(max_workers=len(non_submit)) as ex:
            fs = {
                ex.submit(_run_tool, tu, task_id, workdir, state): tu.id
                for tu in non_submit
            }
            for fut in as_completed(fs):
                result_map[fs[fut]] = fut.result()

    for tu in submits:
        result_map[tu.id] = _run_tool(tu, task_id, workdir, state)

    return [result_map[tu.id] for tu in tool_uses]


def _run_tool(tu, task_id: str, workdir: str, state: dict) -> dict:
    if tu.name == "run_python":
        code = tu.input.get("code", "")
        timeout = min(int(tu.input.get("timeout") or 20), 60)
        result = tools.run_python(code, workdir, timeout=timeout)

        stdout = result.get("stdout", "") or ""
        answer = _capture_answer(stdout)

        if answer is not None:
            state["answer"] = answer

        # captured_answer first so a truncated stdout can't push it out of the
        # serialized blob -- the model must always see what will be submitted.
        payload = {
            "captured_answer": answer,
            **result,
        }

        return {
            "type": "tool_result",
            "tool_use_id": tu.id,
            "content": json.dumps(payload)[:TOOL_RESULT_CHARS],
        }

    if tu.name == "http_request":
        result = tools.http_request(
            task_id,
            tu.input.get("method", "GET"),
            tu.input.get("url", ""),
            headers=tu.input.get("headers"),
            params=tu.input.get("params"),
            data=tu.input.get("data"),
            json_body=tu.input.get("json_body"),
        )
        return {
            "type": "tool_result",
            "tool_use_id": tu.id,
            "content": json.dumps(result)[:TOOL_RESULT_CHARS],
        }

    if tu.name == "submit_answer":
        if state.get("answer") is None:
            return {
                "type": "tool_result",
                "tool_use_id": tu.id,
                "is_error": True,
                "content": (
                    "No ANSWER: line captured yet. Print one via "
                    "run_python (as the last stdout line) before calling "
                    "submit_answer."
                ),
            }

        state["done"] = True
        return {
            "type": "tool_result",
            "tool_use_id": tu.id,
            "content": "acknowledged",
        }

    return {
        "type": "tool_result",
        "tool_use_id": tu.id,
        "is_error": True,
        "content": f"unknown tool {tu.name}",
    }


def solve_tile(
    task_id: str,
    verbose: bool = False,
    is_open=None,
    prefetched: tuple[dict, object, list[str]] | None = None,
) -> tuple[str | None, dict]:
    """Attempt one tile end-to-end using already-prefetched inputs when given."""
    if prefetched is None:
        detail = jp.task(task_id)
        workdir = jp.workdir(task_id)
        names = jp.fetch_files(task_id, detail, workdir)
    else:
        detail, workdir, names = prefetched

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
        jp.log(
            f"{task_id} system:\n{system}\n---\n"
            f"user:\n{messages[0]['content']}\n---"
        )

    state: dict = {"answer": None, "done": False}
    category = detail.get("category", "")
    turns = CATEGORY_MAX_TURNS.get(category, MAX_TURNS)

    for _turn in range(turns):
        if is_open is not None and not is_open(task_id):
            jp.log(f"{task_id}: claimed by another team, stopping")
            return None, detail

        resp = client.messages.create(
            model=jp.MODEL,
            max_tokens=MAX_TOKENS,
            system=system,
            tools=tools.TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": resp.content})

        tool_uses = [
            b for b in resp.content
            if b.type == "tool_use"
        ]

        if not tool_uses:
            if verbose:
                text = "".join(
                    b.text for b in resp.content
                    if b.type == "text"
                )
                jp.log(
                    f"{task_id} model (no tool call): {text[:300]}"
                )

            messages.append({
                "role": "user",
                "content": (
                    "Use a tool. If you already have the answer, run it "
                    "through run_python with an ANSWER: line, then call "
                    "submit_answer."
                ),
            })
            continue

        tool_results = _run_tools_parallel(
            tool_uses,
            task_id,
            str(workdir),
            state,
        )

        if verbose:
            for tu, tr in zip(tool_uses, tool_results):
                jp.log(
                    f"{task_id} tool {tu.name}({tu.input}) -> "
                    f"{str(tr.get('content'))[:300]}"
                )

        messages.append({
            "role": "user",
            "content": tool_results,
        })

        # The exact answer is captured by Python. submit_answer is merely
        # the model's explicit verification/finish signal.
        if state["done"]:
            return state["answer"], detail

    if verbose:
        jp.log(
            f"{task_id}: MAX_TURNS hit, answer so far: "
            f"{state['answer']!r}"
        )

    return state["answer"], detail