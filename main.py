"""Agent Jeopardy -- the real agent (see agent.py for the tool-use loop,
tools.py for run_python/http_request/submit_answer).

Long-running loop: poll the board, fan open tiles out across a worker pool,
solve each concurrently in its own thread via agent.solve_tile, and submit
through a single rate-limited gate so parallel solves never trip the
team-wide 1-submission-per-3-seconds limit. Wrong answers get an
exponential-backoff cooldown per tile (30s, 60s, 120s... capped at 8min,
matching the server's own lockout curve) so a miss doesn't permanently
abandon a tile, but also doesn't get hammered while it's still locked out.

Env knobs (see .env.example):
  VERBOSE        - passed straight to agent.solve_tile
  TASK_FILTER    - restrict to exactly these tile ids (comma-separated)
  MAX_TILES      - stop after this many solved (0 = unlimited; work the
                   whole board until the round ends)
  MAX_WORKERS    - concurrent tiles in flight (default 8; sandbox is 2 CPU /
                   2GB but these are I/O-bound waits on the model + HTTP, so
                   more workers than cores is fine)
  POLL_INTERVAL  - seconds between board polls (default 4)
"""
from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future

import agent
import jeopardy as jp

VERBOSE = os.environ.get("VERBOSE") == "1"
TASK_FILTER = [t.strip() for t in os.environ.get("TASK_FILTER", "").split(",") if t.strip()]
MAX_TILES = int(os.environ.get("MAX_TILES", "0"))          # 0 = no cap
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "8"))
POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL", "4"))
SUBMIT_MIN_GAP = 3.1                                        # server: 1 per 3s per team

_submit_lock = threading.Lock()
_last_submit = 0.0

_state_lock = threading.Lock()
_solved: set[str] = set()
_inflight: set[str] = set()
_cooldown_until: dict[str, float] = {}     # task_id -> time.monotonic() deadline
_miss_count: dict[str, int] = {}


def _rate_limited_submit(task_id: str, answer: str) -> dict:
    """Serialize every submit across all worker threads to >=3.1s apart."""
    global _last_submit
    with _submit_lock:
        wait = SUBMIT_MIN_GAP - (time.monotonic() - _last_submit)
        if wait > 0:
            time.sleep(wait)
        result = jp.submit(task_id, answer)
        _last_submit = time.monotonic()
        return result


def _backoff_seconds(misses: int) -> float:
    """30s, 60s, 120s, ... capped at 8min -- mirrors the server's own curve
    (README: 'doubling each miss, cap 8 min') so we don't hammer a tile
    that's still locked out, but also don't abandon it forever."""
    return min(30 * (2 ** max(0, misses - 1)), 8 * 60)


def attempt(task_id: str) -> None:
    """Solve one tile (blocking) and submit it. Runs in a worker thread."""
    try:
        answer, detail = agent.solve_tile(task_id, verbose=VERBOSE)
    except jp.AuthError:
        raise                                          # fatal; stop the whole agent
    except jp.TileUnavailable as e:
        jp.log(f"{task_id}: unavailable -- {e}")
        with _state_lock:
            _inflight.discard(task_id)
        return
    except Exception as e:  # noqa: BLE001
        jp.log(f"{task_id}: solve blew up -- {e!r}")
        answer, detail = None, {}
    finally:
        with _state_lock:
            _inflight.discard(task_id)

    if answer is None:
        jp.log(f"{task_id}: no answer captured (category="
               f"{detail.get('category') if detail else '?'})")
        with _state_lock:
            _miss_count[task_id] = _miss_count.get(task_id, 0) + 1
            _cooldown_until[task_id] = time.monotonic() + _backoff_seconds(_miss_count[task_id])
        return

    try:
        result = _rate_limited_submit(task_id, answer)
    except jp.AuthError:
        raise
    except Exception as e:  # noqa: BLE001
        jp.log(f"{task_id}: submit blew up -- {e!r}")
        return

    outcome = result.get("result")
    jp.log(f"{task_id}: answered {answer[:60]!r} -> {outcome}")

    if outcome == "correct":
        with _state_lock:
            _solved.add(task_id)
    elif outcome in ("already_claimed", "voided", "unknown_task"):
        with _state_lock:
            _solved.add(task_id)          # dead work either way; stop retrying
    elif outcome == "forbidden":
        jp.log("  a scored round is live -- only the HOSTED agent may submit "
               "(deploy via /api/agent/submit, or use the practice board)")
    elif outcome in ("incorrect", "locked_out"):
        with _state_lock:
            _miss_count[task_id] = _miss_count.get(task_id, 0) + 1
            _cooldown_until[task_id] = time.monotonic() + _backoff_seconds(_miss_count[task_id])
    elif outcome == "rate_limited":
        retry = float(result.get("retry_in", 3) or 3)
        with _state_lock:
            _cooldown_until[task_id] = time.monotonic() + retry
    # "wrong_phase": leave it uncooled -- it'll be filtered out again next
    # poll by open_tiles() as soon as the board isn't live for it.


def pick_next(b: dict, slots: int) -> list[str]:
    """Points-first, one tile per cell before a second tile in any cell --
    width first, matching the README's 'a serial agent watches the rest
    vanish' warning. Skips solved, in-flight, and cooling-down tiles."""
    now = time.monotonic()
    with _state_lock:
        solved, inflight, cooldowns = set(_solved), set(_inflight), dict(_cooldown_until)

    candidates = [
        t for t in jp.open_tiles(b)
        if t["id"] not in solved
        and t["id"] not in inflight
        and cooldowns.get(t["id"], 0) <= now
    ]
    if TASK_FILTER:
        wanted = set(TASK_FILTER)
        candidates = [t for t in candidates if t["id"] in wanted]

    seen_cells: set[tuple] = set()
    first_pass, rest = [], []
    for t in candidates:
        cell = (t.get("category"), t.get("points"))
        (first_pass if cell not in seen_cells else rest).append(t["id"])
        seen_cells.add(cell)

    picked = (first_pass + rest)[:slots]
    if MAX_TILES:
        with _state_lock:
            room = MAX_TILES - len(_solved) - len(_inflight)
        picked = picked[:max(room, 0)]
    return picked


def main() -> None:
    jp.log(f"starting: MAX_WORKERS={MAX_WORKERS} POLL_INTERVAL={POLL_INTERVAL}s "
           f"MAX_TILES={MAX_TILES or 'unlimited'} TASK_FILTER={TASK_FILTER or 'none'}")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures: dict[Future, str] = {}
        while True:
            try:
                b = jp.board()
            except jp.AuthError:
                raise
            except Exception as e:  # noqa: BLE001
                jp.log(f"board poll failed -- {e!r}")
                time.sleep(POLL_INTERVAL)
                continue

            with _state_lock:
                free = MAX_WORKERS - len(_inflight)
            if free > 0:
                for task_id in pick_next(b, free):
                    with _state_lock:
                        _inflight.add(task_id)
                    fut = pool.submit(attempt, task_id)
                    futures[fut] = task_id

            for f in [f for f in futures if f.done()]:
                task_id = futures.pop(f)
                exc = f.exception()
                if exc is not None:
                    if isinstance(exc, jp.AuthError):
                        raise exc
                    jp.log(f"{task_id}: worker crashed -- {exc!r}")

            with _state_lock:
                cooling = sum(1 for v in _cooldown_until.values() if v > time.monotonic())
                jp.log(f"solved={len(_solved)} inflight={len(_inflight)} cooling={cooling}")
                if MAX_TILES and len(_solved) >= MAX_TILES:
                    jp.log(f"MAX_TILES={MAX_TILES} reached, stopping.")
                    return

            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    try:
        main()
    except jp.AuthError as e:
        raise SystemExit(f"[auth] {e}")
    except KeyboardInterrupt:
        jp.log("stopped")
