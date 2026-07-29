"""Agent Jeopardy -- concurrent, prefetching runner.

Optimisations:
- Flatten every cell's open_ids into independent tile work.
- Prefetch task metadata + files ahead of model workers.
- Capture ANSWER: lines in agent.py and pass the exact captured string
  directly to the single submission gate.
- Keep submission serialized at the server's 1-per-3-seconds limit.
- Retry incorrect/locked-out tiles with exponential backoff.

Environment:
  VERBOSE=1
  TASK_FILTER=PR-N4,PR-C5
  MAX_TILES=0
  MAX_WORKERS=8
  PREFETCH_WORKERS=8
  PREFETCH_AHEAD=32
  POLL_INTERVAL=2
"""
from __future__ import annotations

import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor

import agent
import jeopardy as jp

VERBOSE = os.environ.get("VERBOSE") == "1"
TASK_FILTER = [
    t.strip()
    for t in os.environ.get("TASK_FILTER", "").split(",")
    if t.strip()
]
MAX_TILES = int(os.environ.get("MAX_TILES", "0"))
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "8"))
PREFETCH_WORKERS = int(os.environ.get("PREFETCH_WORKERS", "8"))
PREFETCH_AHEAD = int(os.environ.get("PREFETCH_AHEAD", "32"))
POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL", "2"))
DISPATCH_INTERVAL = float(os.environ.get("DISPATCH_INTERVAL", "0.1"))
SUBMIT_MIN_GAP = float(os.environ.get("SUBMIT_MIN_GAP", "3.1"))

_submit_lock = threading.Lock()
_last_submit = 0.0

_state_lock = threading.Lock()
_solved: set[str] = set()
_inflight: set[str] = set()
_cooldown_until: dict[str, float] = {}
_miss_count: dict[str, int] = {}

_board_lock = threading.Lock()
_board_cache: dict | None = None

_prefetch_lock = threading.Lock()
_prefetched: dict[str, tuple[dict, object, list[str]]] = {}
_prefetching: set[str] = set()


def _rate_limited_submit(task_id: str, answer: str) -> dict:
    global _last_submit
    with _submit_lock:
        wait = SUBMIT_MIN_GAP - (time.monotonic() - _last_submit)
        if wait > 0:
            time.sleep(wait)
        result = jp.submit(task_id, answer)
        _last_submit = time.monotonic()
        return result


def _backoff_seconds(misses: int) -> float:
    return min(30 * (2 ** max(0, misses - 1)), 8 * 60)


def _flatten_open_ids(b: dict) -> list[dict]:
    """Flatten every open_ids entry into one independent tile record."""
    live = jp.live_board(b)
    cells = b.get("boards", {}).get(live, [])
    tiles: list[dict] = []

    for cell in cells:
        category = cell.get("category")
        points = cell.get("points")
        for task_id in cell.get("open_ids") or []:
            tiles.append({
                "id": task_id,
                "category": category,
                "points": points,
            })
    return tiles


def _is_tile_open(task_id: str) -> bool:
    """Check the cached board without doing another network request."""
    with _board_lock:
        b = _board_cache

    if not b:
        return True

    live = jp.live_board(b)
    for cell in b.get("boards", {}).get(live, []):
        if task_id in (cell.get("open_ids") or []):
            return True
    return False


def _prefetch_tile(task_id: str) -> None:
    """Fetch metadata and all files before the model worker needs them."""
    try:
        detail = jp.task(task_id)
        workdir = jp.workdir(task_id)
        names = jp.fetch_files(task_id, detail, workdir)

        with _prefetch_lock:
            _prefetched[task_id] = (detail, workdir, names)
            _prefetching.discard(task_id)

    except jp.TileUnavailable:
        with _prefetch_lock:
            _prefetching.discard(task_id)

    except jp.AuthError:
        with _prefetch_lock:
            _prefetching.discard(task_id)
        raise

    except Exception as e:
        jp.log(f"{task_id}: prefetch failed -- {e!r}")
        with _prefetch_lock:
            _prefetching.discard(task_id)


CATEGORY_SPEED: dict[str, float] = {
    "Needle in the Haystack": 1.2,
    "Ship It": 1.2,
    "Cryptic": 1.0,
    "Heavy Compute": 0.9,
    "Ancient Scrolls": 0.9,
    "The Dark Web": 0.7,
}


def _eligible_tiles(b: dict) -> list[dict]:
    now = time.monotonic()

    with _state_lock:
        solved = set(_solved)
        inflight = set(_inflight)
        cooldowns = dict(_cooldown_until)

    with _prefetch_lock:
        ready = set(_prefetched)
        prefetching = set(_prefetching)

    wanted = set(TASK_FILTER) if TASK_FILTER else None

    return [
        t for t in _flatten_open_ids(b)
        if t["id"] not in solved
        and t["id"] not in inflight
        and t["id"] not in ready
        and t["id"] not in prefetching
        and cooldowns.get(t["id"], 0) <= now
        and (wanted is None or t["id"] in wanted)
    ]


def pick_prefetch(b: dict, limit: int) -> list[str]:
    """Fill a wide prefetch buffer from ALL open_ids, not just card faces."""
    candidates = _eligible_tiles(b)
    candidates.sort(
        key=lambda t: -t.get("points", 0)
        * CATEGORY_SPEED.get(t.get("category", ""), 1.0)
    )
    return [t["id"] for t in candidates[:limit]]


def pick_ready(b: dict, slots: int) -> list[str]:
    """Choose already-prefetched tiles for model workers."""
    now = time.monotonic()

    with _state_lock:
        solved = set(_solved)
        inflight = set(_inflight)
        cooldowns = dict(_cooldown_until)

    with _prefetch_lock:
        ready = set(_prefetched)

    candidates = [
        t for t in _flatten_open_ids(b)
        if t["id"] in ready
        and t["id"] not in solved
        and t["id"] not in inflight
        and cooldowns.get(t["id"], 0) <= now
    ]

    candidates.sort(
        key=lambda t: -t.get("points", 0)
        * CATEGORY_SPEED.get(t.get("category", ""), 1.0)
    )

    picked = [t["id"] for t in candidates[:slots]]

    if MAX_TILES:
        with _state_lock:
            room = MAX_TILES - len(_solved) - len(_inflight)
        picked = picked[:max(room, 0)]

    return picked


def _release_after_failure(task_id: str, cooldown: bool = True) -> None:
    with _state_lock:
        if cooldown:
            misses = _miss_count.get(task_id, 0) + 1
            _miss_count[task_id] = misses
            _cooldown_until[task_id] = (
                time.monotonic() + _backoff_seconds(misses)
            )
        _inflight.discard(task_id)


def attempt(task_id: str) -> None:
    """Solve one prefetched tile and submit the exact captured answer."""
    with _prefetch_lock:
        prefetched = _prefetched.pop(task_id, None)

    if prefetched is None:
        jp.log(f"{task_id}: prefetched data missing")
        _release_after_failure(task_id)
        return

    try:
        answer, detail = agent.solve_tile(
            task_id,
            verbose=VERBOSE,
            is_open=_is_tile_open,
            prefetched=prefetched,
        )
    except jp.AuthError:
        with _state_lock:
            _inflight.discard(task_id)
        raise
    except jp.TileUnavailable as e:
        jp.log(f"{task_id}: unavailable -- {e}")
        _release_after_failure(task_id, cooldown=False)
        return
    except Exception as e:
        jp.log(f"{task_id}: solve blew up -- {e!r}")
        _release_after_failure(task_id)
        return

    if answer is None:
        category = detail.get("category") if detail else "?"
        jp.log(f"{task_id}: no ANSWER line captured (category={category})")
        _release_after_failure(task_id)
        return

    try:
        # answer came directly from stdout parsing in agent.py; no model
        # transcription is involved in this submission.
        result = _rate_limited_submit(task_id, answer)
    except jp.AuthError:
        with _state_lock:
            _inflight.discard(task_id)
        raise
    except Exception as e:
        jp.log(f"{task_id}: submit blew up -- {e!r}")
        with _state_lock:
            _inflight.discard(task_id)
        return

    outcome = result.get("result")
    jp.log(f"{task_id}: answered {answer[:60]!r} -> {outcome}")

    if outcome == "correct":
        with _state_lock:
            _solved.add(task_id)
            _inflight.discard(task_id)

    elif outcome in ("already_claimed", "voided", "unknown_task"):
        with _state_lock:
            _solved.add(task_id)
            _inflight.discard(task_id)

    elif outcome == "forbidden":
        with _state_lock:
            _inflight.discard(task_id)

    elif outcome in ("incorrect", "locked_out"):
        _release_after_failure(task_id)

    elif outcome == "rate_limited":
        retry = float(result.get("retry_in", 3) or 3)
        with _state_lock:
            _cooldown_until[task_id] = time.monotonic() + retry
            _inflight.discard(task_id)

    else:
        with _state_lock:
            _inflight.discard(task_id)


def _board_poller() -> None:
    global _board_cache
    while True:
        try:
            b = jp.board()
            with _board_lock:
                _board_cache = b
        except jp.AuthError:
            return
        except Exception as e:
            jp.log(f"board poll failed -- {e!r}")
        time.sleep(POLL_INTERVAL)


def main() -> None:
    global _board_cache

    jp.log(
        f"starting: MAX_WORKERS={MAX_WORKERS} "
        f"PREFETCH_WORKERS={PREFETCH_WORKERS} "
        f"PREFETCH_AHEAD={PREFETCH_AHEAD} "
        f"POLL_INTERVAL={POLL_INTERVAL}s "
        f"MAX_TILES={MAX_TILES or 'unlimited'} "
        f"TASK_FILTER={TASK_FILTER or 'none'}"
    )

    try:
        _board_cache = jp.board()
    except jp.AuthError:
        raise

    server_solved = set(
        (_board_cache.get("you") or {}).get("solved_ids") or []
    )
    if server_solved:
        with _state_lock:
            _solved.update(server_solved)
        jp.log(
            f"resuming: {len(server_solved)} tiles already solved, "
            "skipping them"
        )

    poller = threading.Thread(target=_board_poller, daemon=True)
    poller.start()

    _last_log = 0.0

    with (
        ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool,
        ThreadPoolExecutor(max_workers=PREFETCH_WORKERS) as prefetch_pool,
    ):
        futures: dict[Future, str] = {}

        while True:
            with _board_lock:
                b = _board_cache

            # Keep task metadata/files ready ahead of model workers.
            if b:
                with _prefetch_lock:
                    prefetching_count = len(_prefetching)
                    ready_count = len(_prefetched)

                need = max(
                    0,
                    PREFETCH_AHEAD - prefetching_count - ready_count,
                )

                if need:
                    for task_id in pick_prefetch(b, need):
                        with _prefetch_lock:
                            if (
                                task_id in _prefetched
                                or task_id in _prefetching
                            ):
                                continue
                            _prefetching.add(task_id)

                        prefetch_pool.submit(_prefetch_tile, task_id)

            # Clean up completed model workers.
            for f in [f for f in futures if f.done()]:
                task_id = futures.pop(f)
                exc = f.exception()
                if exc is not None:
                    if isinstance(exc, jp.AuthError):
                        raise exc
                    jp.log(f"{task_id}: worker crashed -- {exc!r}")

            with _state_lock:
                free = MAX_WORKERS - len(_inflight)

            if free > 0 and b:
                for task_id in pick_ready(b, free):
                    with _state_lock:
                        _inflight.add(task_id)

                    fut = pool.submit(attempt, task_id)
                    futures[fut] = task_id

            now = time.monotonic()
            if now - _last_log >= 10:
                with _state_lock:
                    cooling = sum(
                        1 for v in _cooldown_until.values()
                        if v > now
                    )
                    solved_count = len(_solved)
                    inflight_count = len(_inflight)

                with _prefetch_lock:
                    ready_count = len(_prefetched)
                    prefetching_count = len(_prefetching)

                jp.log(
                    f"solved={solved_count} "
                    f"inflight={inflight_count} "
                    f"prefetched={ready_count} "
                    f"prefetching={prefetching_count} "
                    f"cooling={cooling}"
                )

                if MAX_TILES and solved_count >= MAX_TILES:
                    jp.log(f"MAX_TILES={MAX_TILES} reached, stopping.")
                    return

                _last_log = now

            time.sleep(DISPATCH_INTERVAL)


if __name__ == "__main__":
    try:
        main()
    except jp.AuthError as e:
        raise SystemExit(f"[auth] {e}")
    except KeyboardInterrupt:
        jp.log("stopped")
