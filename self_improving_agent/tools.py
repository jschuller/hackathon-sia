"""
Custom tools for the Self-Improving Agent.
Experience memory + improvement tracking.
"""

import fcntl
import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

# Persistent experience store (survives agent restarts during hackathon)
_MEMORY_FILE = Path(__file__).parent.parent / "experience_memory.json"
_EXPERIENCE_MEMORY: list[dict] = []


def _read_memory() -> list[dict]:
    """Read the experience memory from disk ([] when there is none yet)."""
    if _MEMORY_FILE.exists():
        return json.loads(_MEMORY_FILE.read_text())
    return []


def _load_memory() -> None:
    """Refresh the in-memory copy from disk."""
    global _EXPERIENCE_MEMORY
    _EXPERIENCE_MEMORY = _read_memory()


@contextmanager
def _memory_lock() -> Iterator[None]:
    """Hold an exclusive lock for a read-modify-write of the memory file.

    The lock is a sibling file, so it survives the os.replace of the data file.
    Other processes (a second adk worker, the eval) block here instead of
    writing over each other.
    """
    lock_path = _MEMORY_FILE.with_name(_MEMORY_FILE.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _write_memory(memory: list[dict]) -> None:
    """Replace the memory file atomically: write a temp file, then os.replace.

    A reader sees the old file or the new one, never a half-written one.
    """
    _MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=_MEMORY_FILE.parent, prefix=f".{_MEMORY_FILE.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as tmp:
            json.dump(memory, tmp, indent=2)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_name, _MEMORY_FILE)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def store_experience(category: str, resolution: str, score: float) -> dict:
    """Store a successful resolution pattern for future retrieval.

    Call this after a resolution scores >= 0.85 to save the pattern.

    Args:
        category: Incident category (e.g. 'cpu', 'memory', 'disk', 'network', 'ssl')
        resolution: The resolution steps that worked well
        score: Quality score from the critic evaluation (0.0 to 1.0)

    Returns:
        Confirmation with total stored experiences.
    """
    global _EXPERIENCE_MEMORY
    with _memory_lock():
        # Re-read under the lock: another process may have stored since our last read.
        memory = _read_memory()
        entry = {
            "id": max((e.get("id", 0) for e in memory), default=0) + 1,
            "timestamp": datetime.now().isoformat(),
            "category": category.lower().strip(),
            "resolution": resolution,
            "score": round(score, 3),
        }
        memory.append(entry)
        _write_memory(memory)
    _EXPERIENCE_MEMORY = memory
    return {
        "status": "stored",
        "experience_id": entry["id"],
        "total_experiences": len(memory),
    }


def retrieve_experiences(category: str, top_k: int = 3) -> dict:
    """Retrieve past successful resolutions for a given incident category.

    Use this BEFORE proposing a resolution to learn from past successes.

    Args:
        category: Incident category to search for (e.g. 'cpu', 'memory')
        top_k: Max number of top-scoring experiences to return

    Returns:
        List of past experiences sorted by score (best first).
    """
    _load_memory()  # refresh from disk
    cat = category.lower().strip()
    relevant = [
        e for e in _EXPERIENCE_MEMORY
        if cat in e["category"]
    ]
    relevant.sort(key=lambda x: x["score"], reverse=True)
    return {
        "category_searched": cat,
        "experiences": relevant[:top_k],
        "total_matching": len(relevant),
        "total_stored": len(_EXPERIENCE_MEMORY),
    }


def get_improvement_stats() -> dict:
    """Get statistics showing how the agent has improved over time.

    Returns:
        Summary stats: total resolutions, average score, improvement trend.
    """
    _load_memory()
    if not _EXPERIENCE_MEMORY:
        return {"message": "No experiences stored yet.", "count": 0}

    scores = [e["score"] for e in _EXPERIENCE_MEMORY]
    categories = {}
    for e in _EXPERIENCE_MEMORY:
        cat = e["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(e["score"])

    return {
        "total_resolutions": len(scores),
        "overall_average_score": round(sum(scores) / len(scores), 3),
        "best_score": max(scores),
        "latest_score": scores[-1],
        "improvement_first_to_last": round(scores[-1] - scores[0], 3),
        "by_category": {
            cat: {
                "count": len(s),
                "avg": round(sum(s) / len(s), 3),
            }
            for cat, s in categories.items()
        },
    }


def get_experience_timeline() -> list[dict]:
    """Get experience data formatted for charting.

    Returns a list of entries with timestamp, score, category, and
    cumulative average — suitable for Streamlit dashboard charts.

    Returns:
        List of timeline entries with running averages.
    """
    _load_memory()
    timeline = []
    running_sum = 0.0
    for i, e in enumerate(_EXPERIENCE_MEMORY, 1):
        running_sum += e["score"]
        timeline.append({
            "index": i,
            "timestamp": e["timestamp"],
            "category": e["category"],
            "score": e["score"],
            "cumulative_avg": round(running_sum / i, 3),
        })
    return timeline


def clear_experience_memory() -> dict:
    """Reset all stored experiences. Use for demo resets.

    Returns:
        Confirmation that memory was cleared.
    """
    global _EXPERIENCE_MEMORY
    with _memory_lock():
        _write_memory([])
    _EXPERIENCE_MEMORY = []
    return {"status": "cleared", "total_experiences": 0}
