"""
Custom tools for the Self-Improving Agent.
Experience memory + improvement tracking.
"""

import json
from datetime import datetime
from pathlib import Path

# Persistent experience store (survives agent restarts during hackathon)
_MEMORY_FILE = Path(__file__).parent.parent / "experience_memory.json"
_EXPERIENCE_MEMORY: list[dict] = []


def _load_memory():
    """Load experience memory from disk."""
    global _EXPERIENCE_MEMORY
    if _MEMORY_FILE.exists():
        _EXPERIENCE_MEMORY = json.loads(_MEMORY_FILE.read_text())


def _save_memory():
    """Persist experience memory to disk."""
    _MEMORY_FILE.write_text(json.dumps(_EXPERIENCE_MEMORY, indent=2))


# Load on import
_load_memory()


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
    entry = {
        "id": len(_EXPERIENCE_MEMORY) + 1,
        "timestamp": datetime.now().isoformat(),
        "category": category.lower().strip(),
        "resolution": resolution,
        "score": round(score, 3),
    }
    _EXPERIENCE_MEMORY.append(entry)
    _save_memory()
    return {
        "status": "stored",
        "experience_id": entry["id"],
        "total_experiences": len(_EXPERIENCE_MEMORY),
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
    _EXPERIENCE_MEMORY = []
    _save_memory()
    return {"status": "cleared", "total_experiences": 0}
