"""The experience memory keeps every writer's entries and never leaves a partial file."""

import json
import multiprocessing
from pathlib import Path

import pytest

from self_improving_agent import tools


def _stored(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def test_store_rereads_the_file_before_writing() -> None:
    # Another process stores between two of our writes; our stale copy must not win.
    tools.store_experience("cpu", "restart the service", 0.9)
    other = _stored(tools._MEMORY_FILE) + [
        {"id": 2, "timestamp": "t", "category": "disk", "resolution": "rotate logs", "score": 0.88}
    ]
    tools._MEMORY_FILE.write_text(json.dumps(other))

    result = tools.store_experience("memory", "raise the heap limit", 0.87)

    entries = _stored(tools._MEMORY_FILE)
    assert [e["resolution"] for e in entries] == [
        "restart the service",
        "rotate logs",
        "raise the heap limit",
    ]
    assert result == {"status": "stored", "experience_id": 3, "total_experiences": 3}


def test_ids_stay_unique_after_a_gap() -> None:
    tools._MEMORY_FILE.write_text(json.dumps([
        {"id": 7, "timestamp": "t", "category": "cpu", "resolution": "r", "score": 0.9}
    ]))

    assert tools.store_experience("cpu", "again", 0.9)["experience_id"] == 8


def test_write_is_atomic(monkeypatch: pytest.MonkeyPatch) -> None:
    tools.store_experience("cpu", "first", 0.9)
    before = tools._MEMORY_FILE.read_text()

    def fail(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(tools.json, "dump", fail)
    with pytest.raises(OSError):
        tools.store_experience("cpu", "second", 0.9)

    assert tools._MEMORY_FILE.read_text() == before
    leftovers = [p.name for p in tools._MEMORY_FILE.parent.iterdir() if p.suffix == ".tmp"]
    assert leftovers == []


def test_clear_empties_the_file() -> None:
    tools.store_experience("cpu", "first", 0.9)

    tools.clear_experience_memory()

    assert _stored(tools._MEMORY_FILE) == []


def _writer(memory_file: str, name: str, count: int, start: "multiprocessing.synchronize.Event") -> None:
    tools._MEMORY_FILE = Path(memory_file)
    start.wait()
    for i in range(count):
        tools.store_experience("cpu", f"{name}-{i}", 0.9)


def test_two_writer_processes_lose_no_entry() -> None:
    ctx = multiprocessing.get_context("spawn")
    start = ctx.Event()
    count = 25
    procs = [
        ctx.Process(target=_writer, args=(str(tools._MEMORY_FILE), name, count, start))
        for name in ("a", "b")
    ]
    for proc in procs:
        proc.start()
    start.set()
    for proc in procs:
        proc.join(timeout=60)
        assert proc.exitcode == 0

    entries = _stored(tools._MEMORY_FILE)
    resolutions = {e["resolution"] for e in entries}
    assert resolutions == {f"{name}-{i}" for name in ("a", "b") for i in range(count)}
    assert len({e["id"] for e in entries}) == 2 * count
